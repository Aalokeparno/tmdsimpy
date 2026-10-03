"""Manifold-based steady-periodic elastic dry friction force evaluation.

The static force, history behavior, and prestress friction handling are
inherited unchanged from ``ElasticDryFriction2D``. Only the local
steady-periodic force history used by AFT is replaced: the normal force keeps
the same linear penalty law, and the tangential force is predicted from the
split polynomial manifold fitted by ``manifolds/elastic_dry_friction/main3.py``
and evaluated as in ``manifolds/elastic_dry_friction/main3_continue.py``
(prediction functions copied from ``manifolds/elastic_dry_friction/
utilities.py``).
"""

# Standard imports
import numpy as np
from os import PathLike
from pathlib import Path
from typing import Mapping

# JAX imports
import jax
import jax.numpy as jnp

# Decoractions for Partial compilation
from functools import partial

# Imports of Custom Functions and Classes
from ...utils import harmonic as hutils
from ...jax import harmonic_utils as jhutils # Jax version of harmonic utils
from .elastic_dry_fric_2d import ElasticDryFriction2D

jax.config.update("jax_enable_x64", True)


_POLY_REQUIRED_KEYS = (
    'split_coords_N',
    'split_coords_u',
    'coeffs1',
    'coeffs2',
    'coeffs3',
    'degree',
    'u_scaling',
    'f_scaling',
    'u0_ref',
    'kt',
    'mu',
)


class ElastManifoldFriction(ElasticDryFriction2D):
    """
    2D Elastic Dry Friction with a polynomial manifold replacing only the
    steady-periodic (AFT) tangential force evaluation.

    Parameters
    ----------
    Q : (Nnl, N) numpy.ndarray
        Matrix tranform from the `N` degrees of freedom (DOFs) of the system
        to the `Nnl` local nonlinear DOFs.
        `Nnl` should be even.
        Rows `0::2` correspond to local tangential DOFs.
        Rows `1::2` correspond to local normal DOFs.
    T : (N, Nnl) numpy.ndarray
        Matrix tranform from the local `Nnl` forces to the `N` global DOFs.
        Columns `0::2` correspond to local tangential forces.
        Columns `1::2` correspond to local normal forces.
    kt : float
        Tangential stiffness (used by the inherited static `force`).
    kn : float
        Normal stiffness (used by `force` and by AFT).
    mu : float
        Friction coefficient (used by the inherited static `force`).
    u0 : float, (Nnl,) numpy.ndarray, or None, optional
        Kept for interface compatibility with `ElasticDryFriction2D`. The
        manifold AFT is centered on each tangential cycle and does not use
        the slider initialization.
        The default is 0.
    poly_params : str, path-like, or mapping, optional
        Path to the `poly_params.npz` written by `save_poly_params` in
        `manifolds/elastic_dry_friction/main3.py`, or an already loaded
        mapping of the same arrays (e.g., from `load_poly_params`).
        The default is 'poly_params.npz' (relative to the working directory).

    See Also
    --------
    ElasticDryFriction2D :
        Parent class providing the static force, history, and prestress
        methods.

    Notes
    -----
    Tangential manifold parameters `kt`, `mu`, and `u0_ref` are read from
    `poly_params` because the polynomial was fit for those values. The
    constructor `kt` and `mu` are only used for the static `force` (prestress
    and linearization), so they should match the fitted values.

    Tangential forces are set to zero at instants with zero normal force and
    for sliders that do not move over the cycle.

    """

    def __init__(self, Q, T, kt, kn, mu, u0=0, *,
                 poly_params='poly_params.npz'):

        super().__init__(Q, T, kt, kn, mu, u0=u0)

        if isinstance(poly_params, (str, PathLike, Path)):
            source = str(poly_params)
            poly_params = load_poly_params(poly_params)
        elif isinstance(poly_params, Mapping):
            source = '<mapping>'
            poly_params = dict(poly_params)
        else:
            raise TypeError('poly_params must be a path or a parameter '
                            'mapping; received {}.'.format(
                                type(poly_params).__name__))

        self.poly_params = poly_params

        self.poly_arrays, self.poly_static = _unpack_poly_params(poly_params,
                                                                 source)

    def aft(self, U, w, h, Nt=128, tol=1e-7, calc_grad=True):
        """
        Implementation of the alternating frequency-time (AFT) method to
        extract harmonic nonlinear force coefficients.

        Parameters
        ----------
        U : (N*Nhc,) numpy.ndarray
            Displacement harmonic DOFs (global)
        w : float
            Frequency in rad/s. Needed in case there is velocity dependency.
        h : numpy.ndarray, sorted
            List of harmonics. The list corresponds to `Nhc` harmonic
            components.
        Nt : int power of 2, optional
            Number of time steps used in evaluation.
            The default is 128.
        tol : float, optional
            This argument is ignored, and is included for compatability of
            interface.
            The default is 1e-7.
        calc_grad : bool, optional
            Flag to calculate the gradients. If False, only `Fnl` is returned
            (in a tuple).
            The default is True.

        Returns
        -------
        Fnl : (N*Nhc,) numpy.ndarray
            Nonlinear hamonic force coefficients
        dFnldU : (N*Nhc,N*Nhc) numpy.ndarray
            Jacobian of `Fnl` with respect to `U`
            Only returned if `calc_grad` is True.
        dFnldw : (N*Nhc,) numpy.ndarray
            Jacobian of `Fnl` with respect to `w`
            Only returned if `calc_grad` is True.

        Notes
        -----
        The manifold directly evaluates the steady-state cycle, so no
        repeated hysteresis loops (and no tolerance) are needed.

        """

        #########################
        # Transform to Local Coordinates

        Nhc = 2*(h !=0).sum() + (h==0).sum() # Number of Harmonic Components
        Ulocal = (self.Q @ np.reshape(U, (self.Q.shape[1], Nhc), 'F')).T

        # Number of Nonlinear DOFs
        Ndnl = self.Q.shape[0]


        #########################
        # Conduct AFT in Local Coordinates with JAX
        Uwlocal = np.hstack((np.reshape(Ulocal.T, (Ndnl*Nhc,), 'F'), w))

        pars = np.array([self.kt, self.kn, self.mu])

        if calc_grad:
            # Case with gradient and local force
            dFdUwlocal, Flocal = _local_aft_eldry_grad(Uwlocal, pars,
                                                       self.poly_arrays,
                                                       self.poly_static,
                                                       tuple(h), Nt)
        else:
            Flocal = _local_aft_eldry(Uwlocal, pars, self.poly_arrays,
                                      self.poly_static, tuple(h), Nt)[0]


        #########################
        # Convert AFT to Global Coordinates

        # Reshape Flocal
        Flocal = jnp.reshape(Flocal, (Ndnl, Nhc), 'F')

        # Global coordinates
        Fnl = np.reshape(self.T @ Flocal, (U.shape[0],), 'F')

        if not calc_grad:
            return (Fnl,)

        dFnldU = np.kron(np.eye(Nhc), self.T) @ dFdUwlocal[:, :-1] \
                                                @ np.kron(np.eye(Nhc), self.Q)

        dFnldw = np.reshape(self.T @ \
                            np.reshape(dFdUwlocal[:, -1], (Ndnl, Nhc)), \
                            (U.shape[0],), 'F')

        return Fnl, dFnldU, dFnldw


    def local_force_history(self, unlt, unltdot, h, cst, unlth0, max_repeats=2,
                            atol=1e-10, rtol=1e-10):
        """
        Evaluate the local forces for steady-state harmonic motion used in AFT.

        Parameters
        ----------
        unlt : (Nt,Nnl) numpy.ndarray
            Local displacements, rows are different time instants and
            columns are different displacement DOFs.
        unltdot : (Nt,Nnl) numpy.ndarray
            Ignored here, included for compatibility of interface.
        h : 1D numpy.ndarray, sorted
            Ignored here, included for compatibility of interface.
        cst: (Nt,Nhc) numpy.ndarray
            Ignored here, included for compatibility of interface.
        unlth0 : (Nnl,) numpy.ndarray
            Ignored here, included for compatibility of interface.
        max_repeats : int, optional
            Ignored here, included for compatibility of interface.
            The default is 2.
        atol : float, optional
            Ignored here, included for compatibility of interface.
            The default is 1e-10.
        rtol : float, optional
            Ignored here, included for compatibility of interface.
            The default is 1e-10.

        Returns
        -------
        ft : (Nt,Nnl) numpy.ndarray
            Local nonlinear forces. First index is time instants, second index
            is which local nonlinear force DOF. This is returned as the first
            entry in a tuple.

        Notes
        -----
        This method is for the post-processing of force displacement
        relationships of the model from harmonic solutions. It calls the same
        private JAX function that AFT uses.

        """

        pars = np.array([self.kt, self.kn, self.mu])

        fxyn_t = _local_force_history(jnp.asarray(unlt, dtype=jnp.float64),
                                      pars, self.poly_arrays,
                                      self.poly_static)

        return (fxyn_t,)


###############################################################################
####### Polynomial Manifold Parameters                                  #######
###############################################################################

def load_poly_params(path: str) -> dict:
    """Load .npz into a dict of numpy arrays."""
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _unpack_poly_params(params, source='<mapping>'):
    """
    Validate a `poly_params` mapping and split it into JAX array and static
    (hashable) parts.

    Parameters
    ----------
    params : dict
        Arrays saved by `save_poly_params` in
        `manifolds/elastic_dry_friction/main3.py`.
    source : str, optional
        Description of where `params` came from for error messages.

    Returns
    -------
    poly_arrays : tuple of (K,) or (n_terms,) jax.numpy.ndarray (float64)
        `(split_coords_N, split_coords_u, coeffs1, coeffs2, coeffs3)`
    poly_static : tuple
        `(u_scaling, f_scaling, u0_ref, degree, kt, mu)` as Python scalars so
        they can be passed as static arguments to JIT compiled functions.

    """
    missing = [k for k in _POLY_REQUIRED_KEYS if k not in params]
    if missing:
        raise KeyError('Incomplete poly_params {}; missing keys: {}'.format(
                                                source, ', '.join(missing)))

    degree = int(np.asarray(params['degree']).reshape(()))
    n_terms = (degree + 1)*(degree + 2)//2

    split_coords_N = np.asarray(params['split_coords_N'], dtype=np.float64)
    split_coords_u = np.asarray(params['split_coords_u'], dtype=np.float64)

    if split_coords_N.ndim != 1 or split_coords_N.size == 0 \
        or split_coords_u.shape != split_coords_N.shape:
        raise ValueError('split_coords_N and split_coords_u in {} must be '
                         'nonempty 1D arrays of equal length.'.format(source))

    if np.any(np.diff(split_coords_N) < 0):
        raise ValueError('split_coords_N in {} must be sorted in increasing '
                         'order (required by jnp.interp).'.format(source))

    coeffs = []
    for name in ('coeffs1', 'coeffs2', 'coeffs3'):
        c = np.asarray(params[name], dtype=np.float64)
        if c.shape != (n_terms,):
            raise ValueError('{} in {} has shape {}; expected ({},) for '
                             'degree {}.'.format(name, source, c.shape,
                                                 n_terms, degree))
        coeffs.append(c)

    scalars = []
    for name in ('u_scaling', 'f_scaling', 'u0_ref', 'kt', 'mu'):
        val = float(np.asarray(params[name]).reshape(()))
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError('{} in {} must be finite and positive.'.format(
                                                                name, source))
        scalars.append(val)

    u_scaling, f_scaling, u0_ref, kt, mu = scalars

    poly_arrays = tuple(jnp.asarray(a, dtype=jnp.float64) for a in
                        (split_coords_N, split_coords_u, *coeffs))

    poly_static = (u_scaling, f_scaling, u0_ref, degree, kt, mu)

    return poly_arrays, poly_static


def predict_manifold_jax(u, N, params):
    """Tangential force history from the fitted manifold (one period of u, N)."""
    unlt = jnp.array(np.stack([u, u], axis=1), dtype=jnp.float64)  # (Nt, 2)
    ft = jnp.array(np.stack([N, N], axis=1), dtype=jnp.float64)    # (Nt, 2)

    vector_solver = jax.vmap(
        _local_force_history_manifold,
        in_axes=(1, 1, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(1)
    )
    ft = vector_solver(
        ft,
        unlt,
        jnp.array(params['split_coords_N'], dtype=jnp.float64),
        jnp.array(params['split_coords_u'], dtype=jnp.float64),
        jnp.array(params['coeffs1'], dtype=jnp.float64),
        jnp.array(params['coeffs2'], dtype=jnp.float64),
        jnp.array(params['coeffs3'], dtype=jnp.float64),
        float(params['u_scaling']),
        float(params['f_scaling']),
        float(params['u0_ref']),
        int(params['degree']),
        float(params['kt']),
        float(params['mu'])
    )
    return np.array(ft[:, 0])


###############################################################################
####### AFT Functions                                                   #######
###############################################################################

@partial(jax.jit, static_argnums=(3,4,5))
def _local_aft_eldry(Uwlocal, pars, poly_arrays, poly_static, htuple, Nt):
    """
    Conducts AFT in a functional form that can be used with JAX and JIT

    Parameters
    ----------
    Uwlocal : (Nnl*Nhc + 1,) numpy.ndarray
        Local nonlinear displacements for each harmonic component in order
        followed by the frequency in rad/s. Each harmonic component is listed
        in full before the next one.
    pars : (3,) numpy.ndarray
        Contains `kt = pars[0]` (tangential stiffness),
        `kn = pars[1]` (normal stiffness),
        and `mu = pars[2]` (friction coefficient).
    poly_arrays : tuple of jax.numpy.ndarray
        Array part of the manifold parameters, see `_unpack_poly_params`.
    poly_static : tuple
        Static part of the manifold parameters, see `_unpack_poly_params`.
    htuple : tuple of int, sorted
        tuple list of harmonics. Tuple is used so that it can be made static
    Nt : int, power of 2
        Number of AFT time steps to be used.

    Returns
    -------
    Flocal : (Nhc*Nnl,) numpy.ndarray
        Harmonic force coefficients locally, same format as U part of Uwlocal
    Flocal : (Nhc*Nnl,) numpy.ndarray
        Repeated return value to allow for JAX auto diff calculation.

    """

    ########################################
    #### Initialization

    # Size Calculation
    Nhc = hutils.Nhc(np.array(htuple))
    Ndnl = int(Uwlocal.shape[0] / Nhc)

    # Ulocal is (Nhc x Ndnl) - each column is the harmonic components for a
    # single nonlinear DOF.
    Ulocal = jnp.reshape(Uwlocal[:-1], (Ndnl, Nhc), 'F').T


    ########################################
    #### Displacements

    # Nonlinear displacements in time
    # Nt x Ndnl
    unlt = jhutils.time_series_deriv(Nt, htuple, Ulocal, 0) # Nt x Ndnl

    ft = _local_force_history(unlt, pars, poly_arrays, poly_static)

    # Convert back into frequency domain
    Flocal = jhutils.get_fourier_coeff(htuple, ft)

    # Flatten back to a 1D array
    Flocal = jnp.reshape(Flocal.T, (-1,), 'F')

    return Flocal,Flocal


@partial(jax.jit, static_argnums=(3,4,5))
def _local_aft_eldry_grad(Uwlocal, pars, poly_arrays, poly_static, htuple, Nt):
    """
    Private function that computes the gradient of AFT for manifold elastic
    dry friction.

    Parameters
    ----------
    Same as `_local_aft_eldry`.

    Returns
    -------
    J : (Nhc*Nnl,Nhc*Nnl+1) numpy.ndarray
        Derivative of `F` with respect to `Uwlocal`
    F : (Nhc*Nnl,) numpy.ndarray
        Repeated return value to allow for JAX auto diff calculation.

    """

    J,F = jax.jacfwd(_local_aft_eldry, has_aux=True)(Uwlocal, pars,
                                                     poly_arrays, poly_static,
                                                     htuple, Nt)
    return J,F


@partial(jax.jit, static_argnums=(3,))
def _local_force_history(unlt, pars, poly_arrays, poly_static):
    """
    Calculates the steady-state force history with the polynomial manifold.

    Parameters
    ----------
    unlt : (Nt,Nnl)
        Time history of displacements over one cycle. Rows are time
        instants, columns are local displacements for nonlinear evaluation.
        Columns `0::2` are tangential, columns `1::2` are normal.
    pars : (3,) numpy.ndarray
        Contains `kt = pars[0]`, `kn = pars[1]`, and `mu = pars[2]`.
        Only `kn` is used here.
    poly_arrays : tuple of jax.numpy.ndarray
        Array part of the manifold parameters, see `_unpack_poly_params`.
    poly_static : tuple
        Static part of the manifold parameters, see `_unpack_poly_params`.

    Returns
    -------
    fxyn_t : (Nt,Nnl) jax.numpy.ndarray
        History of total forces

    Notes
    -----
    Normal force is the same linear penalty as `ElasticDryFriction2D`.
    Tangential force of each slider is `_local_force_history_manifold`
    evaluated with that slider's normal force history.

    """
    Nt = unlt.shape[0]

    kn = pars[1]

    split_coords_N, split_coords_u, coeffs1, coeffs2, coeffs3 = poly_arrays
    u_scaling, f_scaling, u0_ref, degree, kt, mu = poly_static

    # Initialize force time memory
    ft = jnp.zeros_like(unlt)

    # Normal Force
    ft = ft.at[:, 1::2].set(jnp.maximum(unlt[:, 1::2]*kn, 0.0))

    # The manifold normalizes by the cycle amplitude, so a slider with no
    # motion is evaluated on a dummy cycle and then zeroed. Using `where` on
    # the input (not only output) keeps forces and gradients free of NaN.
    ut = unlt[:, 0::2]
    moving = (jnp.max(ut, axis=0) - jnp.min(ut, axis=0)) > 0.0

    dummy = u0_ref*jnp.cos(2*jnp.pi*jnp.arange(Nt)/Nt)
    ut_safe = jnp.where(moving, ut, dummy[:, None])

    vector_solver = jax.vmap(
        _local_force_history_manifold,
        in_axes=(1, 1, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(1)
    )

    ft_tan = vector_solver(ft[:, 1::2], ut_safe,
                           split_coords_N, split_coords_u,
                           coeffs1, coeffs2, coeffs3,
                           u_scaling, f_scaling,
                           u0_ref, degree, kt, mu)

    # No tangential force when stationary or out of contact (|ft| <= mu*fn)
    ft_tan = jnp.where(moving & (ft[:, 1::2] > 0.0), ft_tan, 0.0)

    ft = ft.at[:, 0::2].set(ft_tan)

    return ft


###############################################################################
####### Manifold Prediction (from manifolds/elastic_dry_friction/utilities.py)
###############################################################################

@partial(jax.jit, static_argnums=(1,))
def _find_first_n_extrema(u_rot, n=4):
    """
    Find positions of first n extrema in rotated signal.
    Returns fixed-size array padded with -1 sentinel.

    Parameters
    ----------
    u_rot : (Nt,) jax.numpy.ndarray
        Signal rotated so index 0 is global minimum.
    n : int (static)
        Number of extrema to find.

    Returns
    -------
    extrema_pos : (n,) jax.numpy.ndarray of int
        Positions of first n extrema, padded with -1.
    n_found : int scalar
        Number of extrema actually found (capped at n).
    """
    Nt = u_rot.shape[0]
    du_fwd = jnp.roll(u_rot, -1) - u_rot
    du_bwd = u_rot - jnp.roll(u_rot, 1)

    is_extremum = ((du_bwd > 0) & (du_fwd < 0)) | \
                  ((du_bwd < 0) & (du_fwd > 0))

    indices = jnp.arange(Nt)
    rank = jnp.cumsum(is_extremum)   # rank[i] = how many extrema up to and including i

    extrema_pos = jnp.full(n, -1, dtype=jnp.int32)
    for k in range(n):
        # Find first index where cumsum reaches k+1
        pos = jnp.where(rank == (k + 1), indices, Nt).min()
        extrema_pos = extrema_pos.at[k].set(
            jnp.where(pos < Nt, pos, -1).astype(jnp.int32)
        )

    n_found = jnp.minimum(jnp.sum(is_extremum).astype(jnp.int32), n)
    return extrema_pos, n_found


@partial(jax.jit, static_argnums=(2,))
def _build_poly_features_jax(X, Y, degree):
    """
    Build polynomial design matrix — JAX JIT compatible.
    """

    # Precompute powers
    x_powers = jnp.stack([X**i for i in range(degree + 1)], axis=1)
    y_powers = jnp.stack([Y**i for i in range(degree + 1)], axis=1)

    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append(x_powers[:, i] * y_powers[:, j])

    return jnp.stack(cols, axis=1)

@partial(jax.jit, static_argnums=(7,8,9,10,11,12))
def _local_force_history_manifold(N_t,u_t,
                                   split_coords_N, split_coords_u,
                                   coeffs1, coeffs2, coeffs3,
                                   u_scaling, f_scaling,
                                   u0_ref, degree, kt, mu):
    """
    Manifold-based local force history for AFT — fully JAX traceable.

    Parameters
    ----------
    N_t : (Nt,) jax.numpy.ndarray
        Normal force history.
    u_t : (Nt,) jax.numpy.ndarray
        Tangential displacement history. Must not be constant.
    split_coords_N : (K,) jax.numpy.ndarray
        Sorted split curve N values (pre-sorted outside JAX).
    split_coords_u : (K,) jax.numpy.ndarray
        Split curve u values.
    coeffs1 : (n_terms,) jax.numpy.ndarray
        Polynomial coefficients for left patch.
    coeffs2 : (n_terms,) jax.numpy.ndarray
        Polynomial coefficients for right patch.
    coeffs3 : (n_terms,) jax.numpy.ndarray
        Polynomial coefficients for top patch.
    u_scaling : float (static)
        Displacement normalization used in the polynomial fit.
    f_scaling : float (static)
        Force normalization used in the polynomial fit.
    u0_ref : float (static)
        Reference displacement amplitude for normalization.
    degree : int (static)
        Polynomial degree.
    kt : float (static)
        Tangential stiffness the manifold was fit for.
    mu : float (static)
        Friction coefficient the manifold was fit for.

    Returns
    -------
    f_tangential : (Nt,) jax.numpy.ndarray
        Tangential force history.

    Notes
    -----
    State machine with 4 states:
        0 = big loop increasing
        1 = big loop decreasing
        2 = small loop first half  (inc for flag=True,  dec for flag=False)
        3 = small loop second half (dec for flag=True,  inc for flag=False)

    Sign convention:
        Big loop:   velocity-based (+X if inc, -X if dec)
        Small loop: fixed by flag_second_is_global_max
                    flag=True  → always +X (is_inc=True)
                    flag=False → always -X (is_inc=False)

    Continuity enforced at:
        - Entry into small loop  (first point of small region)
        - Turning point of small loop (i_turning)
    No correction applied at exit of small loop.

    Same as ``manifolds/elastic_dry_friction/utilities.py`` except that the
    top patch polynomial is evaluated at no more than its fitted edge
    ``Y = split_coords_N[-1]`` (see ``eval_all_vec``).
    """
    Nt = u_t.shape[0]
    # u_t = unlt[:, 0]
    # N_t = ft[:, 0]
    indices = jnp.arange(Nt)

    # --------------------------------------------------
    # Step 1: Big loop normalization
    # --------------------------------------------------
    u_max = jnp.max(u_t)
    u_min = jnp.min(u_t)
    u_avg_big = 0.5 * (u_max + u_min)
    u_scale_big = (u_max - u_min) / 2.0 / u0_ref

    # --------------------------------------------------
    # Step 2: Rotate signal to start at global minimum
    #         and detect extrema
    # --------------------------------------------------
    i0 = jnp.argmin(u_t)
    u_rot = jnp.roll(u_t, -i0)
    N_rot = jnp.roll(N_t, -i0)

    extrema_pos_rot, n_found = _find_first_n_extrema(u_rot, n=4)

    # Unroll extrema positions back to original indices
    # extrema_pos = (extrema_pos_rot + i0) % Nt

    # --------------------------------------------------
    # Step 3: Determine small loop presence and structure
    # --------------------------------------------------
    has_small_loop = n_found >= 3

    i_global_max_rot = jnp.argmax(u_rot)
    flag_second_is_global_max = (extrema_pos_rot[1] == i_global_max_rot)

    # Small loop spans from i_small_start to i_small_end
    # Case B (flag=True):  ext[2] → ext[3]  (local_min → local_max)
    # Case C (flag=False): ext[1] → ext[2]  (local_max → local_min)
    i_small_start = jnp.where(flag_second_is_global_max, extrema_pos_rot[2], extrema_pos_rot[1])
    i_turning   = jnp.where(flag_second_is_global_max, extrema_pos_rot[3], extrema_pos_rot[2])
    u_ref_cross = jnp.where(flag_second_is_global_max, u_rot[extrema_pos_rot[2]], u_rot[extrema_pos_rot[1]])
    cross_condition = jnp.where(flag_second_is_global_max, u_rot <= u_ref_cross, u_rot >= u_ref_cross)
    search_start = jnp.where(flag_second_is_global_max, extrema_pos_rot[3], extrema_pos_rot[2])
    valid_mask = (indices > search_start) & cross_condition
    i_small_end = jnp.where(jnp.any(valid_mask), jnp.argmax(valid_mask), Nt - 1)

    # jax.debug.print("i_small_start = {}, u_small_start = {}",i_small_start, u_rot[i_small_start])
    # jax.debug.print("i_small_end = {}, u_small_end = {}",i_small_end, u_rot[i_small_end])
    # jax.debug.print("i_turning = {}, u_turning = {}",i_turning, u_rot[i_turning])

    small_1 = u_rot[i_small_start]
    small_2 = u_rot[i_turning]


    u_avg_small = 0.5 * (small_1 + small_2)
    u_scale_small = jnp.abs(small_1 - small_2) / 2.0 / u0_ref

    anchor = jnp.where(flag_second_is_global_max, u_max, u_min)

    u_max = jnp.maximum(jnp.maximum(small_1, small_2), anchor)
    u_min = jnp.minimum(jnp.minimum(small_1, small_2), anchor)

    u_avg_case2 = 0.5 * (u_max + u_min)
    u_scale_case2 = jnp.abs(u_max - u_min) / (2.0 * u0_ref)

    # --------------------------------------------------
    # Step 5: Region masks
    # --------------------------------------------------
    in_small_range = has_small_loop & \
                     (indices >= i_small_start) & \
                     (indices <= i_small_end)


    # --------------------------------------------------
    # Step 6: Vectorized manifold evaluation helper
    # --------------------------------------------------
    def eval_all_vec(u_t, N_t, is_inc_arr, avg, scale):

        """Evaluate manifold force at all Nt points simultaneously."""
        X_inc =  (u_t - avg) / scale
        X_dec = -(u_t - avg) / scale
        Y     =   N_t / scale

        X    = jnp.where(is_inc_arr, X_inc, X_dec)
        sign = jnp.where(is_inc_arr, 1.0, -1.0)

        N_max    = split_coords_N[-1]

        # Top patch (Y > N_max) is the fully stuck regime. Small amplitudes
        # give Y = N/scale far beyond the fitted range, so evaluate the
        # polynomial at the fitted edge instead of extrapolating in Y.
        Y_fit = jnp.minimum(Y, N_max)

        A = _build_poly_features_jax(X/u_scaling, Y_fit/f_scaling, degree)

        u_corner = jnp.interp(Y, split_coords_N, split_coords_u)
        cond3    = Y > N_max
        cond1    = (~cond3) & (X <= u_corner)   # left patch
        cond2    = (~cond3) & (X >  u_corner)   # right patch

        Z1 = A @ coeffs1 * f_scaling
        Z2 = A @ coeffs2 * f_scaling
        Z3 = A @ coeffs3 * f_scaling
        Z  = jnp.where(cond1, Z1, jnp.where(cond2, Z2, Z3))

        return sign * Z * scale, cond3

    # --------------------------------------------------
    # Step 7: Big loop forces — velocity-based sign
    # --------------------------------------------------
    du_rot  = jnp.roll(u_rot, -1) - u_rot
    f_big, _ = eval_all_vec(u_rot, N_rot, du_rot >= 0, u_avg_big, u_scale_big)

    # For delta calculation
    # f_big_extreme, _ = eval_all_vec(u_rot*0 + u_max, N_rot, jnp.ones_like(du_rot, dtype=bool), u_avg_big, u_scale_big)
    # u_bound = jnp.maximum(jnp.abs(u_max), jnp.abs(u_min))
    # N_shift_limit = u_bound * kt / mu
    # f_max, _ = eval_all_vec(jnp.atleast_1d(u_max), jnp.atleast_1d(N_shift_limit), jnp.atleast_1d(True), u_avg_big, u_scale_big)
    # delta = jnp.where(N_rot < N_shift_limit, N_rot*mu - f_big_extreme, 0.0)
    # delta = jnp.where(N_rot > N_shift_limit, u_bound* kt - f_max, delta) * jnp.sign(u_avg_big)

    # N_shift_limit = u_scale_big * kt / mu
    # delta = jnp.where(N_rot < N_shift_limit, N_rot*mu - kt*u_scale_big, 0.0)
    # delta = jnp.where(N_rot > N_shift_limit, kt*u_avg_big, delta) * jnp.sign(u_avg_big)
    N_lim_1 = u_scale_big * u0_ref * kt / mu
    N_lim_2 = (u_scale_big * u0_ref + jnp.abs(u_avg_big)) * kt / mu
    delta = jnp.where(N_rot > N_lim_1, N_rot*mu - kt*u_scale_big* u0_ref, 0.0) * jnp.sign(u_avg_big)
    delta = jnp.where(N_rot > N_lim_2, kt*u_avg_big, delta)


    # jax.debug.print("Nrot: max={}, min={}", jnp.max(N_rot), jnp.min(N_rot))
    # jax.debug.print("N_shift_limit={}", N_shift_limit)
    # jax.debug.print("delta1: max={}, min={}", jnp.max(N_rot*mu - f_big_extreme), jnp.min(N_rot*mu - f_big_extreme))
    # jax.debug.print("delta2 ={}", u_max * kt - f_max)
    # jax.debug.print("delta: max={}, min={}", jnp.max(delta), jnp.min(delta))


    f_small_case1, condX  = eval_all_vec(u_rot, N_rot, du_rot > 0, u_avg_small, u_scale_small)

    X0_inc = -(u_rot[i_small_start] - u_avg_big) / u_scale_big
    X0_dec = (u_rot[i_small_start] - u_avg_big) / u_scale_big
    X0 = jnp.where(flag_second_is_global_max, X0_inc, X0_dec)
    Y0     = N_rot[i_small_start] / u_scale_big
    N_max    = split_coords_N[-1]
    u_corner = jnp.interp(Y0, split_coords_N, split_coords_u)
    cond3    = Y0 > N_max
    cond2    = (~cond3) & (X0 >  u_corner)   # right patch
    sign = jnp.full_like(u_rot, jnp.where(flag_second_is_global_max, -1.0, 1.0))
    du_rot = jnp.where(in_small_range, sign, du_rot)
    f_all_case3, _ = eval_all_vec(u_rot, N_rot, du_rot > 0, u_avg_big, u_scale_big)

    false_array = jnp.full_like(condX, False)
    case2_ind = jnp.where(in_small_range, condX, false_array)
    f_small_case2, _  = eval_all_vec(u_rot, N_rot, du_rot < 0, u_avg_case2, u_scale_case2)
    f_small_case1 = jnp.where(case2_ind, f_small_case2, f_small_case1)

    # --------------------------------------------------
    # Step 10: Combine — use small loop forces in small
    #          region, big loop forces everywhere else
    # --------------------------------------------------
    f_tangential = jnp.where(in_small_range, f_small_case1, f_big)
    f_tangential = jnp.where(cond2, f_tangential, f_all_case3)
    # f_tangential = jnp.where(jnp.abs(u_avg_big/u_max)>1E-3, f_tangential + delta, f_tangential)
    f_tangential = f_big
    f_tangential = jnp.roll(f_tangential, i0)

    return f_tangential


__all__ = [
    'ElastManifoldFriction',
    'load_poly_params',
    'predict_manifold_jax',
]

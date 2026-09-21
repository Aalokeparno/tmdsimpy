"""Manifold-based steady-periodic rough-contact force evaluation.

The static force and history behavior are inherited unchanged from
``RoughContactFriction``.  Only the local steady-periodic force history used
by AFT is replaced by the fitted normal-GP and tangential-KNN models.
"""

from functools import partial
from os import PathLike
from pathlib import Path
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ....jax import harmonic_utils as jhutils
from .rough_contact import RoughContactFriction


_NORMAL_REQUIRED_KEYS = (
    "gp_X_train",
    "gp_alpha",
    "gp_kernel_constant",
    "gp_kernel_length_scale",
    "input_scaler_mean",
    "input_scaler_scale",
    "output_scaler_mean",
    "output_scaler_scale",
    "cutoff_un_amp",
    "cutoff_u_min",
)

_TANGENTIAL_DISPLACEMENT_AMPLITUDE = 1.0e-5
"""Tangential-displacement offset used to break trajectory symmetry.

This matches ``utamp`` in ``manifolds/rough_contact/mainb.py`` and
``manifolds/rough_contact/main_knn.py``. Unlike the older neural-network
surface archives, the fitted KNN surface archives do not store this value,
so it is hard-coded here rather than read from the tangential-model file.
"""

_TANGENTIAL_REQUIRED_KEYS = (
    "X_train",
    "Z_train",
    "n_neighbors",
    "distance_weighted",
    "xy_mean",
    "xy_scale",
    "z_mean",
    "z_scale",
    "scale_u",
    "scale_F",
)


class NormalGPParameters(NamedTuple):
    """Numerical normal-GP parameters passed through JAX as a pytree."""

    x_train: jax.Array
    alpha: jax.Array
    kernel_constant: jax.Array
    length_scale: jax.Array
    input_mean: jax.Array
    input_scale: jax.Array
    output_mean: jax.Array
    output_scale: jax.Array
    cutoff_un_amp: jax.Array
    cutoff_u_min: jax.Array


class TangentialNNParameters:
    """Numerical tangential-KNN parameters passed through JAX as a pytree.

    ``n_neighbors`` and ``distance_weighted`` are kept as static auxiliary
    pytree data rather than array leaves: ``jax.lax.top_k`` requires ``k`` to
    be a concrete Python integer and the distance-weighting branch is a
    Python-level ``if``, so both must stay outside of JAX tracing.
    """

    __slots__ = (
        "x_train",
        "z_train",
        "xy_mean",
        "xy_scale",
        "z_mean",
        "z_scale",
        "scale_u",
        "scale_F",
        "utamp",
        "n_neighbors",
        "distance_weighted",
    )

    def __init__(
        self,
        x_train,
        z_train,
        xy_mean,
        xy_scale,
        z_mean,
        z_scale,
        scale_u,
        scale_F,
        utamp,
        n_neighbors,
        distance_weighted,
    ):
        self.x_train = x_train
        self.z_train = z_train
        self.xy_mean = xy_mean
        self.xy_scale = xy_scale
        self.z_mean = z_mean
        self.z_scale = z_scale
        self.scale_u = scale_u
        self.scale_F = scale_F
        self.utamp = utamp
        self.n_neighbors = n_neighbors
        self.distance_weighted = distance_weighted

    def _tree_flatten(self):
        children = (
            self.x_train,
            self.z_train,
            self.xy_mean,
            self.xy_scale,
            self.z_mean,
            self.z_scale,
            self.scale_u,
            self.scale_F,
            self.utamp,
        )
        aux_data = (self.n_neighbors, self.distance_weighted)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        n_neighbors, distance_weighted = aux_data
        return cls(*children, n_neighbors, distance_weighted)


jax.tree_util.register_pytree_node(
    TangentialNNParameters,
    TangentialNNParameters._tree_flatten,
    TangentialNNParameters._tree_unflatten,
)


def _load_mapping(model, model_name):
    """Load a non-pickled archive or copy an already-loaded mapping."""
    if isinstance(model, (str, PathLike, Path)):
        path = Path(model)
        try:
            with np.load(path, allow_pickle=False) as archive:
                loaded = {name: archive[name].copy() for name in archive.files}
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"Could not load {model_name} model {path}: {error}"
            ) from error
        return loaded, str(path)
    if isinstance(model, Mapping):
        return dict(model), "<mapping>"
    raise TypeError(
        f"{model_name} model must be a path or a parameter mapping; "
        f"received {type(model).__name__}."
    )


def _require_keys(model, required, model_name, source):
    missing = [name for name in required if name not in model]
    if missing:
        raise KeyError(
            f"Incomplete {model_name} model {source}; missing keys: "
            + ", ".join(missing)
        )


def _numeric_array(model, name, source, shape=None, ndim=None):
    raw = np.asarray(model[name])
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"Model value {name!r} in {source} must be numerical.")
    value = np.asarray(raw, dtype=np.float64)
    if shape is not None and value.shape != shape:
        raise ValueError(
            f"Model value {name!r} in {source} has shape {value.shape}; "
            f"expected {shape}."
        )
    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"Model value {name!r} in {source} has {value.ndim} dimensions; "
            f"expected {ndim}."
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Model value {name!r} in {source} must be finite.")
    return value


def _scalar(model, name, source):
    value = np.asarray(model[name])
    if value.shape != ():
        raise ValueError(f"Model value {name!r} in {source} must be scalar.")
    return value.item()


def _positive_scalar(model, name, source):
    raw = np.asarray(model[name])
    if raw.size != 1 or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"Model value {name!r} in {source} must be a numerical scalar.")
    value = np.float64(raw.reshape(()).item())
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Model value {name!r} in {source} must be finite and positive."
        )
    return value


def load_normal_model(model):
    """Validate and convert normal-GP inference parameters to JAX arrays."""
    if isinstance(model, NormalGPParameters):
        return model
    values, source = _load_mapping(model, "normal-GP")
    _require_keys(values, _NORMAL_REQUIRED_KEYS, "normal-GP", source)

    if (
        "format_version" in values
        and int(_scalar(values, "format_version", source)) != 1
    ):
        raise ValueError(
            f"Unsupported normal-GP format version in {source}; expected 1."
        )
    if (
        "model_type" in values
        and str(_scalar(values, "model_type", source)) != "normal_load_gaussian_process"
    ):
        raise ValueError(f"Unsupported normal-GP model type in {source}.")

    x_train = _numeric_array(values, "gp_X_train", source, ndim=2)
    if x_train.shape[0] == 0 or x_train.shape[1] != 2:
        raise ValueError(
            f"gp_X_train in {source} must have nonempty shape (Ntrain, 2)."
        )
    alpha = _numeric_array(values, "gp_alpha", source, shape=(x_train.shape[0],))
    kernel_constant = _positive_scalar(values, "gp_kernel_constant", source)
    length_scale = _numeric_array(values, "gp_kernel_length_scale", source)
    if length_scale.shape not in ((), (1,), (2,)) or np.any(length_scale <= 0.0):
        raise ValueError(
            f"gp_kernel_length_scale in {source} must be positive with shape (), (1,), or (2,)."
        )
    length_scale = length_scale.reshape(-1)

    input_mean = _numeric_array(values, "input_scaler_mean", source, shape=(2,))
    input_scale = _numeric_array(values, "input_scaler_scale", source, shape=(2,))
    if np.any(input_scale <= 0.0):
        raise ValueError(f"input_scaler_scale in {source} must be positive.")
    output_mean = _numeric_array(values, "output_scaler_mean", source)
    output_scale = _numeric_array(values, "output_scaler_scale", source)
    if output_mean.size != 1 or output_scale.size != 1 or output_scale.item() <= 0.0:
        raise ValueError(
            f"Output scaler mean/scale in {source} must be length-one, with positive scale."
        )

    cutoff_un_amp = _numeric_array(values, "cutoff_un_amp", source, ndim=1)
    cutoff_u_min = _numeric_array(values, "cutoff_u_min", source, ndim=1)
    if cutoff_un_amp.size == 0 or cutoff_un_amp.shape != cutoff_u_min.shape:
        raise ValueError(
            f"Normal cutoff arrays in {source} must be nonempty and have equal shape."
        )
    if np.any(np.diff(cutoff_un_amp) <= 0.0):
        raise ValueError(f"cutoff_un_amp in {source} must be strictly increasing.")

    return NormalGPParameters(
        *(
            jnp.asarray(value)
            for value in (
                x_train,
                alpha,
                kernel_constant,
                length_scale,
                input_mean,
                input_scale,
                output_mean.reshape(()),
                output_scale.reshape(()),
                cutoff_un_amp,
                cutoff_u_min,
            )
        )
    )


def load_tangential_model(model):
    """Validate and convert tangential-KNN inference parameters to JAX arrays.

    Mirrors the archive format written by
    ``manifolds/rough_contact/tangential_load_prediction_knn.py``
    (``KNNFitter.save_knn_params``) and consumed by
    ``manifolds/rough_contact/main_knn.py``.
    """
    if isinstance(model, TangentialNNParameters):
        return model
    values, source = _load_mapping(model, "tangential-KNN")
    _require_keys(values, _TANGENTIAL_REQUIRED_KEYS, "tangential-KNN", source)

    x_train = _numeric_array(values, "X_train", source, ndim=2)
    if x_train.shape[0] == 0 or x_train.shape[1] != 2:
        raise ValueError(
            f"X_train in {source} must have nonempty shape (Ntrain, 2)."
        )
    z_train = _numeric_array(values, "Z_train", source, shape=(x_train.shape[0],))

    n_neighbors_raw = np.asarray(values["n_neighbors"]).reshape(-1)
    if n_neighbors_raw.size != 1 or not np.issubdtype(
        n_neighbors_raw.dtype, np.integer
    ):
        raise TypeError(f"n_neighbors in {source} must be a single integer.")
    n_neighbors = int(n_neighbors_raw[0])
    if not 0 < n_neighbors <= x_train.shape[0]:
        raise ValueError(
            f"n_neighbors in {source} must satisfy 0 < n_neighbors <= Ntrain."
        )

    distance_weighted_raw = np.asarray(values["distance_weighted"]).reshape(-1)
    if distance_weighted_raw.size != 1:
        raise TypeError(f"distance_weighted in {source} must be a single value.")
    distance_weighted = bool(distance_weighted_raw[0])

    xy_mean = _numeric_array(values, "xy_mean", source, shape=(2,))
    xy_scale = _numeric_array(values, "xy_scale", source, shape=(2,))
    if np.any(xy_scale <= 0.0):
        raise ValueError(f"xy_scale in {source} must be positive.")

    z_mean = _numeric_array(values, "z_mean", source, shape=(1,))
    z_scale = _numeric_array(values, "z_scale", source, shape=(1,))
    if z_scale.item() <= 0.0:
        raise ValueError(f"z_scale in {source} must be positive.")

    scale_u = _positive_scalar(values, "scale_u", source)
    scale_F = _positive_scalar(values, "scale_F", source)

    return TangentialNNParameters(
        x_train=jnp.asarray(x_train),
        z_train=jnp.asarray(z_train),
        xy_mean=jnp.asarray(xy_mean),
        xy_scale=jnp.asarray(xy_scale),
        z_mean=jnp.asarray(z_mean.reshape(())),
        z_scale=jnp.asarray(z_scale.reshape(())),
        scale_u=jnp.asarray(scale_u),
        scale_F=jnp.asarray(scale_F),
        utamp=jnp.asarray(_TANGENTIAL_DISPLACEMENT_AMPLITUDE),
        n_neighbors=n_neighbors,
        distance_weighted=distance_weighted,
    )


def _opening_displacement(normal_displacement, parameters):
    maximum_displacement = jnp.max(normal_displacement)
    return _linear_interpolate(
        maximum_displacement,
        parameters.cutoff_un_amp,
        parameters.cutoff_u_min,
    )


def _linear_interpolate(query, coordinates, values):
    """JAX equivalent of ``numpy.interp`` for one scalar query."""
    if coordinates.shape[0] == 1:
        return values[0]
    index = jnp.sum(query >= coordinates) - 1
    index = jnp.clip(index, 0, coordinates.shape[0] - 2)
    coordinate_left = coordinates[index]
    coordinate_right = coordinates[index + 1]
    value_left = values[index]
    value_right = values[index + 1]
    fraction = (query - coordinate_left) / (coordinate_right - coordinate_left)
    interpolated = value_left + fraction * (value_right - value_left)
    return jnp.where(
        query <= coordinates[0],
        values[0],
        jnp.where(query >= coordinates[-1], values[-1], interpolated),
    )


@jax.jit
def predict_normal_history(normal_displacement, parameters):
    """Evaluate the saved normal GP over a complete displacement history."""
    un = jnp.asarray(normal_displacement)
    if un.ndim != 1 or un.shape[0] == 0:
        raise ValueError(
            "normal_displacement must be a nonempty one-dimensional history."
        )
    maximum_displacement = jnp.max(un)
    maximum_history = jnp.broadcast_to(maximum_displacement, un.shape)
    query = jnp.stack((un, maximum_history), axis=-1)
    scaled_query = (query - parameters.input_mean) / parameters.input_scale

    difference = scaled_query[:, None, :] - parameters.x_train[None, :, :]
    squared_distance = jnp.sum((difference / parameters.length_scale) ** 2, axis=-1)
    cross_covariance = parameters.kernel_constant * jnp.exp(-0.5 * squared_distance)
    scaled_prediction = cross_covariance @ parameters.alpha
    prediction = scaled_prediction * parameters.output_scale + parameters.output_mean

    opening_displacement = _linear_interpolate(
        maximum_displacement,
        parameters.cutoff_un_amp,
        parameters.cutoff_u_min,
    )
    return jnp.where(
        (un < opening_displacement) | (prediction < 0.0),
        jnp.zeros_like(prediction),
        prediction,
    )


def _predict_knn_surface(displacement, normal_force, parameters):
    """Evaluate the saved KNN surface, mirroring ``evaluate_jax`` in
    ``manifolds/rough_contact/tangential_load_prediction_knn.py``."""
    normal_force = jnp.maximum(normal_force, 0.0)
    scaled_displacement = displacement / parameters.scale_u
    scaled_force = normal_force / parameters.scale_F
    query = jnp.stack((scaled_displacement, scaled_force), axis=-1)
    standardized_query = (query - parameters.xy_mean) / parameters.xy_scale

    difference = standardized_query[..., None, :] - parameters.x_train[None, :, :]
    squared_distance = jnp.sum(difference ** 2, axis=-1)

    negative_squared_distance, neighbor_index = jax.lax.top_k(
        -squared_distance, parameters.n_neighbors
    )
    neighbor_squared_distance = -negative_squared_distance
    neighbor_targets = parameters.z_train[neighbor_index]

    if parameters.distance_weighted:
        neighbor_distance = jnp.sqrt(jnp.maximum(neighbor_squared_distance, 0.0))
        weights = 1.0 / jnp.maximum(neighbor_distance, 1e-12)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        standardized_output = jnp.sum(weights * neighbor_targets, axis=-1)
    else:
        standardized_output = jnp.mean(neighbor_targets, axis=-1)

    output = standardized_output * parameters.z_scale + parameters.z_mean
    return output * parameters.scale_F


def _masked_minimum(values, mask):
    minimum = jnp.min(jnp.where(mask, values, jnp.inf))
    return jnp.where(jnp.any(mask), minimum, jnp.zeros_like(minimum))


def _masked_maximum(values, mask):
    maximum = jnp.max(jnp.where(mask, values, -jnp.inf))
    return jnp.where(jnp.any(mask), maximum, jnp.zeros_like(maximum))


@jax.jit
def predict_tangential_history(tangential_displacement, normal_force, parameters):
    """Evaluate the saved KNN surface using the periodic trajectory symmetry in mainb."""
    displacement = jnp.asarray(tangential_displacement)
    normal_force = jnp.asarray(normal_force)
    if (
        displacement.ndim != 1
        or displacement.shape[0] == 0
        or normal_force.shape != displacement.shape
    ):
        raise ValueError(
            "Tangential displacement and normal force must be equal, nonempty "
            "one-dimensional histories."
        )
    centered = displacement - 0.5 * (jnp.min(displacement) + jnp.max(displacement))
    previous = jnp.roll(centered, 1)
    loading_direction = jnp.sign(centered - previous)
    forward = loading_direction >= 0.0

    minimum_centered = jnp.min(centered)
    forward_query = centered - minimum_centered - parameters.utamp
    reverse_query = -centered - minimum_centered - parameters.utamp

    forward_min = _masked_minimum(forward_query, forward)
    forward_max = _masked_maximum(forward_query, forward)
    forward_force = _predict_knn_surface(forward_query, normal_force, parameters)
    forward_offset = 0.5 * (
        _predict_knn_surface(
            jnp.full_like(forward_query, forward_min), normal_force, parameters
        )
        + _predict_knn_surface(
            jnp.full_like(forward_query, forward_max), normal_force, parameters
        )
    )

    reverse = ~forward
    reverse_min = _masked_minimum(reverse_query, reverse)
    reverse_max = _masked_maximum(reverse_query, reverse)
    reverse_force = -_predict_knn_surface(reverse_query, normal_force, parameters)
    reverse_offset = 0.5 * (
        _predict_knn_surface(
            jnp.full_like(reverse_query, reverse_min), normal_force, parameters
        )
        + _predict_knn_surface(
            jnp.full_like(reverse_query, reverse_max), normal_force, parameters
        )
    )

    return jnp.where(
        forward,
        forward_force - forward_offset,
        reverse_force + reverse_offset,
    )


@jax.jit
def _manifold_local_force_history(unlt, normal_parameters, tangential_parameters):
    """Return independent x/y tangential and normal manifold histories."""
    fn_t = predict_normal_history(unlt[:, 2], normal_parameters)
    ftx_t = predict_tangential_history(unlt[:, 0], fn_t, tangential_parameters)
    fty_t = predict_tangential_history(unlt[:, 1], fn_t, tangential_parameters)

    opening_displacement = _opening_displacement(unlt[:, 2], normal_parameters)
    contact_closed = unlt[:, 2] >= opening_displacement
    fn_t = jnp.where(contact_closed, fn_t, 0.0)
    ftx_t = jnp.where(contact_closed, ftx_t, 0.0)
    fty_t = jnp.where(contact_closed, fty_t, 0.0)
    return jnp.stack((ftx_t, fty_t, fn_t), axis=1)


@partial(jax.jit, static_argnames=("htuple", "Nt"))
def _manifold_local_aft(
    Uwlocal,
    normal_parameters,
    tangential_parameters,
    htuple,
    Nt,
):
    nhc = 2 * sum(harmonic != 0 for harmonic in htuple) + sum(
        harmonic == 0 for harmonic in htuple
    )
    local_coefficients = jnp.reshape(Uwlocal[:-1], (3, nhc), order="F").T
    displacement_history = jhutils.time_series_deriv(Nt, htuple, local_coefficients, 0)
    force_history = _manifold_local_force_history(
        displacement_history,
        normal_parameters,
        tangential_parameters,
    )
    force_coefficients = jhutils.get_fourier_coeff(htuple, force_history)
    flattened = jnp.reshape(force_coefficients.T, (-1,), order="F")
    return flattened, flattened


@partial(jax.jit, static_argnames=("htuple", "Nt"))
def _manifold_local_aft_grad(
    Uwlocal,
    normal_parameters,
    tangential_parameters,
    htuple,
    Nt,
):
    return jax.jacfwd(_manifold_local_aft, has_aux=True)(
        Uwlocal,
        normal_parameters,
        tangential_parameters,
        htuple,
        Nt,
    )


class ManifoldFriction(RoughContactFriction):
    """Rough contact with manifold models replacing only periodic evaluation.

    ``normal_model`` and ``tangential_model`` accept either paths to the
    repository's saved ``.npz`` archives or already-loaded parameter mappings.
    Archive I/O and conversion to numerical JAX pytrees occur only here.
    """

    def __init__(
        self,
        Q,
        T,
        ElasticMod,
        PoissonRatio,
        Radius,
        TangentMod,
        YieldStress,
        mu,
        u0=0,
        meso_gap=0,
        gaps=None,
        gap_weights=None,
        tangent_model="TAN",
        N_radial_quad=100,
        *,
        normal_model,
        tangential_model,
    ):
        super().__init__(
            Q,
            T,
            ElasticMod,
            PoissonRatio,
            Radius,
            TangentMod,
            YieldStress,
            mu,
            u0=u0,
            meso_gap=meso_gap,
            gaps=gaps,
            gap_weights=gap_weights,
            tangent_model=tangent_model,
            N_radial_quad=N_radial_quad,
        )
        self.normal_parameters = load_normal_model(normal_model)
        self.tangential_parameters = load_tangential_model(tangential_model)

    def local_force_history(
        self,
        unlt,
        unltdot,
        h,
        cst,
        unlth0,
        max_repeats=2,
        atol=1e-10,
        rtol=1e-10,
    ):
        """Evaluate manifold forces for a local steady-periodic history."""
        del unltdot, h, cst, unlth0, max_repeats, atol, rtol
        unlt = jnp.asarray(unlt)
        if unlt.ndim != 2 or unlt.shape[1] != 3 or unlt.shape[0] == 0:
            raise ValueError("unlt must have nonempty shape (Nt, 3).")
        return (
            _manifold_local_force_history(
                unlt,
                self.normal_parameters,
                self.tangential_parameters,
            ),
        )

    def aft(
        self,
        U,
        w,
        h,
        Nt=128,
        tol=1e-7,
        max_repeats=2,
        return_local=False,
        calc_grad=True,
    ):
        """Evaluate manifold steady-periodic forces with the existing AFT API."""
        del tol, max_repeats
        harmonics = np.asarray(h)
        nhc = 2 * np.count_nonzero(harmonics) + np.count_nonzero(harmonics == 0)
        htuple = tuple(int(value) for value in harmonics)
        U = jnp.asarray(U)
        ndnl = self.Q.shape[0]
        if ndnl != 3:
            raise ValueError("ManifoldFriction requires exactly three local DOFs.")
        if U.ndim != 1 or U.shape[0] != self.Q.shape[1] * nhc:
            raise ValueError("U has an incompatible harmonic coefficient shape.")

        local_coefficients = (
            jnp.asarray(self.Q) @ jnp.reshape(U, (self.Q.shape[1], nhc), order="F")
        ).T
        Uwlocal = jnp.concatenate(
            (jnp.reshape(local_coefficients.T, (-1,), order="F"), jnp.atleast_1d(w))
        )

        if calc_grad:
            local_jacobian, _ = _manifold_local_aft_grad(
                Uwlocal,
                self.normal_parameters,
                self.tangential_parameters,
                htuple,
                Nt,
            )
        # Always obtain the returned force from the same primal executable so
        # calc_grad=False preserves the existing bitwise force behavior.
        local_force, _ = _manifold_local_aft(
            Uwlocal,
            self.normal_parameters,
            self.tangential_parameters,
            htuple,
            Nt,
        )

        local_force = jnp.reshape(local_force, (ndnl, nhc), order="F")
        if return_local:
            if calc_grad:
                return local_force, local_jacobian[:, :-1], local_jacobian[:, -1]
            return (local_force,)

        global_force = jnp.reshape(jnp.asarray(self.T) @ local_force, (-1,), order="F")
        if not calc_grad:
            return (global_force,)

        left_transform = jnp.kron(jnp.eye(nhc, dtype=U.dtype), jnp.asarray(self.T))
        right_transform = jnp.kron(jnp.eye(nhc, dtype=U.dtype), jnp.asarray(self.Q))
        global_jacobian = left_transform @ local_jacobian[:, :-1] @ right_transform
        global_frequency_gradient = jnp.reshape(
            jnp.asarray(self.T)
            @ jnp.reshape(local_jacobian[:, -1], (ndnl, nhc), order="F"),
            (-1,),
            order="F",
        )
        return global_force, global_jacobian, global_frequency_gradient


__all__ = [
    "ManifoldFriction",
    "NormalGPParameters",
    "TangentialNNParameters",
    "load_normal_model",
    "load_tangential_model",
    "predict_normal_history",
    "predict_tangential_history",
]

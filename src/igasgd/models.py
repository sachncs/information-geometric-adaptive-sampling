"""Simplified denoising network approximations for GruM and GDSS.

The paper (arXiv:2605.00250v1) does not provide full architectural
details or pretrained weights for the GruM and GDSS backbone models.
The real networks are deep learning models (likely PyTorch / PyTorch
Geometric) with graph attention, message passing, and time embedding
layers.  Because we cannot ship a faithful reproduction, this module
provides a stand-in that satisfies the ``drift_function(X, A, t)``
interface contract used by :class:`~igasgd.sampler.DVSSampler`.

What we provide
---------------
* :class:`SimpleGraphDenoiser` -- a 3-layer MLP applied per-node that
  consumes the node's own features, its adjacency row, and a
  sinusoidal time embedding.  This satisfies the drift interface but
  is **not** a real message-passing network.
* :class:`GruMApproximation` -- :class:`SimpleGraphDenoiser` with a
  larger hidden width (32), intended as a stand-in for GruM.
* :class:`GDSSApproximation` -- :class:`SimpleGraphDenoiser` with a
  smaller hidden width (24), intended as a stand-in for GDSS.
* :func:`make_drift_function` -- convenience wrapper that adapts any
  :class:`SimpleGraphDenoiser` instance to the call signature expected
  by :class:`~igasgd.sampler.DVSSampler`.

What is **missing** (and explicitly marked as gaps)
---------------------------------------------------
* Exact layer dimensions, activation functions, and normalisation
  layers from the original GruM / GDSS papers.
* Pretrained weights.
* Graph neural network message-passing operations.
* Time-embedding Fourier features or learned embeddings (we use a
  generic sinusoidal positional encoding).

Users should replace these simplified models with the real trained
networks once they become available; the only contract is that the
replacement must accept ``(X, A, t)`` and return ``(f_X, f_A)`` with
matching shapes.

Assumptions
-----------
* ``feature_dim`` and ``num_nodes`` are fixed for the lifetime of a
  model instance (this is a consequence of using ``__init__``-time
  Xavier-initialised weight matrices).
* The drift network is deterministic for fixed inputs.

Limitations
-----------
* Pure-Python nested-list operations make these approximations an
  order of magnitude slower than a vectorised NumPy or PyTorch
  implementation.  They are intended for smoke tests and
  demonstrations only.

References:
----------
* GruM (Jo et al., 2024).
* GDSS (Jo et al., 2022).
* Sinusoidal time embedding convention: Vaswani et al., "Attention is
  All You Need", 2017.
"""

import math
import random
from typing import Callable, List, Tuple


def _xavier_uniform(rows: int, cols: int, rng: random.Random) -> List[List[float]]:
    """Generate a ``rows x cols`` matrix with Xavier uniform initialisation.

    Xavier (Glorot) uniform initialisation scales weights by
    ``sqrt(6 / (rows + cols))`` so that the variance of activations is
    roughly preserved across layers.

    Args:
        rows: Number of rows in the output matrix.
        cols: Number of columns in the output matrix.
        rng: Random number generator used for sampling.

    Returns:
        A nested list of shape ``(rows, cols)`` with entries drawn
        uniformly from ``[-limit, limit]`` where ``limit =
        sqrt(6 / (rows + cols))``.
    """
    limit = math.sqrt(6.0 / (rows + cols))
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def _matvec(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Multiply a matrix by a column vector.

    Args:
        matrix: Two-dimensional list of shape ``(m, n)``.
        vector: One-dimensional list of length ``n``.

    Returns:
        A list of length ``m`` containing ``matrix @ vector``.

    Raises:
        ValueError: Propagated from Python's ``zip`` if the inner
            dimensions mismatch.

    Complexity:
        O(m * n).
    """
    return [sum(w * v for w, v in zip(row, vector)) for row in matrix]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices using pure-Python loops.

    Used only by :class:`SimpleGraphDenoiser` for small matrices; for
    larger workloads a vectorised backend should be substituted.

    Args:
        a: Left matrix of shape ``(m, k)``.
        b: Right matrix of shape ``(k, n)``.

    Returns:
        A list of shape ``(m, n)`` representing ``a @ b``.

    Complexity:
        O(m * k * n).
    """
    result: List[List[float]] = []
    for row in a:
        new_row: List[float] = []
        for col_idx in range(len(b[0])):
            val = sum(row[k] * b[k][col_idx] for k in range(len(b)))
            new_row.append(val)
        result.append(new_row)
    return result


def _relu(values: List[float]) -> List[float]:
    """Apply the element-wise ReLU activation.

    Args:
        values: Iterable of floats.

    Returns:
        A new list of the same length with ``max(0, v)`` applied per
        element.
    """
    return [max(0.0, v) for v in values]


def _time_embedding(time: float, dim: int) -> List[float]:
    """Compute a sinusoidal time embedding (Fourier features).

    This is a standard diffusion-model convention originating from the
    Transformer literature (Vaswani et al., 2017).  The paper does not
    specify how time is embedded into GruM / GDSS, so this is a
    reasonable default that satisfies the only requirement we have on
    a time embedding: it must vary with ``time`` and produce a
    deterministic vector.

    Args:
        time: Diffusion time ``t``.
        dim: Embedding dimensionality.  Must be non-negative; ``dim=0``
            returns an empty list.

    Returns:
        A list of length ``dim`` containing alternating sine and
        cosine components with exponentially growing frequencies.

    Edge cases:
        * ``dim = 0`` returns ``[]`` without sampling.
        * Negative ``dim`` is treated like ``0`` by the loop guard.
    """
    embedding: List[float] = []
    # Frequencies grow exponentially so the embedding can resolve
    # both coarse and fine time scales.
    for i in range(dim):
        freq = math.exp(i * math.log(10000.0) / dim) if dim > 0 else 1.0
        if i % 2 == 0:
            embedding.append(math.sin(time * freq))
        else:
            embedding.append(math.cos(time * freq))
    return embedding


class SimpleGraphDenoiser:
    """Simplified time-conditioned denoising network approximation.

    This is a toy model intended only for interface compatibility,
    smoke tests, and demonstrations.  It does **not** perform real
    message passing or graph attention; it is essentially a 3-layer
    MLP applied independently to each node's feature/adjacency slice.

    Architecture
    ------------
    For each node ``i`` the input is the concatenation of:

    1. The node's own feature vector of length ``feature_dim``.
    2. The node's adjacency row of length ``num_nodes``.
    3. A sinusoidal time embedding of length ``time_dim`` (= 8).

    The concatenated vector is then passed through a 3-layer MLP with
    ReLU activations; the final layer splits into two halves that are
    reshaped into the feature drift and adjacency drift for that node.

    Thread-safety:
        Instances are **not** thread-safe because the weight matrices
        are mutable (although in practice no method mutates them).
        Build one instance per thread for safety.

    Lifecycle:
        The model is constructed once with fixed shapes
        (``num_nodes``, ``feature_dim``, ``hidden_dim``); subsequent
        calls are stateless evaluations.

    Example:
        >>> model = SimpleGraphDenoiser(num_nodes=4, feature_dim=3, seed=0)
        >>> drift_x, drift_a = model(
        ...     [[0.1, 0.2, 0.3]] * 4,
        ...     [[0.5] * 4 for _ in range(4)],
        ...     time=0.5,
        ... )
        >>> len(drift_x), len(drift_a)
        (4, 4)
    """

    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        hidden_dim: int = 16,
        seed: int = 42,
    ) -> None:
        """Initialise the simplified denoising network.

        Allocates three weight matrices with Xavier uniform
        initialisation and three bias vectors sampled from a small
        Gaussian.  Shapes are computed from ``num_nodes``,
        ``feature_dim``, ``hidden_dim``, and the internal ``time_dim``
        (= 8).

        Args:
            num_nodes: Number of nodes in the graph (``N``).
            feature_dim: Dimensionality of node features (``D``).
            hidden_dim: Width of the two hidden layers in the MLP.
            seed: Random seed for reproducible weight initialisation.
        """
        self._num_nodes = num_nodes
        self._feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        self._rng = random.Random(seed)

        # Per-node input = own features + adjacency row + time embedding.
        time_dim = 8
        input_dim = feature_dim + num_nodes + time_dim
        output_dim = feature_dim + num_nodes  # drift for X and A concatenated

        self._weight1 = _xavier_uniform(input_dim, hidden_dim, self._rng)
        self._bias1 = [self._rng.gauss(0.0, 0.01) for _ in range(hidden_dim)]
        self._weight2 = _xavier_uniform(hidden_dim, hidden_dim, self._rng)
        self._bias2 = [self._rng.gauss(0.0, 0.01) for _ in range(hidden_dim)]
        self._weight3 = _xavier_uniform(hidden_dim, output_dim, self._rng)
        self._bias3 = [self._rng.gauss(0.0, 0.01) for _ in range(output_dim)]
        self._time_dim = time_dim

    def _mlp(self, features: List[float]) -> List[float]:
        """Forward pass through the 3-layer MLP with ReLU activations.

        Applies the affine transformation ``W_i x + b_i`` at each
        layer and ReLU between layers 1-2 and 2-3.  The output is
        post-bias only (no final activation) so the drift can take any
        real value.

        Args:
            features: Input vector of length ``input_dim``.

        Returns:
            Output vector of length ``output_dim``.
        """
        h = _matvec(self._weight1, features)
        h = [v + b for v, b in zip(h, self._bias1)]
        h = _relu(h)
        h = _matvec(self._weight2, h)
        h = [v + b for v, b in zip(h, self._bias2)]
        h = _relu(h)
        out = _matvec(self._weight3, h)
        out = [v + b for v, b in zip(out, self._bias3)]
        return out

    def __call__(
        self,
        features: List[List[float]],
        adjacency: List[List[float]],
        time: float,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Predict the per-modality drift at the given state and time.

        For each node the model concatenates the node features, the
        corresponding adjacency row, and the sinusoidal time
        embedding, then runs the resulting vector through the MLP.
        The MLP output is split into two halves that are reshaped into
        the feature drift and adjacency drift respectively.

        Args:
            features: Node feature matrix ``X`` of shape ``(N, D)``.
            adjacency: Adjacency matrix ``A`` of shape ``(N, N)``.
            time: Current diffusion time ``t``.

        Returns:
            A pair ``(drift_features, drift_adjacency)`` of shapes
            ``(N, D)`` and ``(N, N)`` respectively.

        Complexity:
            O(N * input_dim * hidden_dim) for the per-node MLPs.
        """
        time_emb = _time_embedding(time, self._time_dim)
        drift_features: List[List[float]] = []
        drift_adjacency_rows: List[List[float]] = []

        for node_idx in range(self._num_nodes):
            node_feat = features[node_idx]
            adj_row = adjacency[node_idx]
            inp = node_feat + adj_row + time_emb
            out = self._mlp(inp)

            # Split the MLP output into feature drift and adjacency drift.
            feat_drift = out[: self._feature_dim]
            adj_drift = out[self._feature_dim :]
            drift_features.append(feat_drift)
            drift_adjacency_rows.append(adj_drift)

        return drift_features, drift_adjacency_rows


class GruMApproximation(SimpleGraphDenoiser):
    """Simplified approximation of the GruM denoising model.

    Inherits the simple MLP approximation from
    :class:`SimpleGraphDenoiser` and overrides the hidden width to
    ``32`` so that the parameter count loosely matches the published
    GruM configuration.  This is an **educational stand-in** until
    the real GruM architecture (Jo et al., 2024) and pretrained
    weights are publicly available; it does not perform the
    multi-scale graph message passing that defines GruM.

    Example:
        >>> approx = GruMApproximation(num_nodes=5, feature_dim=3, seed=42)
        >>> approx._hidden_dim
        32
    """

    def __init__(self, num_nodes: int, feature_dim: int, seed: int = 42) -> None:
        """Initialise the GruM approximation.

        Args:
            num_nodes: Number of nodes (``N``).
            feature_dim: Feature dimensionality (``D``).
            seed: Random seed for weight initialisation.

        Note:
            ``hidden_dim`` is hard-coded to ``32`` to mirror a typical
            GruM hidden width.  There is no exposed constructor
            argument for it because the class is a placeholder.
        """
        super().__init__(num_nodes, feature_dim, hidden_dim=32, seed=seed)


class GDSSApproximation(SimpleGraphDenoiser):
    """Simplified approximation of the GDSS denoising model.

    Inherits the simple MLP approximation from
    :class:`SimpleGraphDenoiser` and overrides the hidden width to
    ``24``.  This is an **educational stand-in** until the real GDSS
    architecture (Jo et al., 2022) and pretrained weights are publicly
    available; it does not perform the Schrödinger-bridge-augmented
    message passing that defines GDSS.

    Example:
        >>> approx = GDSSApproximation(num_nodes=5, feature_dim=3, seed=42)
        >>> approx._hidden_dim
        24
    """

    def __init__(self, num_nodes: int, feature_dim: int, seed: int = 42) -> None:
        """Initialise the GDSS approximation.

        Args:
            num_nodes: Number of nodes (``N``).
            feature_dim: Feature dimensionality (``D``).
            seed: Random seed for weight initialisation.

        Note:
            ``hidden_dim`` is hard-coded to ``24`` to mirror a typical
            GDSS hidden width.  There is no exposed constructor
            argument for it because the class is a placeholder.
        """
        super().__init__(num_nodes, feature_dim, hidden_dim=24, seed=seed)


def make_drift_function(model: SimpleGraphDenoiser) -> Callable:
    """Wrap a :class:`SimpleGraphDenoiser` into the standard drift interface.

    The wrapper closes over ``model`` so the returned callable can be
    passed directly to :class:`~igasgd.sampler.DVSSampler`.  It exists
    mainly as a documentation aid: it makes explicit that the
    underlying object is a model and the wrapping callable is the
    drift function.

    Args:
        model: A simplified denoising network (typically a
            :class:`GruMApproximation` or :class:`GDSSApproximation`).

    Returns:
        A callable ``(X, A, t) -> (f_X, f_A)`` compatible with
        :class:`~igasgd.sampler.DVSSampler`.  The wrapper simply
        delegates to ``model.__call__``; it performs no additional
        computation or state management.

    Example:
        >>> from igasgd import GruMApproximation, make_drift_function
        >>> approx = GruMApproximation(num_nodes=3, feature_dim=2, seed=0)
        >>> drift = make_drift_function(approx)
        >>> drift([[0.0, 0.0]] * 3, [[0.0] * 3 for _ in range(3)], 0.0)[0]
        [[...], [...], [...]]
    """

    def _drift(
        features: List[List[float]],
        adjacency: List[List[float]],
        time: float,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        return model(features, adjacency, time)

    return _drift

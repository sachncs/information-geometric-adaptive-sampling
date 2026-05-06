"""Simplified denoising network approximations for GruM and GDSS.

The paper (arXiv:2605.00250v1) does not provide full architectural details or
pretrained weights for the GruM and GDSS backbone models.  The real networks
are deep learning models (likely PyTorch / PyTorch Geometric) with graph
attention, message passing, and time embedding layers.

What we implement here:
- Pure-Python callables that satisfy the ``drift_function(X, A, t)`` interface.
- Simple time-conditioned MLP-style transformations as educational stand-ins.
- Random initialisation with fixed seeds so outputs are deterministic for tests.

What is **missing** (and marked as gaps in the fidelity report):
- Exact layer dimensions, activation functions, and normalisation from the
  original GruM / GDSS papers.
- Pretrained weights.
- Graph neural network message-passing operations.
- Time-embedding Fourier features or learned embeddings.

Users should replace these simplified models with the real trained networks
once available.
"""

import math
import random
from typing import Callable, List, Tuple


def _xavier_uniform(rows: int, cols: int, rng: random.Random) -> List[List[float]]:
    """Generate a weight matrix with Xavier uniform initialisation."""
    limit = math.sqrt(6.0 / (rows + cols))
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def _matvec(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Multiply a matrix by a column vector."""
    return [sum(w * v for w, v in zip(row, vector)) for row in matrix]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices (a @ b)."""
    result: List[List[float]] = []
    for row in a:
        new_row: List[float] = []
        for col_idx in range(len(b[0])):
            val = sum(row[k] * b[k][col_idx] for k in range(len(b)))
            new_row.append(val)
        result.append(new_row)
    return result


def _relu(values: List[float]) -> List[float]:
    """Element-wise ReLU activation."""
    return [max(0.0, v) for v in values]


def _time_embedding(time: float, dim: int) -> List[float]:
    """Sinusoidal time embedding (simplified Fourier features).

    This is a common diffusion-model convention.  The paper does not specify
    how time is embedded, so this is a reasonable default approximation.
    """
    embedding: List[float] = []
    for i in range(dim):
        freq = math.exp(i * math.log(10000.0) / dim) if dim > 0 else 1.0
        if i % 2 == 0:
            embedding.append(math.sin(time * freq))
        else:
            embedding.append(math.cos(time * freq))
    return embedding


class SimpleGraphDenoiser:
    """Simplified time-conditioned denoising network approximation.

    This is a toy model intended only for interface compatibility and smoke
    tests.  It does **not** perform real message passing or graph attention.

    Architecture (approximate):
        1. Flatten node features and adjacency per node.
        2. Concatenate sinusoidal time embedding.
        3. Apply a small MLP with Xavier-initialised weights.
        4. Reshape outputs back to feature / adjacency shapes.
    """

    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        hidden_dim: int = 16,
        seed: int = 42,
    ) -> None:
        """Initialise the simplified network.

        Args:
            num_nodes: Number of nodes in the graph (N).
            feature_dim: Dimensionality of node features (D).
            hidden_dim: Hidden layer width of the MLP.
            seed: Random seed for reproducible weight initialisation.
        """
        self._num_nodes = num_nodes
        self._feature_dim = feature_dim
        self._hidden_dim = hidden_dim
        self._rng = random.Random(seed)

        # Input per node = own features + aggregated adjacency row + time embedding
        time_dim = 8
        input_dim = feature_dim + num_nodes + time_dim
        output_dim = feature_dim + num_nodes  # drift for X and A

        self._weight1 = _xavier_uniform(input_dim, hidden_dim, self._rng)
        self._bias1 = [self._rng.gauss(0.0, 0.01) for _ in range(hidden_dim)]
        self._weight2 = _xavier_uniform(hidden_dim, hidden_dim, self._rng)
        self._bias2 = [self._rng.gauss(0.0, 0.01) for _ in range(hidden_dim)]
        self._weight3 = _xavier_uniform(hidden_dim, output_dim, self._rng)
        self._bias3 = [self._rng.gauss(0.0, 0.01) for _ in range(output_dim)]
        self._time_dim = time_dim

    def _mlp(self, features: List[float]) -> List[float]:
        """Forward pass through the 3-layer MLP."""
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
        """Predict the drift (score) for node features and adjacency.

        Args:
            features: Node feature matrix of shape (N, D).
            adjacency: Adjacency matrix of shape (N, N).
            time: Current diffusion time.

        Returns:
            A pair ``(drift_features, drift_adjacency)`` with the same shapes
            as the inputs.
        """
        time_emb = _time_embedding(time, self._time_dim)
        drift_features: List[List[float]] = []
        drift_adjacency_rows: List[List[float]] = []

        for node_idx in range(self._num_nodes):
            node_feat = features[node_idx]
            adj_row = adjacency[node_idx]
            inp = node_feat + adj_row + time_emb
            out = self._mlp(inp)

            feat_drift = out[: self._feature_dim]
            adj_drift = out[self._feature_dim :]
            drift_features.append(feat_drift)
            drift_adjacency_rows.append(adj_drift)

        return drift_features, drift_adjacency_rows


class GruMApproximation(SimpleGraphDenoiser):
    """Simplified approximation of the GruM (Graph Multi-scale) denoising model.

    This inherits the simple MLP approximation and is intended as an
    educational stand-in until the real GruM architecture (Jo et al., 2024)
    and pretrained weights are available.
    """

    def __init__(self, num_nodes: int, feature_dim: int, seed: int = 42) -> None:
        """Initialise the GruM approximation.

        Args:
            num_nodes: Number of nodes (N).
            feature_dim: Feature dimensionality (D).
            seed: Random seed for weight initialisation.
        """
        super().__init__(num_nodes, feature_dim, hidden_dim=32, seed=seed)


class GDSSApproximation(SimpleGraphDenoiser):
    """Simplified approximation of the GDSS (Graph Diffusion with Schrödinger
    Bridge) denoising model.

    This inherits the simple MLP approximation and is intended as an
    educational stand-in until the real GDSS architecture (Jo et al., 2022)
    and pretrained weights are available.
    """

    def __init__(self, num_nodes: int, feature_dim: int, seed: int = 42) -> None:
        """Initialise the GDSS approximation.

        Args:
            num_nodes: Number of nodes (N).
            feature_dim: Feature dimensionality (D).
            seed: Random seed for weight initialisation.
        """
        super().__init__(num_nodes, feature_dim, hidden_dim=24, seed=seed)


def make_drift_function(model: SimpleGraphDenoiser) -> Callable:
    """Wrap a ``SimpleGraphDenoiser`` into the standard drift interface.

    Args:
        model: A simplified denoising network.

    Returns:
        A callable ``(X, A, t) -> (f_X, f_A)`` compatible with ``DVSSampler``.
    """

    def _drift(
        features: List[List[float]],
        adjacency: List[List[float]],
        time: float,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        return model(features, adjacency, time)

    return _drift

"""Observed-label reliability channels shared by diagnostics and future training.

These functions accept predictions, observed labels and model features only.
The Phase 5B.2 caller supplies its own replay; Phase 6 must supply its own run.
No function accepts clean labels, audit groups, or a future-stage feature bank.
"""
from __future__ import annotations

import numpy as np


class InvariantError(ValueError):
    """An input violates the declared experiment protocol."""


def require(condition, message):
    if not bool(condition):
        raise InvariantError(message)


def check_probabilities(values, *, sum_atol=0.002):
    p = np.asarray(values)
    require(p.ndim >= 2 and p.shape[-1] > 1 and p.size > 0, "Invalid probability shape")
    require(p.dtype.kind == "f", "Probabilities must be floating point")
    require(np.isfinite(p).all(), "Nonfinite probability")
    require(((p >= 0) & (p <= 1)).all(), "Probability outside [0,1]")
    sums = p.astype(np.float32).sum(axis=-1)
    require(np.allclose(sums, 1, atol=sum_atol, rtol=0), "Probability rows do not sum to one")
    return p


def check_observed(observed, n, classes):
    y = np.asarray(observed)
    require(y.shape == (n,) and y.dtype.kind in "iu", "Invalid observed-label shape/type")
    require(((y >= 0) & (y < classes)).all(), "Observed label outside class range")
    return y


def confidence(probabilities, observed):
    p = check_probabilities(probabilities)
    require(p.ndim == 2, "Confidence requires an N x classes matrix")
    y = check_observed(observed, len(p), p.shape[1])
    return p[np.arange(len(p)), y].astype(np.float32)


def temporal_label_agreement(completed_history, observed, window=5):
    """T-v2 from the last `window` COMPLETED epochs, with a full window required.

    This explicit boundary is reusable by causal Phase 6 training. For an epoch-e
    diagnostic, pass history[:e-1]; for an epoch-e training rule, pass only the
    completed history available at the time of the loss computation.
    """
    h = np.asarray(completed_history)
    require(isinstance(window, int) and window > 0, "Invalid temporal window")
    require(h.ndim == 3 and h.shape[0] >= window, "Full past temporal window unavailable")
    past = check_probabilities(h[-window:])
    y = check_observed(observed, past.shape[1], past.shape[2])
    return (past.argmax(axis=2) == y[None, :]).mean(axis=0).astype(np.float32)


def stage_confidence_temporal(history, observed, epoch, window=5):
    """One-based epoch; current C and five past epochs for T, excluding current."""
    h = np.asarray(history)
    require(h.ndim == 3 and isinstance(epoch, int), "Invalid history/epoch")
    require(window < epoch <= len(h), "Stage lacks a full past window")
    return {
        "C": confidence(h[epoch - 1], observed),
        "T_v2": temporal_label_agreement(h[:epoch - 1], observed, window),
    }


def validate_neighbors(neighbors, n):
    ids = np.asarray(neighbors)
    require(ids.ndim == 2 and ids.shape[0] == n and 0 < ids.shape[1] < n,
            "Invalid neighbor matrix shape")
    require(ids.dtype.kind in "iu", "Neighbor IDs must be integers")
    require(((ids >= 0) & (ids < n)).all(), "Neighbor ID outside local sample range")
    require(not (ids == np.arange(n)[:, None]).any(), "Self neighbor detected")
    require((np.diff(np.sort(ids, axis=1), axis=1) != 0).all(), "Duplicate neighbor")
    return ids


def neighborhood_support(probabilities, observed, neighbors):
    """Neighbor support for the QUERY's observed label, not the neighbor's label."""
    p = check_probabilities(probabilities)
    require(p.ndim == 2, "Neighborhood scoring requires one stage")
    y = check_observed(observed, len(p), p.shape[1])
    ids = validate_neighbors(neighbors, len(p))
    support = p[ids, y[:, None]]
    votes = p.argmax(axis=1)[ids] == y[:, None]
    return {"N_soft": support.astype(np.float32).mean(axis=1),
            "N_vote": votes.mean(axis=1).astype(np.float32)}


def normalized_features(features):
    f = np.asarray(features, dtype=np.float32)
    require(f.ndim == 2 and f.shape[0] > 1 and f.shape[1] > 0, "Invalid feature shape")
    require(np.isfinite(f).all(), "Nonfinite embedding")
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    require(np.isfinite(norms).all() and (norms > 0).all(), "Zero/nonfinite feature norm")
    return f / norms


def exact_knn_numpy(features, k, query_batch=128):
    """Small-data CPU reference; ties favor ascending local IDs, including duplicates."""
    f = normalized_features(features)
    n = len(f)
    require(isinstance(k, int) and 0 < k < n, "Invalid K")
    require(query_batch > 0, "Invalid query batch size")
    ids = np.empty((n, k), dtype=np.int64)
    similarities = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, query_batch):
        stop = min(start + query_batch, n)
        sim = f[start:stop] @ f.T
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        ranked = np.argsort(-sim, axis=1, kind="stable")[:, :k]
        ids[start:stop] = ranked
        similarities[start:stop] = np.take_along_axis(sim, ranked, axis=1)
    validate_neighbors(ids, n)
    return ids, similarities, f


class PredictionParityError(InvariantError):
    def __init__(self, report):
        self.report = report
        super().__init__("Checkpoint/history parity failed: " + str(report))


def prediction_parity(recomputed, recorded, max_abs_error=0.0005):
    """Fixed initial gate. Never automatically relax on failure."""
    require(0 < max_abs_error <= 0.0005, "Parity tolerance cannot exceed approved initial gate")
    new = check_probabilities(recomputed, sum_atol=2e-5)
    old = check_probabilities(recorded)
    require(new.ndim == 2 and new.shape == old.shape, "Parity shape mismatch")
    error = np.abs(new.astype(np.float32) - old.astype(np.float32))
    report = {
        "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
        "p99_abs_error": float(np.quantile(error, 0.99)),
        "argmax_disagreements": int((new.argmax(1) != old.argmax(1)).sum()),
        "threshold": float(max_abs_error), "rtol": 0,
        "passed": bool(error.max() <= max_abs_error),
    }
    if not report["passed"]:
        raise PredictionParityError(report)
    return report

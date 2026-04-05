import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def compute_within_cluster_variances(
    members_by_cluster: Sequence[np.ndarray], centers: np.ndarray
) -> List[float]:
    """Compute variance of member distances to each cluster center."""
    variances: List[float] = []
    for cid, members in enumerate(members_by_cluster):
        if members.shape[0] < 2:
            logger.warning("Cluster %d has <2 members; skipped in tau estimation", cid)
            continue
        distances = np.linalg.norm(members - centers[cid], axis=1)
        variances.append(float(np.var(distances)))
    return variances


def recommend_tau(
    members_by_cluster: Sequence[np.ndarray],
    centers: np.ndarray,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    round_digits: Optional[int] = None,
) -> Dict[str, object]:
    """Estimate tau from cluster spread and return summary values."""
    within_vars = compute_within_cluster_variances(members_by_cluster, centers)
    if not within_vars:
        raise ValueError("No valid clusters to estimate tau")

    mean_within_var = float(np.mean(within_vars))
    tau_raw = float(np.sqrt(mean_within_var))
    tau_used = tau_raw

    if clamp_min is not None:
        tau_used = max(float(clamp_min), tau_used)
    if clamp_max is not None:
        tau_used = min(float(clamp_max), tau_used)
    if round_digits is not None:
        tau_used = round(tau_used, int(round_digits))

    return {
        "within_cluster_variances": within_vars,
        "mean_within_cluster_variance": mean_within_var,
        "recommended_tau_raw": tau_raw,
        "recommended_tau": tau_used,
    }

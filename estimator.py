"""Local sensor packets and constant-velocity observation-band reachability."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Observation:
    """Copied local 3-D tracks, with the craft's onboard availability flag."""
    time: float
    agent_id: int
    chaser_ids: np.ndarray
    chaser_pos: np.ndarray
    chaser_vel: np.ndarray
    target_ids: np.ndarray
    target_pos: np.ndarray
    target_vel: np.ndarray
    active: bool = True
    target_available: np.ndarray | None = None


@dataclass(frozen=True)
class PairEstimates:
    relative_pos: np.ndarray
    relative_vel: np.ndarray
    distance: np.ndarray
    closing: np.ndarray
    t_go: np.ndarray
    zem: np.ndarray
    omega_los: np.ndarray
    probability: np.ndarray


def estimate_pairs(chaser_pos, chaser_vel, target_pos, target_vel, *,
                   r_dock: float, sigma_base: float, zem_scale: float,
                   ground_z: float = 0., min_separation: float = 0.,
                   perimeter_x: float | None = None) -> PairEstimates:
    """Estimate arrival at the outer observation radius from local tracks.

    r_dock is a legacy name for the OUTER observation radius, not contact range.
    t_go=max(distance-r_dock,0)/max(closing,1e-3). Safe in-band pairs retain
    positive reachability when stationary or receding; below-minimum pairs have
    zero reachability. The +x operational perimeter is the only flight deadline.
    ground_z is accepted for compatibility but does not declare an escape.

    The Rayleigh-style score and constant-velocity projection are heuristics,
    not calibrated service probabilities or guarantees of sustained tracking.
    Vector LOS angular velocity remains cross(r,v_rel)/||r||^2.
    """
    parameters = [r_dock, sigma_base, zem_scale, ground_z, min_separation]
    if (not np.isfinite(parameters).all() or r_dock <= 0 or sigma_base <= 0
            or zem_scale < 0 or not 0 <= min_separation < r_dock
            or (perimeter_x is not None and not np.isfinite(perimeter_x))):
        raise ValueError('invalid observation reachability parameters')
    cp, cv, tp, tv = [np.asarray(x, dtype=float) for x in
                      (chaser_pos, chaser_vel, target_pos, target_vel)]
    if any(x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all()
           for x in (cp, cv, tp, tv)):
        raise ValueError('states must be finite (count, 3) arrays')
    if cp.shape != cv.shape or tp.shape != tv.shape:
        raise ValueError('position and velocity shapes must agree')
    r, rv = tp[None, :, :]-cp[:, None, :], tv[None, :, :]-cv[:, None, :]
    distance = np.linalg.norm(r, axis=-1)
    closing = -np.sum(r*rv, axis=-1)/np.maximum(distance, 1e-9)
    t_go = np.maximum(distance-r_dock, 0.)/np.maximum(closing, 1e-3)
    zem = np.linalg.norm(r+rv*t_go[..., None], axis=-1)
    omega = np.cross(r, rv)/np.maximum(distance**2, 1e-12)[..., None]
    boundary_time = np.full(len(tp), np.inf)
    beyond = np.zeros(len(tp), dtype=bool)
    if perimeter_x is not None:
        toward_perimeter = tv[:, 0] > 0
        boundary_time[toward_perimeter] = np.maximum(0.,
            (perimeter_x-tp[toward_perimeter, 0])/tv[toward_perimeter, 0])
        beyond = tp[:, 0] >= perimeter_x
        boundary_time[beyond] = 0.
    sigma = sigma_base+zem_scale*zem
    probability = -np.expm1(-r_dock**2/(2*sigma**2))
    safe_band = (distance >= min_separation) & (distance < r_dock)
    probability[((closing <= 0) & ~safe_band) | (distance < min_separation)
                | (t_go > boundary_time[None, :]) | beyond[None, :]] = 0.
    return PairEstimates(r, rv, distance, closing, t_go, zem, omega, probability)

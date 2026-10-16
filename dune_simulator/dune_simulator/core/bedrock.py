"""Mass-conserving enforcement of a non-erodible bedrock floor.

A naive ``np.maximum(floor, h)`` clip silently *creates* sand whenever a cell is
eroded below the floor, which breaks the Exner mass balance (it was found to
cancel the entire downstream outflow of a 2D barchan).  The helper in this module
instead removes the created volume again from the sand that is actually present,
so the total sand volume is conserved to machine precision whenever enough sand
exists in the domain.
"""

from typing import Optional, Tuple
import numpy as np


def enforce_floor_conservative(
    h: np.ndarray,
    bedrock_floor: Optional[object] = None,
) -> Tuple[np.ndarray, float]:
    """Clips ``h`` to the bedrock floor while conserving the total sand volume.

    Cells below the floor are raised to the floor (creating a volume deficit D);
    the same volume D is then removed proportionally to the available sand
    ``h - floor`` of all other cells. Because the removal factor is < 1 the result
    never falls below the floor again.

    Parameters:
        h: Bed elevation array (any shape).
        bedrock_floor: Scalar or array broadcastable to ``h``. Default 0.0.

    Returns:
        (h_new, unresolved): The floored bed, and the volume (in units of
        elevation-sum) that could NOT be removed because the domain holds less
        sand than the deficit (0.0 in all normal situations).
    """
    h = np.asarray(h, dtype=np.float64)
    floor = np.zeros_like(h) if bedrock_floor is None else np.broadcast_to(
        np.asarray(bedrock_floor, dtype=np.float64), h.shape
    )

    deficit_cells = floor - h
    below = deficit_cells > 0.0
    if not np.any(below):
        return h, 0.0

    deficit = float(np.sum(deficit_cells[below]))
    h_new = np.where(below, floor, h)

    available = h_new - floor
    total_available = float(np.sum(available))
    if total_available <= 0.0:
        return h_new, deficit

    removable = min(deficit, total_available)
    h_new = h_new - available * (removable / total_available)
    # Guard against round-off pushing a cell a hair below the floor
    h_new = np.maximum(h_new, floor)
    return h_new, deficit - removable

from typing import List, Optional

import numpy as np

level_progression = [f"{w}-{l}" for w in range(1, 9) for l in range(1, 5)]

schedule_cache = {0: []}


def get_linear_level_schedule(num_repetitions_per_level: int = 1) -> List[str]:
    return sum(
        [[level] * num_repetitions_per_level for level in level_progression], start=[]
    )


def get_level_schedule(up_to_idx: Optional[int] = None) -> List[str]:
    if up_to_idx is None:
        up_to_idx = len(level_progression)
    if up_to_idx in schedule_cache:
        return schedule_cache[up_to_idx]
    np.random.seed(up_to_idx)
    conc = np.concatenate([[l] * 5 for l in level_progression[:up_to_idx]])
    np.random.shuffle(conc)
    result = (
        get_level_schedule(up_to_idx - 1)
        + [level_progression[up_to_idx - 1]] * 5
        + list(conc)
    )
    schedule_cache[up_to_idx] = result
    return result

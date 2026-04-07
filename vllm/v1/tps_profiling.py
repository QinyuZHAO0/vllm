import os
import threading
from collections import defaultdict
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_TPS_PROFILING_ENABLED = os.getenv("VLLM_TPS_PROFILING", "0") == "1"
_TPS_PROFILING_EVERY = max(1, int(os.getenv("VLLM_TPS_PROFILING_EVERY",
                                            "1")))
_TPS_PROFILING_COUNTERS: dict[str, int] = defaultdict(int)
_TPS_PROFILING_LOCK = threading.Lock()


def tps_profiling_enabled() -> bool:
    return _TPS_PROFILING_ENABLED


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def tps_profile_log(stage: str, **metrics: Any) -> None:
    if not _TPS_PROFILING_ENABLED:
        return

    with _TPS_PROFILING_LOCK:
        _TPS_PROFILING_COUNTERS[stage] += 1
        count = _TPS_PROFILING_COUNTERS[stage]

    if count % _TPS_PROFILING_EVERY != 0:
        return

    metric_str = " ".join(
        f"{key}={_format_metric(value)}"
        for key, value in sorted(metrics.items()))
    logger.warning("[TPS_PROFILE] stage=%s count=%d %s", stage, count,
                   metric_str)

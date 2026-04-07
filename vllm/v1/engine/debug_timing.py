import contextvars
import datetime
import time

frontend_step_id_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "frontend_step_id", default=-1)
engine_step_id_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "engine_step_id", default=-1)


def wall_time_us() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")


def epoch_time_ns() -> int:
    return time.time_ns()

"""Delayed-SA Answer-force experiment.

The package is intentionally isolated from the historical delayed-SA pipeline.
It reads the frozen capture/probe artifacts and writes only to a new run
directory under :mod:`dp_SA.answer_force.outputs`.
"""

__all__ = ["config", "selection", "probe_runtime"]

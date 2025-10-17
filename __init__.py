"""
Next-generation mean-variance pipeline utilities.

This package provides a self-contained workflow for optimising parameters,
running backtests, producing reports, and generating daily signals — all
without modifying the legacy implementation.
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from .settings import PipelineConfig
from .data_manager import DataManager
from .optimizer import SharpeParameterOptimizer
from .reporting import ReportOrchestrator
from .daily_runner import DailySignalRunner

__all__ = [
    "PipelineConfig",
    "DataManager",
    "SharpeParameterOptimizer",
    "ReportOrchestrator",
    "DailySignalRunner",
]

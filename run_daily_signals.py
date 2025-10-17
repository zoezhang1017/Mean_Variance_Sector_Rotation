from __future__ import annotations

import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mean_variance_system.logging_config import setup_logging

try:
    from next_pipeline.data_manager import DataManager
    from next_pipeline.daily_runner import DailySignalRunner
    from next_pipeline.settings import PipelineConfig
except ImportError:  # pragma: no cover
    from .data_manager import DataManager
    from .daily_runner import DailySignalRunner
    from .settings import PipelineConfig

LOGGER = logging.getLogger("next_pipeline.run_daily")


def main() -> None:
    setup_logging()
    config = PipelineConfig()

    data_manager = DataManager(config, force_refresh=False)
    price_map = data_manager.fetch_all()

    runner = DailySignalRunner(config)
    results = runner.run(price_map)
    output_paths = runner.export(results)

    LOGGER.info(
        "Daily signals generated. Excel: %s, Heatmap: %s",
        output_paths.get("excel"),
        output_paths.get("heatmap"),
    )


if __name__ == "__main__":
    main()

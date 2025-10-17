from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mean_variance_system.logging_config import setup_logging

try:
    from next_pipeline.data_manager import DataManager
    from next_pipeline.optimizer import SharpeParameterOptimizer
    from next_pipeline.reporting import ReportOrchestrator
    from next_pipeline.settings import PipelineConfig
except ImportError:  # pragma: no cover - package execution
    from .data_manager import DataManager
    from .optimizer import SharpeParameterOptimizer
    from .reporting import ReportOrchestrator
    from .settings import PipelineConfig

LOGGER = logging.getLogger("next_pipeline.run_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise parameters, run backtests, and generate reports."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached market data and fetch from rqdatac again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    config = PipelineConfig()
    data_manager = DataManager(config, force_refresh=args.force_refresh)
    price_map = data_manager.fetch_all()

    optimizer = SharpeParameterOptimizer(config)
    optimisation_results = optimizer.optimise(price_map)
    params_path = optimizer.save_results(optimisation_results)

    reporter = ReportOrchestrator(config)
    summary_df = reporter.summary_dataframe(optimisation_results)
    reporter.save_summary(summary_df)
    reporter.build_word_report(summary_df, optimisation_results)

    LOGGER.info("Pipeline completed successfully. Parameters saved to %s", params_path)


if __name__ == "__main__":
    main()

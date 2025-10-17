# Mean Variance System 模块说明

本文件夹介绍新拆分出来的一组 Python 模块，帮助你了解各文件作用、依赖关系以及使用方式。

## 目录结构

```
mean_variance_system/
├── __init__.py
├── config.py
├── logging_config.py
├── strategy.py
├── backtest.py
├── reporting.py
└── factors/
    ├── __init__.py
    ├── technical_indicators.py
    └── factor_enhancer.py
```

## 模块说明

- `config.py`  
  集中存放策略运行所需的默认配置，包括因子增强算子、优化参数、回测配置、数据集切分比例（默认 8:1:1）、交叉验证折数/最小训练占比以及输出文件路径。可以在运行前按需覆盖其中的字典，避免直接改动核心逻辑。

- `logging_config.py`  
  提供统一的日志初始化方法 `setup_logging`，默认输出到控制台及 `logs/mean_variance.log`。策略主脚本导入后即可获得结构化的日志记录。

- `factors/technical_indicators.py`  
  原策略中的 100+ 技术指标全部整合在此类 `TechnicalIndicators` 中。每个指标方法返回 -1/0/1 信号，供策略构建持仓。

- `factors/factor_enhancer.py`  
  新增的因子增强器模块。`FactorEnhancer` 会根据配置对基础信号执行算子生成派生特征。除了传统的 z-score、EMA、滚动均值/标准差外，已内置大批非线性变换（多阶正弦余弦、多项式、双曲函数、Sigmoid、带尺度的对数/Softsign/Arctan 等）。所有算子均以名称注册，可在 `config.FACTOR_CONFIG["operators"]` 中按需增删或限定作用范围。

- `strategy.py`  
  核心的 `MeanVarianceStrategy`，负责：  
  1. 调用技术指标生成信号；  
  2. 使用 `FactorEnhancer` 做因子增强；  
  3. 根据 IC 阈值筛选因子后执行均值方差优化（支持多求解器、回退策略）；  
  4. 生成未来持仓权重和组合持仓序列。

- `backtest.py`  
  `PositionBacktest` 类独立出来处理回测逻辑：  
  - 依据仓位序列模拟交易（含交易成本）；  
  - 计算年化收益、波动率、夏普、最大回撤等指标；  
  - 返回交易记录、收益曲线以及用于 Word 报告的图形数据。

- `reporting.py`  
  封装 Word/CSV 输出流程的 `ReportManager`，支持：  
  - 创建 Word 文档并插入表格、走势图；  
  - 为每个指数生成独立子目录保存绩效指标、交易记录；  
  - 汇总最新信号与交易快照。

- `parameter_optimizer.py`  
  新增的参数优化工具，基于 `skopt` 对 `rebalance_freq`、`min_periods`、`future_days`、`ic_lookback_period` 做贝叶斯（或森林）搜索。  
  - 默认按照 8:1:1（训练/验证/测试）比例拆分时间序列，并支持时间序列交叉验证（配置 `CROSS_VALIDATION_CONFIG['folds']`）。每个折使用验证段的年化夏普比率作为目标函数，显式避免过拟合。  
  - `optimise_parameters` / `optimise_and_save` 会写入 `mean_variance_system/optimised_params.json`，`config.py` 导入时自动覆盖默认值。  
  - `evaluate_on_splits` 可在给定参数下分别回测 train/val/test，返回关键绩效指标，方便检验泛化能力。

## 主脚本 `每日择时/mean_variance_strategy_行业遍历改.py` 的用法

1. **准备环境与数据：**  
   - 确认已安装 `pypfopt`, `rqdatac`, `python-docx`, `skopt`, `matplotlib` 等依赖，并完成 `rqdatac.init()` 的账号设置。  
   - 准备好包含最少 `open`, `high`, `low`, `close`, `volume`, `prev_close` 列的日频 DataFrame（指数或标的）并按日期升序。

2. **初始化日志（可选但推荐）：**  
   ```python
   from mean_variance_system.logging_config import setup_logging
   logger = setup_logging()
   ```
   这一步会确保后续所有模块共享统一的日志输出。

2. **运行批量策略：**  
   脚本入口处已经封装了 `run_strategy_batch`，默认从 Excel `行业最佳参数(加入窗口大小).xlsx` 读取参数并遍历全部指数。直接执行脚本即可开始跑数：
   ```bash
   python 每日择时/mean_variance_strategy_行业遍历改.py
   ```

3. **调整配置：**  
   - 若需要修改优化窗口、IC 阈值、算子组合等，在运行前覆盖 `config.FACTOR_CONFIG` 或 `config.OPTIMISATION_CONFIG` 对应键值即可。  
   - 示例：
     ```python
     from mean_variance_system.config import FACTOR_CONFIG

     custom_config = FACTOR_CONFIG.copy()
     custom_config["ic_threshold"] = 0.05
     custom_config["operators"] = [
         {"name": "zscore", "window": 40, "suffix": "z40"}
     ]
     ```
     然后将 `custom_config` 传入 `build_strategy` 或自定义的策略实例。

4. **因子增强进阶用法：**  
   - `operator_scope` 用于限定算子仅作用于指定因子：
     ```python
     FACTOR_CONFIG["operator_scope"] = {
         "__all__": ["zscore", "rolling_mean"],
         "MACD": ["sin", "cos", "power"]
     }
     ```
   - 推荐根据业务挑选部分非线性算子，避免特征数爆炸（默认已开启全部算子，必要时请结合 `operator_scope` 精选）。

5. **调参助手：**  
   通过 `mean_variance_system.parameter_optimizer` 调用优化函数，例如：
   ```python
   from mean_variance_system.parameter_optimizer import optimise_and_save, evaluate_on_splits

   best_params = optimise_and_save(price_df, n_calls=40, method="gp")
   print(best_params)  # 已写入 optimised_params.json，后续运行自动应用
   metrics = evaluate_on_splits(price_df, best_params)
   print(metrics["test"])  # 查看测试集指标
   ```

6. **在自定义脚本中快速调用模型：**  
   ```python
   import rqdatac
   import pandas as pd
   from mean_variance_system import (
       MeanVarianceStrategy,
       optimise_and_save,
       evaluate_on_splits,
   )
   from mean_variance_system.config import FACTOR_CONFIG, OPTIMISATION_CONFIG
   from mean_variance_system.logging_config import setup_logging

   logger = setup_logging()
   rqdatac.init()
   df = rqdatac.get_price("000300.XSHG", start_date="2015-01-01", end_date="2024-12-31", frequency="1d", expect_df=True).droplevel(0)

   best_params = optimise_and_save(df, n_calls=30)
   metrics = evaluate_on_splits(df, best_params)
   print(metrics["test"])

   strategy = MeanVarianceStrategy(
       index_df=df,
       optimisation_config={**OPTIMISATION_CONFIG, **best_params},
       factor_config=FACTOR_CONFIG,
       logger=logger,
   )
   future_position_series, weights_df = strategy.generate_future_position()
   ```

7. **自动化脚本：**  
  - `automation/run_pipeline.py`：一键式整合。支持任务：
    ```bash
    python automation/run_pipeline.py --task optimize  # 针对 index_list.txt 全量调参
    python automation/run_pipeline.py --task daily     # 读取参数表生成最新信号
    python automation/run_pipeline.py --task full      # 先调参再出信号
    ```
    调参范围、默认参数、起始日期等可在 `automation/pipeline_config.json` 中调整；指数列表位于 `automation/index_list.txt`。调参完成后，脚本会自动更新 Excel 参数表并生成带有调参指标的 JSON 摘要。  
  - `automation/run_optimizer.py`：保留的单指数调参工具，便于单独测试；`automation/run_daily_signals.py` 为仅出信号的轻量版。  
  - 如需每天 00:01 自动执行，可复制 `automation/com.meanvariance.daily.plist` 到 `~/Library/LaunchAgents/` 并 `launchctl load`，或在 Linux 上添加 `cron` 任务，例如：
    ```
    1 0 * * * /usr/bin/python3 /Users/apple/Desktop/zyjj/automation/run_pipeline.py --task daily >> /Users/apple/Desktop/zyjj/logs/daily.log 2>&1
    ```
    请根据实际 Python 路径、日志位置调整命令。

## 注意事项

- 运行脚本前请确保环境中已安装 `pypfopt`, `rqdatac`, `python-docx`, `skopt`, `matplotlib` 等依赖。
- 日志默认写入 `logs/mean_variance.log`，如需调整路径可在 `setup_logging` 调用时传入自定义 `log_path`。
- Word/CSV 输出默认落在 `config.REPORTING_CONFIG["output_dir"]`，如需改成其他路径（例如同步到网盘），请修改配置后再运行。

如需进一步扩展，例如新增自定义算子或替换优化器，可直接在对应模块新增函数／类，并通过配置注入。欢迎根据业务需求继续演化该模块体系。

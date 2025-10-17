from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PositionBacktest:
    def __init__(
        self,
        position_series: pd.Series,
        price_df: pd.DataFrame,
        cost_rate: float = 0.0005,
        risk_free_rate: float = 0.02,
        logger: logging.Logger | None = None,
    ):
        """
        初始化回测类
        
        参数:
        position_series: pandas Series, 每日仓位比例 (0-1之间)
        price_df: pandas DataFrame, 至少包含'open'和'close'列
        cost_rate: float, 交易成本率 (默认万五)
        risk_free_rate: float, 无风险利率 (默认2%)
        """
        self.logger = logger or logging.getLogger("mean_variance")
        self.position = position_series.copy()
        price_df = price_df.copy()
        price_df['return'] = (price_df['close'] - price_df['prev_close']) / price_df['prev_close']
        self.price_df = price_df
        self.cost_rate = cost_rate
        self.risk_free_rate = risk_free_rate
        self.trade_records = None
        self.daily_returns = None
        self.results = {}
        self.logger.info("PositionBacktest initialised with %d trading days", len(self.position))
        
    def run_backtest(self) -> pd.Series:
        """运行回测"""
        # 确保数据对齐
        common_index = self.position.index.intersection(self.price_df.index)
        self.position = self.position.loc[common_index]
        self.price_df = self.price_df.loc[common_index]
        
        # 初始化变量
        n_days = len(self.position)
        cash = 1.0  # 初始现金
        shares = 0.0  # 持有份额
        prev_position = 0.0  # 前一日仓位
        
        # 记录每日数据
        self.daily_values = pd.Series(index=self.position.index, dtype=float)
        self.trade_records = []
        
        for i, date in enumerate(self.position.index):
            open_price = self.price_df.loc[date, 'open']
            close_price = self.price_df.loc[date, 'close']
            target_position = self.position.loc[date]
            
            # 计算前一日资产总值
            if i == 0:
                prev_value = 1.0
            else:
                prev_value = self.daily_values.iloc[i-1]
            
            # 计算需要调整的仓位
            position_diff = target_position - prev_position
            trade_value = position_diff * prev_value
            
            # 处理交易
            if trade_value > 0:  # 买入
                trade_shares = trade_value / open_price
                cost = trade_value * self.cost_rate
                shares += trade_shares
                cash -= trade_value + cost
                self.trade_records.append({
                    'date': date, 'action': 'buy', 'trade_shares': trade_shares,
                    'trade_price': open_price, 'trade_value': trade_value, 'transaction_cost': cost, 
                    'position_diff': position_diff
                })
            elif trade_value < 0:  # 卖出
                trade_shares = abs(trade_value) / open_price
                cost = abs(trade_value) * self.cost_rate
                shares -= trade_shares
                cash += abs(trade_value) - cost
                self.trade_records.append({
                    'date': date, 'action': 'sell', 'trade_shares': trade_shares,
                    'trade_price': open_price, 'trade_value': abs(trade_value), 'transaction_cost': cost,
                    'position_diff': position_diff
                })
            
            # 计算当日资产总值
            current_value = cash + shares * close_price
            self.daily_values[date] = current_value
            
            # 更新前一日仓位
            prev_position = target_position
            self.logger.debug(
                "Date %s: target %.2f, diff %.4f, cash %.6f, shares %.4f",
                date, target_position, position_diff, cash, shares
            )
        
        # 计算每日收益率
        self.daily_returns = self.daily_values.pct_change().fillna(0)
        if not self.daily_values.empty:
            total_return = self.daily_values.iloc[-1] - 1
            self.logger.info("Backtest complete, total return %.2f%%", total_return * 100)
        
        return self.daily_returns
    
    def calculate_metrics(self, returns):
        """计算策略指标"""
        # 年化收益率
        annual_return = (1 + returns.mean()) ** 252 - 1
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown)
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        avg_profit = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        profit_ratio = avg_profit / avg_loss if avg_loss != 0 else np.nan
        
        return {
            '年化收益': f"{annual_return:.2%}",
            '年化波动率': f"{annual_volatility:.2%}",
            '最大回撤': f"{max_drawdown:.2%}",
            '夏普比率': f"{sharpe_ratio:.2}",
            '卡玛比率': f"{calmar_ratio:.2}",
            '胜率': f"{win_rate:.2%}",
            '盈亏比': f"{profit_ratio:.2}"
        }
    
    def analyze_results(self) -> pd.DataFrame:
        """分析回测结果"""
        # 分年度结果
        yearly_returns = self.daily_returns.groupby(self.daily_returns.index.year)
        for year, returns in yearly_returns:
            if len(returns) > 10:  # 确保有足够的数据点
                yearly_metrics = self.calculate_metrics(returns)
                self.results[year] = yearly_metrics
        
        # 全样本结果
        full_sample_metrics = self.calculate_metrics(self.daily_returns)
        self.results['全样本'] = full_sample_metrics
        
        # 转换为DataFrame
        results_df = pd.DataFrame(self.results).T
        self.logger.info("Generated %d annual performance entries", len(results_df))
        return results_df
    
    def run_full_backtest(self):
        """完整回测流程"""
        self.run_backtest()
        results_df = self.analyze_results()
        return results_df

    def get_trade_records(self):
        """获取交易记录"""
        if self.trade_records is None:
            self.run_backtest()
        return pd.DataFrame(self.trade_records).set_index('date')
    
    def get_ret(self):
        """获取每日收益率序列"""
        if self.daily_returns is None:
            self.run_backtest()
        return self.daily_returns.sum()

    def calculate_metrics_index(self):
        """计算标的绩效指标"""
        
        # 准备结果容器
        results = []
        index_df = self.price_df['return']
        all_years = index_df.index.year.unique()
        
        # 计算各年度指标
        for year in all_years:
            year_returns = index_df[index_df.index.year == year]
            if len(year_returns) == 0:
                continue
                
            metrics = self.calculate_metrics(year_returns)
            metrics['时间'] = str(year)
            results.append(metrics)
        
        # 计算全周期指标
        all_period_metrics = self.calculate_metrics(index_df)
        all_period_metrics['时间'] = '全样本'
        results.append(all_period_metrics)
        
        # 转换为DataFrame
        metrics_df = pd.DataFrame(results).set_index('时间')
        return metrics_df
    
    def plot_performance(self) -> io.BytesIO:
        """绘制策略表现图并返回图像流。"""
        if self.daily_returns is None:
            raise ValueError("请先运行回测")
        
        index_nav = self.price_df['close'] / self.price_df['close'].iloc[0]

        plt.figure(figsize=(10, 7))
        
        # 绘制资产曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.daily_values.index, self.daily_values, label='组合价值')
        plt.plot(index_nav.index, index_nav, label='基准')
        plt.title('组合价值 vs 基准')
        plt.legend()
        
        # 绘制每日收益率
        plt.subplot(2, 1, 2)
        plt.bar(self.daily_returns.index, self.daily_returns, 
                color=np.where(self.daily_returns >= 0, 'g', 'r'))
        plt.title('每日收益率')
        
        plt.tight_layout()
        
        # 将图表保存到内存中的字节流
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
        img_stream.seek(0)  # 将流指针移到开头
        plt.close()
        return img_stream

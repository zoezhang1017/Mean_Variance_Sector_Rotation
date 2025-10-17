import numpy as np
import pandas as pd

class TechnicalIndicators:
    def __init__(self, data):
        """
        初始化技术指标计算类
        :param data: DataFrame，包含以下列：'open', 'high', 'low', 'close', 'volume'
        """
        self.data = data.copy()
        self._precalculate()
    
    def _precalculate(self):
        """预计算常用中间变量"""
        df = self.data
        df['ref_close_1'] = df['close'].shift(1)
        df['ref_open_1'] = df['open'].shift(1)
        df['ref_low_1'] = df['low'].shift(1)
        df['ref_high_1'] = df['high'].shift(1)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['ref_close_1']), 
                                   abs(df['low'] - df['ref_close_1'])))
    
    # ================= 辅助函数 =================
    def _cross_above(self, series1, series2):
        """判断series1是否上穿series2"""
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    
    def _cross_below(self, series1, series2):
        """判断series1是否下穿series2"""
        if isinstance(series2, (int, float)):
            series2 = pd.Series(series2, index=series1.index)
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    
    def _ma(self, series, window):
        """简单移动平均"""
        return series.rolling(window).mean()
    
    def _ema(self, series, window):
        """指数移动平均"""
        return series.ewm(span=window, adjust=False).mean()

    def _sma(self, series, N, M):
        """加权简单移动平均"""
        return series.ewm(alpha=M/N, adjust=False).mean()

    # ================= 价格动量指标 =================
    def DPO(self, N=20):
        """
        DPO指标 - 价格与延迟移动平均线的差值
        信号: DPO上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        MA = self._ma(close, N)
        DPO = close - MA.shift(N//2 + 1)
        
        signals = np.where(self._cross_above(DPO, 0), 1,
                          np.where(self._cross_below(DPO, 0), -1, 0))
        return signals

    def ER(self, N=20):
        """
        ER指标 - 衡量多空力量对比
        信号: BearPower上穿0买入(1), BullPower下穿0卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        EMA = self._ema(close, N)
        BullPower = high - EMA
        BearPower = low - EMA
        
        signals = np.where(self._cross_above(BearPower, 0), 1,
                          np.where(self._cross_below(BullPower, 0), -1, 0))
        return signals

    def PO(self, N1=9, N2=26):
        """
        PO指标 - 短期均线与长期均线变化率
        信号: PO上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        EMA_short = self._ema(close, N1)
        EMA_long = self._ema(close, N2)
        PO = (EMA_short - EMA_long) / EMA_long * 100
        
        signals = np.where(self._cross_above(PO, 0), 1,
                          np.where(self._cross_below(PO, 0), -1, 0))
        return signals

    def TII(self, N1=40, N2=9):
        """
        TII指标 - 类似RSI但基于价格与均线差值
        信号: TII上穿TII_SIGNAL买入(1), 下穿卖出(-1)
        """
        df = self.data
        close = df['close']
        M = N1 // 2 + 1
        CLOSE_MA = self._ma(close, N1)
        DEV = close - CLOSE_MA
        DEVPOS = np.where(DEV > 0, DEV, 0)
        DEVNEG = np.where(DEV < 0, -DEV, 0)
        SUMPOS = pd.Series(DEVPOS).rolling(M).sum()
        SUMNEG = pd.Series(DEVNEG).rolling(M).sum()
        TII = 100 * SUMPOS / (SUMPOS + SUMNEG + 1e-8)
        TII_SIGNAL = self._ema(TII, N2)
        
        signals = np.where(self._cross_above(TII, TII_SIGNAL), 1,
                          np.where(self._cross_below(TII, TII_SIGNAL), -1, 0))
        return signals

    # ================= 价格反转指标 =================
    def RSI(self, N=14):
        """
        RSI指标 - 相对强弱指数
        信号: RSI上穿40买入(1), 下穿60卖出(-1)
        """
        df = self.data
        close = df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = self._ema(gain, N)
        avg_loss = self._ema(loss, N)
        rs = avg_gain / (avg_loss + 1e-8)
        RSI = 100 - (100 / (1 + rs))
        
        signals = np.where(self._cross_above(RSI, 40), 1,
                          np.where(self._cross_below(RSI, 60), -1, 0))
        return signals

    def KDJ(self, N=40, M=3):
        """
        KDJ指标 - 随机指标
        信号: D<20且K上穿D买入(1), D>80且K下穿D卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        LOW_N = low.rolling(N).min()
        HIGH_N = high.rolling(N).max()
        RSV = (close - LOW_N) / (HIGH_N - LOW_N) * 100
        K = self._ema(RSV, M)
        D = self._ema(K, M)
        
        buy_cond = (D < 20) & self._cross_above(K, D)
        sell_cond = (D > 80) & self._cross_below(K, D)
        signals = np.where(buy_cond, 1,
                          np.where(sell_cond, -1, 0))
        return signals

    # ================= 成交量指标 =================
    def OBV(self):
        """
        OBV指标 - 能量潮
        信号: OBV上穿其均线买入(1), 下穿卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_ma = self._ma(obv, 20)
        
        signals = np.where(self._cross_above(obv, obv_ma), 1,
                          np.where(self._cross_below(obv, obv_ma), -1, 0))
        return signals

    def VWAP(self, N=20):
        """
        VWAP指标 - 成交量加权平均价
        信号: 价格上穿VWAP买入(1), 下穿卖出(-1)
        """
        df = self.data
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        typical = (high + low + close) / 3
        vwap = (typical * volume).rolling(N).sum() / volume.rolling(N).sum()
        
        signals = np.where(self._cross_above(close, vwap), 1,
                          np.where(self._cross_below(close, vwap), -1, 0))
        return signals

    # ================= 价量指标 =================
    def CMF(self, N=20):
        """
        CMF指标 - 资金流量指标
        信号: CMF上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        clv = ((close - low) - (high - close)) / (high - low)
        cmf = (clv * volume).rolling(N).sum() / volume.rolling(N).sum()
        
        signals = np.where(self._cross_above(cmf, 0), 1,
                          np.where(self._cross_below(cmf, 0), -1, 0))
        return signals

    def MFI(self, N=14):
        """
        MFI指标 - 资金流量指数
        信号: MFI上穿20买入(1), 下穿80卖出(-1)
        """
        df = self.data
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        typical = (high + low + close) / 3
        money_flow = typical * volume
        pos_mf = np.where(typical.diff() > 0, money_flow, 0)
        neg_mf = np.where(typical.diff() < 0, money_flow, 0)
        mfr = (pd.Series(pos_mf).rolling(N).sum() / 
               (pd.Series(neg_mf).rolling(N).sum() + 1e-8))
        mfi = 100 - (100 / (1 + mfr))
        
        signals = np.where(self._cross_above(mfi, 20), 1,
                          np.where(self._cross_below(mfi, 80), -1, 0))
        return signals

    # 继续添加更多指标...
    def ADTM(self, N=20):
        """
        ADTM指标 - 动态人气指标
        信号: ADTM上穿0.5买入(1), 下穿-0.5卖出(-1)
        """
        df = self.data
        open_ = df['open']
        ref_open_1 = df['ref_open_1']
        high, low = df['high'], df['low']
        
        DTM = np.where(open_ > ref_open_1, 
                      np.maximum(high - open_, open_ - ref_open_1), 0)
        DBM = np.where(open_ < ref_open_1, 
                      np.maximum(open_ - low, ref_open_1 - open_), 0)
        STM = pd.Series(DTM).rolling(N).sum()
        SBM = pd.Series(DBM).rolling(N).sum()
        ADTM = (STM - SBM) / np.maximum(STM, SBM)
        
        signals = np.where(self._cross_above(ADTM, 0.5), 1,
                          np.where(self._cross_below(ADTM, -0.5), -1, 0))
        return signals

    def ZLMACD(self, N1=20, N2=100):
        """
        ZLMACD指标 - 改进版MACD
        信号: ZLMACD上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        EMA1 = 2 * self._ema(close, N1) - self._ema(self._ema(close, N1), N1)
        EMA2 = 2 * self._ema(close, N2) - self._ema(self._ema(close, N2), N2)
        ZLMACD = EMA1 - EMA2
        
        signals = np.where(self._cross_above(ZLMACD, 0), 1,
                          np.where(self._cross_below(ZLMACD, 0), -1, 0))
        return signals

    def POS(self, N=100):
        """
        POS指标 - 价格位置指标
        信号: POS上穿80买入(1), 下穿20卖出(-1)
        """
        df = self.data
        close = df['close']
        PRICE = (close - close.shift(N)) / close.shift(N)
        POS = (PRICE - PRICE.rolling(N).min()) / \
              (PRICE.rolling(N).max() - PRICE.rolling(N).min())
        
        signals = np.where(self._cross_above(POS, 80), 1,
                          np.where(self._cross_below(POS, 20), -1, 0))
        return signals

    def PAC(self, N1=20, N2=20):
        """
        PAC指标 - 价格通道指标
        信号: 收盘价上穿UPPER买入(1), 下穿LOWER卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        UPPER = self._sma(high, N1, 1)
        LOWER = self._sma(low, N2, 1)
        
        signals = np.where(self._cross_above(close, UPPER), 1,
                          np.where(self._cross_below(close, LOWER), -1, 0))
        return signals

    def TRIX(self, N=20):
        """
        TRIX指标 - 三重指数平滑移动平均
        信号: TRIX上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        TRIPLE_EMA = self._ema(self._ema(self._ema(close, N), N), N)
        TRIX = (TRIPLE_EMA - TRIPLE_EMA.shift(1)) / (TRIPLE_EMA.shift(1) + 1e-8)
        
        signals = np.where(self._cross_above(TRIX, 0), 1,
                          np.where(self._cross_below(TRIX, 0), -1, 0))
        return signals

    def WC(self, N1=20, N2=40):
        """
        WC指标 - 加权收盘价均线交叉
        信号: EMA1上穿EMA2买入(1), 下穿EMA2卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        WC = (high + low + 2 * close) / 4
        EMA1 = self._ema(WC, N1)
        EMA2 = self._ema(WC, N2)
        
        signals = np.where(self._cross_above(EMA1, EMA2), 1,
                          np.where(self._cross_below(EMA1, EMA2), -1, 0))
        return signals

    def ADX(self, N=14):
        """
        ADX指标 - 平均趋向指数
        信号: DI+上穿DI-买入(1), DI+下穿DI-卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        MAX_HIGH = np.where(high > df['ref_high_1'], high - df['ref_high_1'], 0)
        MAX_LOW = np.where(df['ref_low_1'] > low, df['ref_low_1'] - low, 0)
        XPDM = np.where(MAX_HIGH > MAX_LOW, high - df['ref_high_1'], 0)
        PDM = pd.Series(XPDM).rolling(N).sum()
        XNDM = np.where(MAX_LOW > MAX_HIGH, df['ref_low_1'] - low, 0)
        NDM = pd.Series(XNDM).rolling(N).sum()
        values = np.column_stack((abs(high - low).values, abs(high - close).values, abs(low - close).values))
        TR = np.max(values, axis=1)
        TR = pd.Series(TR).rolling(N).sum()
        DI_POS = PDM / TR
        DI_NEG = NDM / TR
        
        signals = np.where(self._cross_above(DI_POS, DI_NEG), 1,
                          np.where(self._cross_below(DI_POS, DI_NEG), -1, 0))
        return signals

    def FISHER(self, N=20, PARAM=0.3):
        """
        FISHER指标 - 费雪变换
        信号: FISHER上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low = df['high'], df['low']
        PRICE = (high + low) / 2
        PRICE_CH = 2 * ((PRICE - low.rolling(N).min()) / 
                        (high.rolling(N).max() - low.rolling(N).min() + 1e-8)) - 0.5
        PRICE_CH = pd.Series(np.where(PRICE_CH > 0.99, 0.999, 
                           np.where(PRICE_CH < -0.99, -0.999, PRICE_CH)))
        PRICE_CHANGE = PARAM * PRICE_CH + (1 - PARAM) * PRICE_CH.shift(1)
        FISHER = 0.5 * PRICE_CHANGE.shift(1) + 0.5 * np.log((1 + PRICE_CHANGE) / (1 - PRICE_CHANGE + 1e-8))
        
        signals = np.where(self._cross_above(FISHER, 0), 1,
                          np.where(self._cross_below(FISHER, 0), -1, 0))
        return signals

    def Demarker(self, N=20):
        """
        Demarker指标 - 德马克指标
        信号: Demarker上穿0.7买入(1), 下穿0.3卖出(-1)
        """
        df = self.data
        high, low = df['high'], df['low']
        Demax = high - df['ref_high_1']
        Demax = Demax.where(Demax > 0,  0)
        Demin = df['ref_low_1'] - low
        Demin = Demin.where(Demin > 0, 0)
        Demarker = self._ma(Demax, N) / (self._ma(Demax, N) + self._ma(Demin, N) + 1e-8)
        
        signals = np.where(self._cross_above(Demarker, 0.7), 1,
                          np.where(self._cross_below(Demarker, 0.3), -1, 0))
        return signals

    def TDI(self, N1=20, N2=5, N3=7, N4=20):
        """
        TDI指标 - 交易方向指标
        信号: RSI价格线上穿信号线和市场基线买入(1), 下穿卖出(-1)
        """
        df = self.data
        close = df['close']
        RSI = pd.Series(self.RSI(N1))
        RSI_PriceLine = self._ema(RSI, N2)
        RSI_SignalLine = self._ema(RSI, N3)
        RSI_MarketLine = self._ema(RSI, N4)
        
        buy_cond = self._cross_above(RSI_PriceLine, RSI_SignalLine) & \
                  self._cross_above(RSI_PriceLine, RSI_MarketLine)
        sell_cond = self._cross_below(RSI_PriceLine, RSI_SignalLine) & \
                   self._cross_below(RSI_PriceLine, RSI_MarketLine)
        signals = np.where(buy_cond, 1,
                          np.where(sell_cond, -1, 0))
        return signals

    def IC(self, N1=9, N2=26, N3=52):
        """
        IC指标 - 一目均衡表
        信号: 价格在云上方且SPAN_A>SPAN_B时上穿KS买入(1),
             价格在云下方且SPAN_A<SPAN_B时下穿KS卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TS = (high.rolling(N1).max() + low.rolling(N1).min()) / 2
        KS = (high.rolling(N2).max() + low.rolling(N2).min()) / 2
        SPAN_A = (TS + KS) / 2
        SPAN_B = (high.rolling(N3).max() + low.rolling(N3).min()) / 2
        
        above_cloud = (close > SPAN_A) & (close > SPAN_B)
        below_cloud = (close < SPAN_A) & (close < SPAN_B)
        buy_cond = above_cloud & (SPAN_A > SPAN_B) & self._cross_above(close, KS)
        sell_cond = below_cloud & (SPAN_A < SPAN_B) & self._cross_below(close, KS)
        signals = np.where(buy_cond, 1,
                          np.where(sell_cond, -1, 0))
        return signals

    def TSI(self, N1=25, N2=13):
        """
        TSI指标 - 真实强度指数
        信号: TSI上穿10买入(1), 下穿-10卖出(-1)
        """
        df = self.data
        close = df['close']
        diff = close.diff()
        EMA1 = self._ema(diff, N1)
        EMA2 = self._ema(EMA1, N2)
        EMA3 = self._ema(abs(diff), N1)
        EMA4 = self._ema(EMA3, N2)
        TSI = 100 * EMA2 / (EMA4 + 1e-8)
        
        signals = np.where(self._cross_above(TSI, 10), 1,
                          np.where(self._cross_below(TSI, -10), -1, 0))
        return signals

    def IMI(self, N=14):
        """
        IMI指标 - 日内动量指数
        信号: IMI上穿80买入(1), 下穿20卖出(-1)
        """
        df = self.data
        close, open_ = df['close'], df['open']
        INC = (close - open_).where(close > open_,  0).rolling(N).sum()
        DEC = (open_ - close).where(open_ > close,  0).rolling(N).sum()
        IMI = INC / (INC + DEC + 1e-8)
        
        signals = np.where(self._cross_above(IMI, 80), 1,
                          np.where(self._cross_below(IMI, 20), -1, 0))
        return signals

    def CMO(self, N=20):
        """
        CMO指标 - 钱德动量摆动指标
        信号: CMO上穿30买入(1), 下穿-30卖出(-1)
        """
        df = self.data
        close = df['close']
        SU = close.diff().where(close.diff() > 0, 0).rolling(N).sum()
        SD = -close.diff().where(close.diff() < 0, 0).rolling(N).sum()
        CMO = (SU - SD) / (SU + SD + 1e-8) * 100
        
        signals = np.where(self._cross_above(CMO, 30), 1,
                          np.where(self._cross_below(CMO, -30), -1, 0))
        return signals

    def OSC(self, N=40, M=20):
        """
        OSC指标 - 震荡指标
        信号: OSC上穿OSCMA买入(1), 下穿OSCMA卖出(-1)
        """
        df = self.data
        close = df['close']
        OSC = close - self._ma(close, N)
        OSCMA = self._ma(OSC, M)
        
        signals = np.where(self._cross_above(OSC, OSCMA), 1,
                          np.where(self._cross_below(OSC, OSCMA), -1, 0))
        return signals

    def MACD(self, N1=20, N2=40, N3=5):
        """
        MACD指标 - 指数平滑异同移动平均线
        信号: MACD上穿SIGNAL买入(1), 下穿SIGNAL卖出(-1)
        """
        df = self.data
        close = df['close']
        MACD = self._ema(close, N1) - self._ema(close, N2)
        SIGNAL = self._ema(MACD, N3)
        
        signals = np.where(self._cross_above(MACD, SIGNAL), 1,
                          np.where(self._cross_below(MACD, SIGNAL), -1, 0))
        return signals

    def HA(self):
        """
        HA指标 - 平均K线图
        信号: HA_CLOSE上穿HA_OPEN买入(1), 下穿HA_OPEN卖出(-1)
        """
        df = self.data
        close, open_, high, low = df['close'], df['open'], df['high'], df['low']
        HA_CLOSE = (open_ + high + low + close) / 4
        HA_OPEN = (open_.shift(1) + close.shift(1)) / 2
        # HA_HIGH = np.maximum([high, HA_OPEN, HA_CLOSE])
        # HA_LOW = np.minimum([low, HA_OPEN, HA_CLOSE])
        
        signals = np.where(self._cross_above(HA_CLOSE, HA_OPEN), 1,
                          np.where(self._cross_below(HA_CLOSE, HA_OPEN), -1, 0))
        return signals

    def CLV(self, N=60):
        """
        CLV指标 - 收盘位置指标
        信号: CLVMA上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        CLV = (2 * close - low - high) / (high - low + 1e-8)
        CLVMA = self._ma(CLV, N)
        
        signals = np.where(self._cross_above(CLVMA, 0), 1,
                          np.where(self._cross_below(CLVMA, 0), -1, 0))
        return signals

    def WAD(self, N=20):
        """
        WAD指标 - 威廉姆斯累积/派发线
        信号: WAD上穿WADMA买入(1), 下穿WADMA卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TRH = np.maximum(high, df['ref_close_1'])
        TRL = np.minimum(low, df['ref_close_1'])
        AD = np.where(close > df['ref_close_1'], close - TRL, 
                     np.where(close < df['ref_close_1'], close - TRH, 0))
        WAD = pd.Series(AD).cumsum()
        WADMA = self._ma(WAD, N)
        
        signals = np.where(self._cross_above(WAD, WADMA), 1,
                          np.where(self._cross_below(WAD, WADMA), -1, 0))
        return signals

    def BIAS36(self, N=6):
        """
        BIAS36指标 - 三六乖离
        信号: BIAS36上穿MABIAS36买入(1), 下穿MABIAS36卖出(-1)
        """
        df = self.data
        close = df['close']
        BIAS36 = self._ma(close, 3) - self._ma(close, 6)
        MABIAS36 = self._ma(BIAS36, N)
        
        signals = np.where(self._cross_above(BIAS36, MABIAS36), 1,
                          np.where(self._cross_below(BIAS36, MABIAS36), -1, 0))
        return signals

    def TEMA(self, N1=20, N2=40):
        """
        TEMA指标 - 三重指数移动平均
        信号: 短期TEMA上穿长期TEMA买入(1), 下穿卖出(-1)
        """
        df = self.data
        close = df['close']
        TEMA1 = 3*self._ema(close, N1) - 3*self._ema(self._ema(close, N1), N1) + \
               self._ema(self._ema(self._ema(close, N1), N1), N1)
        TEMA2 = 3*self._ema(close, N2) - 3*self._ema(self._ema(close, N2), N2) + \
               self._ema(self._ema(self._ema(close, N2), N2), N2)
        
        signals = np.where(self._cross_above(TEMA1, TEMA2), 1,
                          np.where(self._cross_below(TEMA1, TEMA2), -1, 0))
        return signals

    def REG(self, N=40):
        """
        REG指标 - 回归通道
        信号: REG上穿0.05买入(1), 下穿-0.05卖出(-1)
        """
        df = self.data
        close = df['close']
        X = np.arange(N)
        Y = close.rolling(N).apply(lambda x: np.polyfit(X, x, 1)[0] * (N-1) + np.polyfit(X, x, 1)[1])
        REG = (close - Y) / (Y + 1e-8)
        
        signals = np.where(self._cross_above(REG, 0.05), 1,
                          np.where(self._cross_below(REG, -0.05), -1, 0))
        return signals

    def ATR(self, N=14, M=7):
        """
        ATR指标 - 平均真实波幅
        信号: 收盘价上穿UPPER买入(1), 下穿LOWER卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TR = np.maximum(abs(high - low), 
                       np.maximum(abs(high - df['ref_close_1']), 
                                 abs(low - df['ref_close_1'])))
        ATR = self._sma(TR, N, 1)
        UPPER = low.rolling(M).min() + 3 * ATR
        LOWER = high.rolling(M).max() - 3 * ATR
        
        signals = np.where(self._cross_above(close, UPPER), 1,
                          np.where(self._cross_below(close, LOWER), -1, 0))
        return signals

    def PSY(self, N=12):
        """
        PSY指标 - 心理线
        信号: PSY上穿60买入(1), 下穿40卖出(-1)
        """
        df = self.data
        close = df['close']
        PSY = (close.diff() > 0).rolling(N).sum() / N * 100
        
        signals = np.where(self._cross_above(PSY, 60), 1,
                          np.where(self._cross_below(PSY, 40), -1, 0))
        return signals

    def DMA(self, N1=10, N2=50, N3=10):
        """
        DMA指标 - 平行线差
        信号: DMA上穿AMA买入(1), 下穿AMA卖出(-1)
        """
        df = self.data
        close = df['close']
        DMA = self._ma(close, N1) - self._ma(close, N2)
        AMA = self._ma(DMA, N3)
        
        signals = np.where(self._cross_above(DMA, AMA), 1,
                          np.where(self._cross_below(DMA, AMA), -1, 0))
        return signals

    def KST(self, N1=10, N2=15, N3=20, N4=30, M=9):
        """
        KST指标 - 确然指标
        信号: KST上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        ROC_MA1 = self._ma(close - close.shift(10), 10)
        ROC_MA2 = self._ma(close - close.shift(15), 10)
        ROC_MA3 = self._ma(close - close.shift(20), 10)
        ROC_MA4 = self._ma(close - close.shift(30), 10)
        KST = ROC_MA1 + ROC_MA2 * 2 + ROC_MA3 * 3 + ROC_MA4 * 4
        KST_SIGNAL = self._ma(KST, M)
        
        signals = np.where(self._cross_above(KST, KST_SIGNAL), 1,
                          np.where(self._cross_below(KST, KST_SIGNAL), -1, 0))
        return signals

    def MICD(self, N=20, N1=10, N2=20, M=10):
        """
        MICD指标 - 动量指标交叉差异
        信号: MICD上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        MI = close - close.shift(1)
        MTMMA = self._sma(MI, N, 1)
        DIF = self._ma(MTMMA.shift(1), N1) - self._ma(MTMMA.shift(1), N2)
        MICD = self._sma(DIF, M, 1)
        
        signals = np.where(self._cross_above(MICD, 0), 1,
                          np.where(self._cross_below(MICD, 0), -1, 0))
        return signals

    def PMO(self, N1=10, N2=40, N3=20):
        """
        PMO指标 - 价格动量振荡器
        信号: PMO上穿PMO_SIGNAL买入(1), 下穿PMO_SIGNAL卖出(-1)
        """
        df = self.data
        close = df['close']
        ROC = (close - close.shift(1)) / close.shift(1) * 100
        ROC_MA = self._ema(ROC, N1)
        ROC_MA10 = ROC_MA * 10
        PMO = self._ema(ROC_MA10, N2)
        PMO_SIGNAL = self._ema(PMO, N3)
        
        signals = np.where(self._cross_above(PMO, PMO_SIGNAL), 1,
                          np.where(self._cross_below(PMO, PMO_SIGNAL), -1, 0))
        return signals

    def RCCD(self, M=40, N1=20, N2=40, T=17):
        """
        RCCD指标 - 变化率交叉差异
        信号: RCCD上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        RC = close / close.shift(M)
        ARC1 = self._sma(RC.shift(1), M, 1)
        DIF = self._ma(ARC1.shift(1), N1) - self._ma(ARC1.shift(1), N2)
        RCCD = self._sma(DIF, T, 1)
        
        signals = np.where(self._cross_above(RCCD, 0), 1,
                          np.where(self._cross_below(RCCD, 0), -1, 0))
        return signals

    def KAMA(self, N=10, N1=2, N2=30):
        """
        KAMA指标 - 考夫曼自适应移动平均
        信号: 收盘价上穿KAMA买入(1), 下穿KAMA卖出(-1)
        """
        df = self.data
        close = df['close']
        DIRECTION = close - close.shift(N)
        VOLATILITY = abs(close.diff()).rolling(N).sum()
        ER = DIRECTION / (VOLATILITY + 1e-8)
        FAST = 2 / (N1 + 1)
        SLOW = 2 / (N2 + 1)
        SMOOTH = ER * (FAST - SLOW) + SLOW
        COF = SMOOTH * SMOOTH
        KAMA = COF * close + (1 - COF) * close.shift(1)
        
        signals = np.where(self._cross_above(close, KAMA), 1,
                          np.where(self._cross_below(close, KAMA), -1, 0))
        return signals

    def DZCCI(self, N=40, M=3, PARAM=1.5):
        """
        DZCCI指标 - 动态CCI
        信号: CCI上穿CCI_UPPER买入(1), 下穿CCI_LOWER卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TP = (high + low + close) / 3
        CCI = (TP - self._ma(TP, N)) / (0.015 * TP.rolling(N).std())
        CCI_MIDDLE = self._ma(CCI, N)
        CCI_UPPER = CCI_MIDDLE + PARAM * CCI.rolling(N).std()
        CCI_LOWER = CCI_MIDDLE - PARAM * CCI.rolling(N).std()
        CCI_MA = self._ma(CCI, M)
        
        signals = np.where(self._cross_above(CCI_MA, CCI_UPPER), 1,
                          np.where(self._cross_below(CCI_MA, CCI_LOWER), -1, 0))
        return signals

    def ADXR(self, N=14):
        """
        ADXR指标 - 平均趋向指数评级
        信号: ADX上穿ADXR买入(1), 下穿ADXR卖出(-1)
        """
        df = self.data
        ADX = pd.Series(self.ADX(N))
        ADXR = (ADX + ADX.shift(N)) / 2
        
        signals = np.where(self._cross_above(ADX, ADXR), 1,
                          np.where(self._cross_below(ADX, ADXR), -1, 0))
        return signals

    def PPO(self, N1=12, N2=26, N3=9):
        """
        PPO指标 - 百分比价格振荡器
        信号: PPO上穿PPO_SIGNAL买入(1), 下穿PPO_SIGNAL卖出(-1)
        """
        df = self.data
        close = df['close']
        PPO = (self._ema(close, N1) - self._ema(close, N2)) / self._ema(close, N2) * 100
        PPO_SIGNAL = self._ema(PPO, N3)
        
        signals = np.where(self._cross_above(PPO, PPO_SIGNAL), 1,
                          np.where(self._cross_below(PPO, PPO_SIGNAL), -1, 0))
        return signals

    def AWS(self, N1=12, N2=26, N3=9):
        """
        AWS指标 - 自适应加权移动平均
        信号: AWS上穿AWS_SIGNAL买入(1), 下穿AWS_SIGNAL卖出(-1)
        """
        df = self.data
        high, low = df['high'], df['low']
        AWS = (self._ema((high + low)/2, N1) - self._ema((high + low)/2, N2)) / \
              self._ema((high + low)/2, N2) * 100
        AWS_SIGNAL = self._ema(AWS, N3)
        
        signals = np.where(self._cross_above(AWS, AWS_SIGNAL), 1,
                          np.where(self._cross_below(AWS, AWS_SIGNAL), -1, 0))
        return signals

    def SMI(self, N1=20, N2=20, N3=20):
        """
        SMI指标 - 随机动量指数
        信号: SMI上穿SMIMA买入(1), 下穿SMIMA卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        M = (high.rolling(N1).max() + low.rolling(N1).min()) / 2
        D = close - M
        DS = self._ema(self._ema(D, N2), N2)
        DHL = self._ema(self._ema(high.rolling(N1).max() - low.rolling(N1).min(), N2), N2)
        SMI = 100 * DS / (DHL + 1e-8)
        SMIMA = self._ma(SMI, N3)
        
        signals = np.where(self._cross_above(SMI, SMIMA), 1,
                          np.where(self._cross_below(SMI, SMIMA), -1, 0))
        return signals

    # ================= 价格反转指标补充 =================
    def RMI(self, N=7):
        """
        RMI指标 - 相对动量指数
        信号: RMI上穿30买入(1), 下穿70卖出(-1)
        """
        df = self.data
        close = df['close']
        momentum = close - close.shift(4)
        gain = momentum.where(momentum > 0, 0)
        loss = -momentum.where(momentum < 0, 0)
        avg_gain = self._sma(gain, N, 1)
        avg_loss = self._sma(loss, N, 1)
        RMI = 100 * avg_gain / (avg_gain + avg_loss + 1e-8)
        
        signals = np.where(self._cross_above(RMI, 30), 1,
                          np.where(self._cross_below(RMI, 70), -1, 0))
        return signals

    def SKDJ(self, N=60, M=5):
        """
        SKDJ指标 - 慢速随机指标
        信号: D<40且K上穿D买入(1), D>60且K下穿D卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        RSV = (close - low.rolling(N).min()) / (high.rolling(N).max() - low.rolling(N).min()) * 100
        MARSV = self._sma(RSV, 3, 1)
        K = self._sma(MARSV, 3, 1)
        D = self._ma(K, 3)
        
        buy_cond = (D < 40) & self._cross_above(K, D)
        sell_cond = (D > 60) & self._cross_below(K, D)
        signals = np.where(buy_cond, 1,
                          np.where(sell_cond, -1, 0))
        return signals

    def CCI(self, N=14):
        """
        CCI指标 - 顺势指标
        信号: CCI上穿-100买入(1), 下穿100卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TP = (high + low + close) / 3
        MA = self._ma(TP, N)
        MD = TP.rolling(N).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        CCI = (TP - MA) / (0.015 * MD + 1e-8)
        
        signals = np.where(self._cross_above(CCI, -100), 1,
                          np.where(self._cross_below(CCI, 100), -1, 0))
        return signals

    def ROC(self, N=100):
        """
        ROC指标 - 变化率指标
        信号: ROC上穿5%买入(1), 下穿-5%卖出(-1)
        """
        df = self.data
        close = df['close']
        ROC = (close - close.shift(N)) / close.shift(N) * 100
        
        signals = np.where(self._cross_above(ROC, 5), 1,
                          np.where(self._cross_below(ROC, -5), -1, 0))
        return signals

    def WR(self, N=14):
        """
        WR指标 - 威廉指标
        信号: WR上穿-80买入(1), 下穿-20卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        HIGH_N = high.rolling(N).max()
        LOW_N = low.rolling(N).min()
        WR = -100 * (HIGH_N - close) / (HIGH_N - LOW_N + 1e-8)
        
        signals = np.where(self._cross_above(WR, -80), 1,
                          np.where(self._cross_below(WR, -20), -1, 0))
        return signals

    def STC(self, N1=23, N2=50, N=40):
        """
        STC指标 - 随机趋势指标
        信号: STC上穿25买入(1), 下穿75卖出(-1)
        """
        df = self.data
        close = df['close']
        MACDX = self._ema(close, N1) - self._ema(close, N2)
        V1 = MACDX.rolling(N).min()
        V2 = MACDX.rolling(N).max() - V1
        FK = ((MACDX - V1) / V2 * 100).where(V2 > 0, 0)
        FD = self._sma(FK, N, 1)
        V3 = FD.rolling(N).min()
        V4 = FD.rolling(N).max() - V3
        SK = ((FD - V3) / V4 * 100).where(V4 > 0, 0)
        STC = self._sma(SK, N, 1)
        
        signals = np.where(self._cross_above(STC, 25), 1,
                          np.where(self._cross_below(STC, 75), -1, 0))
        return signals

    def RVI(self, N1=10, N2=20):
        """
        RVI指标 - 相对波动指数
        信号: RVI上穿30买入(1), 下穿70卖出(-1)
        """
        df = self.data
        close = df['close']
        STD = close.rolling(N1).std()
        USTD = STD.where(close > close.shift(1), 0).rolling(N2).sum()
        DSTD = STD.where(close < close.shift(1), 0).rolling(N2).sum()
        RVI = 100 * USTD / (USTD + DSTD + 1e-8)
        
        signals = np.where(self._cross_above(RVI, 30), 1,
                          np.where(self._cross_below(RVI, 70), -1, 0))
        return signals

    def UOS(self, M=7, N=14, O=28):
        """
        UOS指标 - 终极震荡指标
        信号: UOS上穿30买入(1), 下穿70卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TH = np.maximum(high, df['ref_close_1'])
        TL = np.minimum(low, df['ref_close_1'])
        TR = TH - TL
        XR = close - TL
        XRM = XR.rolling(M).sum() / TR.rolling(M).sum()
        XRN = XR.rolling(N).sum() / TR.rolling(N).sum()
        XRO = XR.rolling(O).sum() / TR.rolling(O).sum()
        UOS = 100 * (XRM*N*O + XRN*M*O + XRO*M*N) / (M*N + M*O + N*O)
        
        signals = np.where(self._cross_above(UOS, 30), 1,
                          np.where(self._cross_below(UOS, 70), -1, 0))
        return signals

    def RSIS(self, N=120, M=20):
        """
        RSIS指标 - RSI平滑指标
        信号: RSISMA上穿40买入(1), 下穿60卖出(-1)
        """
        df = self.data
        close = df['close']
        CLOSEUP = (close - close.shift(1)).where(close > close.shift(1), 0)
        CLOSEDOWN = abs(close - close.shift(1)).where(close < close.shift(1), 0)
        RSI = self._sma(CLOSEUP, N, 1) / self._sma(abs(close - close.shift(1)), N, 1) * 100
        RSIS = (RSI - RSI.rolling(N).min()) / (RSI.rolling(N).max() - RSI.rolling(N).min() + 1e-8) * 100
        RSISMA = self._ema(RSIS, M)
        
        signals = np.where(self._cross_above(RSISMA, 40), 1,
                          np.where(self._cross_below(RSISMA, 60), -1, 0))
        return signals

    # ================= 成交量指标补充 =================
    def MAAMT(self, N=40):
        """
        MAAMT指标 - 成交额移动平均
        信号: 成交额上穿MAAMT买入(1), 下穿MAAMT卖出(-1)
        """
        df = self.data
        # 假设成交额为价格*成交量
        amount = df['close'] * df['volume']
        MAAMT = self._ma(amount, N)
        
        signals = np.where(self._cross_above(amount, MAAMT), 1,
                          np.where(self._cross_below(amount, MAAMT), -1, 0))
        return signals

    def SROCVOL(self, N=20, M=10):
        """
        SROCVOL指标 - 平滑成交量变化率
        信号: SROCVOL上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        volume = df['volume']
        EMAP = self._ema(volume, N)
        SROCVOL = (EMAP - EMAP.shift(M)) / (EMAP.shift(M) + 1e-8)
        
        signals = np.where(self._cross_above(SROCVOL, 0), 1,
                          np.where(self._cross_below(SROCVOL, 0), -1, 0))
        return signals

    def PVO(self, N1=12, N2=26):
        """
        PVO指标 - 价格成交量震荡器
        信号: PVO上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        volume = df['volume']
        PVO = (self._ema(volume, N1) - self._ema(volume, N2)) / (self._ema(volume, N2) + 1e-8) * 100
        
        signals = np.where(self._cross_above(PVO, 0), 1,
                          np.where(self._cross_below(PVO, 0), -1, 0))
        return signals

    def BIASVOL(self, N1=6, N2=12, N3=24):
        """
        BIASVOL指标 - 成交量乖离率
        信号: 三个乖离率均>阈值买入(1), 均<负阈值卖出(-1)
        """
        df = self.data
        volume = df['volume']
        BIASVOL6 = (volume - self._ma(volume, N1)) / (self._ma(volume, N1) + 1e-8) * 100
        BIASVOL12 = (volume - self._ma(volume, N2)) / (self._ma(volume, N2) + 1e-8) * 100
        BIASVOL24 = (volume - self._ma(volume, N3)) / (self._ma(volume, N3) + 1e-8) * 100
        
        buy_cond = (BIASVOL6 > 5) & (BIASVOL12 > 7) & (BIASVOL24 > 11)
        sell_cond = (BIASVOL6 < -5) & (BIASVOL12 < -7) & (BIASVOL24 < -11)
        signals = np.where(buy_cond, 1,
                          np.where(sell_cond, -1, 0))
        return signals

    def MACDVOL(self, N1=20, N2=40, N3=10):
        """
        MACDVOL指标 - 成交量MACD
        信号: MACDVOL上穿SIGNAL买入(1), 下穿SIGNAL卖出(-1)
        """
        df = self.data
        volume = df['volume']
        MACDVOL = self._ema(volume, N1) - self._ema(volume, N2)
        SIGNAL = self._ma(MACDVOL, N3)
        
        signals = np.where(self._cross_above(MACDVOL, SIGNAL), 1,
                          np.where(self._cross_below(MACDVOL, SIGNAL), -1, 0))
        return signals

    def ROCVOL(self, N=80):
        """
        ROCVOL指标 - 成交量变化率
        信号: ROCVOL上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        volume = df['volume']
        ROCVOL = (volume - volume.shift(N)) / (volume.shift(N) + 1e-8) * 100
        
        signals = np.where(self._cross_above(ROCVOL, 0), 1,
                          np.where(self._cross_below(ROCVOL, 0), -1, 0))
        return signals

    # ================= 价量指标补充 =================
    def FI(self, N=13):
        """
        FI指标 - 强力指数
        信号: FIMA上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        FI = (close - close.shift(1)) * volume
        FIMA = self._ema(FI, N)
        
        signals = np.where(self._cross_above(FIMA, 0), 1,
                          np.where(self._cross_below(FIMA, 0), -1, 0))
        return signals

    def NVI(self, N=144):
        """
        NVI指标 - 负成交量指数
        信号: NVI上穿NVI_MA买入(1), 下穿NVI_MA卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        nvi_change = ((close - close.shift(1)) / close.shift(1)).where(volume < volume.shift(1), 0)
        NVI = (1 + nvi_change).cumprod() * 100
        NVI_MA = self._ma(NVI, N)
        
        signals = np.where(self._cross_above(NVI, NVI_MA), 1,
                          np.where(self._cross_below(NVI, NVI_MA), -1, 0))
        return signals

    def PVT(self, N1=10, N2=30):
        """
        PVT指标 - 价量趋势
        信号: PVT_MA1上穿PVT_MA2买入(1), 下穿PVT_MA2卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        PVT = ((close - close.shift(1)) / close.shift(1) * volume).cumsum()
        PVT_MA1 = self._ma(PVT, N1)
        PVT_MA2 = self._ma(PVT, N2)
        
        signals = np.where(self._cross_above(PVT_MA1, PVT_MA2), 1,
                          np.where(self._cross_below(PVT_MA1, PVT_MA2), -1, 0))
        return signals

    def RSIV(self, N=20):
        """
        RSIV指标 - 成交量RSI
        信号: RSIV上穿60买入(1), 下穿40卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        VOLUP = np.where(close > close.shift(1), volume, 0)
        VOLDOWN = np.where(close < close.shift(1), volume, 0)
        SUMUP = pd.Series(VOLUP).rolling(N).sum()
        SUMDOWN = pd.Series(VOLDOWN).rolling(N).sum()
        RSIV = 100 * SUMUP / (SUMUP + SUMDOWN + 1e-8)
        
        signals = np.where(self._cross_above(RSIV, 60), 1,
                          np.where(self._cross_below(RSIV, 40), -1, 0))
        return signals

    def AMV(self, N1=13, N2=34):
        """
        AMV指标 - 成交量加权移动平均
        信号: AMV1上穿AMV2买入(1), 下穿AMV2卖出(-1)
        """
        df = self.data
        open_, close, volume = df['open'], df['close'], df['volume']
        AMOV = volume * (open_ + close) / 2
        AMV1 = AMOV.rolling(N1).sum() / volume.rolling(N1).sum()
        AMV2 = AMOV.rolling(N2).sum() / volume.rolling(N2).sum()
        
        signals = np.where(self._cross_above(AMV1, AMV2), 1,
                          np.where(self._cross_below(AMV1, AMV2), -1, 0))
        return signals

    def VRAMT(self, N=40):
        """
        VRAMT指标 - 成交额比率
        信号: VRAMT上穿180买入(1), 下穿70卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        amount = close * volume
        AV = np.where(close > close.shift(1), amount, 0)
        BV = np.where(close < close.shift(1), amount, 0)
        CV = np.where(close == close.shift(1), amount, 0)
        AVS = pd.Series(AV).rolling(N).sum()
        BVS = pd.Series(BV).rolling(N).sum()
        CVS = pd.Series(CV).rolling(N).sum()
        VRAMT = (AVS + CVS/2) / (BVS + CVS/2 + 1e-8)
        
        signals = np.where(self._cross_above(VRAMT, 180), 1,
                          np.where(self._cross_below(VRAMT, 70), -1, 0))
        return signals

    def WVAD(self, N=20):
        """
        WVAD指标 - 威廉变异离散量
        信号: WVAD上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, close, open_, volume = df['high'], df['low'], df['close'], df['open'], df['volume']
        WVAD = (((close - open_) / (high - low + 1e-8)) * volume).rolling(N).sum()
        
        signals = np.where(self._cross_above(WVAD, 0), 1,
                          np.where(self._cross_below(WVAD, 0), -1, 0))
        return signals

    def PVI(self, N=40):
        """
        PVI指标 - 正成交量指数
        信号: PVI上穿PVI_MA买入(1), 下穿PVI_MA卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        pvi_change = ((close - close.shift(1)) / close.shift(1)).where(volume > volume.shift(1), 0)
        PVI = pvi_change.cumsum()
        PVI_MA = self._ma(PVI, N)
        
        signals = np.where(self._cross_above(PVI, PVI_MA), 1,
                          np.where(self._cross_below(PVI, PVI_MA), -1, 0))
        return signals

    def TMF(self, N=80):
        """
        TMF指标 - 时间货币流量
        信号: TMF上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        HIGH_TRUE = np.maximum(high, df['ref_close_1'])
        LOW_TRUE = np.minimum(low, df['ref_close_1'])
        TMF = (self._ema(volume * (2*close - HIGH_TRUE - LOW_TRUE) / (HIGH_TRUE - LOW_TRUE + 1e-8), N) / 
               (self._ema(volume, N) + 1e-8))
        
        signals = np.where(self._cross_above(TMF, 0), 1,
                          np.where(self._cross_below(TMF, 0), -1, 0))
        return signals

    def ADOSC(self, N1=3, N2=10):
        """
        ADOSC指标 - 累积/派发震荡器
        信号: ADOSC上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        CLV = ((close - low) - (high - close)) / (high - low + 1e-8)
        AD = (CLV * volume).cumsum()
        AD_EMA1 = self._ema(AD, N1)
        AD_EMA2 = self._ema(AD, N2)
        ADOSC = AD_EMA1 - AD_EMA2
        
        signals = np.where(self._cross_above(ADOSC, 0), 1,
                          np.where(self._cross_below(ADOSC, 0), -1, 0))
        return signals

    def VAO(self, N1=10, N2=30):
        """
        VAO指标 - 成交量与开盘价震荡器
        信号: VAO_MA1上穿VAO_MA2买入(1), 下穿VAO_MA2卖出(-1)
        """
        df = self.data
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        WEIGHTED_VOLUME = volume * (close - (high + low) / 2)
        VAO = WEIGHTED_VOLUME.cumsum()
        VAO_MA1 = self._ma(VAO, N1)
        VAO_MA2 = self._ma(VAO, N2)
        
        signals = np.where(self._cross_above(VAO_MA1, VAO_MA2), 1,
                          np.where(self._cross_below(VAO_MA1, VAO_MA2), -1, 0))
        return signals

    def VR(self, N=40):
        """
        VR指标 - 成交量比率
        信号: VR上穿250买入(1), 下穿300卖出(-1)
        """
        df = self.data
        close, volume = df['close'], df['volume']
        AV = np.where(close > close.shift(1), volume, 0)
        BV = np.where(close < close.shift(1), volume, 0)
        CV = np.where(close == close.shift(1), volume, 0)
        AVS = pd.Series(AV).rolling(N).sum()
        BVS = pd.Series(BV).rolling(N).sum()
        CVS = pd.Series(CV).rolling(N).sum()
        VR = (AVS + CVS/2) / (BVS + CVS/2 + 1e-8)
        
        signals = np.where(self._cross_above(VR, 250), 1,
                          np.where(self._cross_below(VR, 300), -1, 0))
        return signals

    def KO(self, N1=34, N2=55):
        """
        KO指标 - 克林格震荡器
        信号: KO上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        TYPICAL = (high + low + close) / 3
        VOLUME_SIGNED = volume.where(TYPICAL >= TYPICAL.shift(1), -volume)
        VOLUME_EMA1 = self._ema(VOLUME_SIGNED, N1)
        VOLUME_EMA2 = self._ema(VOLUME_SIGNED, N2)
        KO = VOLUME_EMA1 - VOLUME_EMA2
        
        signals = np.where(self._cross_above(KO, 0), 1,
                          np.where(self._cross_below(KO, 0), -1, 0))
        return signals

    def EMV(self, N=14, VOLUME_DIVISOR=1000000):
        """
        EMV指标 - 简易波动指标
        信号: EMV上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, volume = df['high'], df['low'], df['volume']
        MID_PT_MOVE = (high + low) / 2 - (df['ref_high_1'] + df['ref_low_1']) / 2
        BOX_RATIO = volume / VOLUME_DIVISOR / (high - low + 1e-8)
        EMV = MID_PT_MOVE / (BOX_RATIO + 1e-8)
        EMV_MA = self._ma(EMV, N)
        
        signals = np.where(self._cross_above(EMV_MA, 0), 1,
                          np.where(self._cross_below(EMV_MA, 0), -1, 0))
        return signals
    def TMA(self, N=20):
        """
        TMA指标 - 三重移动平均
        信号: 收盘价上穿TMA买入(1), 下穿TMA卖出(-1)
        """
        df = self.data
        close = df['close']
        CLOSE_MA = self._ma(close, N)
        TMA = self._ma(CLOSE_MA, N)
        
        signals = np.where(self._cross_above(close, TMA), 1,
                          np.where(self._cross_below(close, TMA), -1, 0))
        return signals

    def TYP(self, N1=10, N2=30):
        """
        TYP指标 - 典型价格均线交叉
        信号: TYPMA1上穿TYPMA2买入(1), 下穿TYPMA2卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TYP = (high + low + close) / 3
        TYPMA1 = self._ema(TYP, N1)
        TYPMA2 = self._ema(TYP, N2)
        
        signals = np.where(self._cross_above(TYPMA1, TYPMA2), 1,
                          np.where(self._cross_below(TYPMA1, TYPMA2), -1, 0))
        return signals

    def KDJD(self, N=20, M=60):
        """
        KDJD指标 - 双重随机指标
        信号: D上穿70买入(1), 下穿30卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        LOW_N = low.rolling(N).min()
        HIGH_N = high.rolling(N).max()
        Stochastics = (close - LOW_N) / (HIGH_N - LOW_N) * 100
        Stochastics_LOW = Stochastics.rolling(M).min()
        Stochastics_HIGH = Stochastics.rolling(M).max()
        Stochastics_DOUBLE = (Stochastics - Stochastics_LOW) / \
                            (Stochastics_HIGH - Stochastics_LOW) * 100
        K = self._sma(Stochastics_DOUBLE, 3, 1)
        D = self._sma(K, 3, 1)
        
        signals = np.where(self._cross_above(D, 70), 1,
                          np.where(self._cross_below(D, 30), -1, 0))
        return signals

    def MA(self, N1=5, N2=20):
        """
        MA指标 - 移动平均线交叉
        信号: 短期MA上穿长期MA买入(1), 下穿长期MA卖出(-1)
        """
        df = self.data
        close = df['close']
        MA_short = self._ma(close, N1)
        MA_long = self._ma(close, N2)
        
        signals = np.where(self._cross_above(MA_short, MA_long), 1,
                          np.where(self._cross_below(MA_short, MA_long), -1, 0))
        return signals

    def BBI(self):
        """
        BBI指标 - 多空指标
        信号: 收盘价上穿BBI买入(1), 下穿BBI卖出(-1)
        """
        df = self.data
        close = df['close']
        MA3 = self._ma(close, 3)
        MA6 = self._ma(close, 6)
        MA12 = self._ma(close, 12)
        MA24 = self._ma(close, 24)
        BBI = (MA3 + MA6 + MA12 + MA24) / 4
        
        signals = np.where(self._cross_above(close, BBI), 1,
                          np.where(self._cross_below(close, BBI), -1, 0))
        return signals

    def MADisplaced(self, N=20, M=10):
        """
        MADisplaced指标 - 位移移动平均
        信号: 收盘价上穿MADisplaced买入(1), 下穿卖出(-1)
        """
        df = self.data
        close = df['close']
        MA_CLOSE = self._ma(close, N)
        MADisplaced = MA_CLOSE.shift(M)
        
        signals = np.where(self._cross_above(close, MADisplaced), 1,
                          np.where(self._cross_below(close, MADisplaced), -1, 0))
        return signals

    def T3(self, N=20, VA=0.5):
        """
        T3指标 - 三重指数移动平均
        信号: 收盘价上穿T3买入(1), 下穿卖出(-1)
        """
        df = self.data
        close = df['close']
        EMA1 = self._ema(close, N)
        EMA2 = self._ema(EMA1, N)
        EMA3 = self._ema(EMA2, N)
        T1 = EMA1 * (1 + VA) - EMA2 * VA
        T2 = EMA2 * (1 + VA) - EMA3 * VA
        T3 = EMA3 * (1 + VA) - self._ema(EMA3, N) * VA
        
        signals = np.where(self._cross_above(close, T3), 1,
                          np.where(self._cross_below(close, T3), -1, 0))
        return signals

    def COPP(self, N1=10, N2=15, M=10):
        """
        COPP指标 - 变化率振荡器
        信号: COPP上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        RC = 100 * ((close - close.shift(N1))/close.shift(N1) + 
                   (close - close.shift(N2))/close.shift(N2))
        COPP = self._ema(RC, M)
        
        signals = np.where(self._cross_above(COPP, 0), 1,
                          np.where(self._cross_below(COPP, 0), -1, 0))
        return signals

    def ENV(self, N=25, PARAM=0.05):
        """
        ENV指标 - 包络线
        信号: 收盘价上穿UPPER买入(1), 下穿LOWER卖出(-1)
        """
        df = self.data
        close = df['close']
        MAC = self._ma(close, N)
        UPPER = MAC * (1 + PARAM)
        LOWER = MAC * (1 - PARAM)
        
        signals = np.where(self._cross_above(close, UPPER), 1,
                          np.where(self._cross_below(close, LOWER), -1, 0))
        return signals

    def RSIH(self, N1=40, N2=120):
        """
        RSIH指标 - RSI直方图
        信号: RSIH上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        CLOSEUP = (close - close.shift(1)).where(close > close.shift(1), 0)
        CLOSEDOWN = abs(close - close.shift(1)).where(close < close.shift(1), 0)
        RSI = self._ema(CLOSEUP, N1) / (self._ema(CLOSEUP, N1) + self._ema(CLOSEDOWN, N1)) * 100
        RSI_SIGNAL = self._ema(RSI, N2)
        RSIH = RSI - RSI_SIGNAL
        
        signals = np.where(self._cross_above(RSIH, 0), 1,
                          np.where(self._cross_below(RSIH, 0), -1, 0))
        return signals

    def HLMA(self, N1=20, N2=20):
        """
        HLMA指标 - 高低价移动平均
        信号: 收盘价上穿HMA买入(1), 下穿LMA卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        HMA = self._ma(high, N1)
        LMA = self._ma(low, N2)
        
        signals = np.where(self._cross_above(close, HMA), 1,
                          np.where(self._cross_below(close, LMA), -1, 0))
        return signals

    def LMA(self, N=20):
        """
        LMA指标 - 最低价移动平均
        信号: 最低价上穿LMA买入(1), 下穿LMA卖出(-1)
        """
        df = self.data
        low = df['low']
        LMA = self._ma(low, N)
        
        signals = np.where(self._cross_above(low, LMA), 1,
                          np.where(self._cross_below(low, LMA), -1, 0))
        return signals

    def VI(self, N=40):
        """
        VI指标 - 波动指数
        信号: VI+上穿VI-买入(1), 下穿VI-卖出(-1)
        """
        df = self.data
        high, low = df['high'], df['low']
        TR = np.maximum(abs(high - low), 
                       np.maximum(abs(low - df['ref_close_1']), 
                                 abs(high - df['ref_close_1'])))
        VMPOS = abs(high - df['ref_low_1'])
        VMNEG = abs(low - df['ref_high_1'])
        SUMPOS = pd.Series(VMPOS).rolling(N).sum()
        SUMNEG = pd.Series(VMNEG).rolling(N).sum()
        TRSUM = pd.Series(TR).rolling(N).sum()
        VI_POS = SUMPOS / TRSUM
        VI_NEG = SUMNEG / TRSUM
        
        signals = np.where(self._cross_above(VI_POS, VI_NEG), 1,
                          np.where(self._cross_below(VI_POS, VI_NEG), -1, 0))
        return signals

    def RWI(self, N=14):
        """
        RWI指标 - 随机漫步指数
        信号: RWIH>1买入(1), RWIL>1卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TR = np.maximum(abs(high - low), 
                       np.maximum(abs(high - df['ref_close_1']), 
                                 abs(low - df['ref_close_1'])))
        ATR = self._ma(TR, N)
        RWIH = (high - df['ref_low_1']) / (ATR * np.sqrt(N))
        RWIL = (df['ref_high_1'] - low) / (ATR * np.sqrt(N))
        
        signals = np.where(RWIH > 1, 1,
                          np.where(RWIL > 1, -1, 0))
        return signals

    def ARBR(self, N=40):
        """
        ARBR指标 - 人气意愿指标
        信号: AR上穿50买入(1), 下穿200卖出(-1)
        """
        df = self.data
        high, low, open_ = df['high'], df['low'], df['open']
        AR = pd.Series(high - open_).rolling(N).sum() / \
             pd.Series(open_ - low).rolling(N).sum() * 100
        BR = pd.Series(high - df['ref_close_1']).rolling(N).sum() / \
             pd.Series(df['ref_close_1'] - low).rolling(N).sum() * 100
        
        signals = np.where(self._cross_above(AR, 50), 1,
                          np.where(self._cross_below(AR, 200), -1, 0))
        return signals

    def DO(self, N=14, M=7):
        """
        DO指标 - 双重平滑RSI
        信号: DO上穿DO_MA买入(1), 下穿DO_MA卖出(-1)
        """
        df = self.data
        close = df['close']
        CLOSEUP = (close - close.shift(1)).where(close > close.shift(1),  0)
        CLOSEDOWN = (abs(close - close.shift(1))).where(close < close.shift(1),  0)
        RSI = self._ema(CLOSEUP, N) / (self._ema(CLOSEUP, N) + self._ema(CLOSEDOWN, N)) * 100
        DO = self._ema(self._ema(RSI, N), M)
        DO_MA = self._ma(DO, M)
        
        signals = np.where(self._cross_above(DO, DO_MA), 1,
                          np.where(self._cross_below(DO, DO_MA), -1, 0))
        return signals

    def SI(self, N=20):
        """
        SI指标 - 摆动指数
        信号: SI上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, close, open_ = df['high'], df['low'], df['close'], df['open']
        A = abs(high - df['ref_close_1'])
        B = abs(low - df['ref_close_1'])
        C = abs(high - df['ref_low_1'])
        D = abs(df['ref_close_1'] - df['ref_open_1'])
        K = np.maximum(A, B)
        M = (high - low).rolling(N).max()
        R1 = A + 0.5*B + 0.25*D
        R2 = B + 0.5*A + 0.25*D
        R3 = C + 0.25*D
        R4 = np.where((A >= B) & (A >= C), R1, R2)
        R = np.where((C >= A) & (C >= B), R3, R4)
        SI = 50 * ((close - df['ref_close_1']) + (df['ref_close_1'] - df['ref_open_1']) + 
                  0.5*(close - open_)) / (R + 1e-8) * K / (M + 1e-8)
        
        signals = np.where(self._cross_above(SI, 0), 1,
                          np.where(self._cross_below(SI, 0), -1, 0))
        return signals

    def DBCD(self, N=5, M=16, T=17):
        """
        DBCD指标 - 异同离差乖离率
        信号: DBCD上穿5%买入(1), 下穿-5%卖出(-1)
        """
        df = self.data
        close = df['close']
        BIAS = (close - self._ma(close, N)) / self._ma(close, N) * 100
        BIAS_DIF = BIAS - BIAS.shift(M)
        DBCD = self._ema(BIAS_DIF, T)
        
        signals = np.where(self._cross_above(DBCD, 5), 1,
                          np.where(self._cross_below(DBCD, -5), -1, 0))
        return signals

    def DZRSI(self, N=40, M=3, PARAM=1.5):
        """
        DZRSI指标 - 动态RSI
        信号: RSI_MA上穿RSI_UPPER买入(1), 下穿RSI_LOWER卖出(-1)
        """
        df = self.data
        close = df['close']
        CLOSEUP = (close - close.shift(1)).where(close > close.shift(1),  0)
        CLOSEDOWN = abs(close - close.shift(1)).where(close < close.shift(1),  0)
        RSI = self._ema(CLOSEUP, N) / (self._ema(CLOSEUP, N) + self._ema(CLOSEDOWN, N)) * 100
        RSI_MIDDLE = self._ma(RSI, N)
        RSI_UPPER = RSI_MIDDLE + PARAM * RSI.rolling(N).std()
        RSI_LOWER = RSI_MIDDLE - PARAM * RSI.rolling(N).std()
        RSI_MA = self._ma(RSI, M)
        
        signals = np.where(self._cross_above(RSI_MA, RSI_UPPER), 1,
                          np.where(self._cross_below(RSI_MA, RSI_LOWER), -1, 0))
        return signals

    def CV(self, N=10):
        """
        CV指标 - 波动系数
        信号: CV绝对值下穿30买入(1), 上穿70卖出(-1)
        """
        df = self.data
        high, low = df['high'], df['low']
        H_L_EMA = self._ema(high - low, N)
        CV = (H_L_EMA - H_L_EMA.shift(N)) / (H_L_EMA.shift(N) + 1e-8) * 100
        CV_ABS = abs(CV)
        
        signals = np.where(self._cross_below(CV_ABS, 30), 1,
                          np.where(self._cross_above(CV_ABS, 70), -1, 0))
        return signals

    def MTM(self, N=60):
        """
        MTM指标 - 动量指标
        信号: MTM上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        MTM = close - close.shift(N)
        
        signals = np.where(self._cross_above(MTM, 0), 1,
                          np.where(self._cross_below(MTM, 0), -1, 0))
        return signals

    def CR(self, N=20):
        """
        CR指标 - 中间意愿指标
        信号: CR上穿200买入(1), 下穿50卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TYP = (high + low + close) / 3
        H = np.maximum(high - TYP.shift(1), 0)
        L = np.maximum(TYP.shift(1) - low, 0)
        CR = pd.Series(H).rolling(N).sum() / (pd.Series(L).rolling(N).sum() + 1e-8) * 100
        
        signals = np.where(self._cross_above(CR, 200), 1,
                          np.where(self._cross_below(CR, 50), -1, 0))
        return signals

    def BOP(self, N=20):
        """
        BOP指标 - 均势指标
        信号: BOP上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low, open_, close = df['high'], df['low'], df['open'], df['close']
        BOP = self._ma((close - open_) / (high - low + 1e-8), N)
        
        signals = np.where(self._cross_above(BOP, 0), 1,
                          np.where(self._cross_below(BOP, 0), -1, 0))
        return signals

    def HULLMA(self, N1=20, N2=80):
        """
        HULLMA指标 - 赫尔移动平均
        信号: 短期HULLMA上穿长期HULLMA买入(1), 下穿卖出(-1)
        """
        df = self.data
        close = df['close']
        X = 2 * self._ema(close, N1//2) - self._ema(close, N1)
        HULLMA_short = self._ema(X, int(np.sqrt(N1)))
        X2 = 2 * self._ema(close, N2//2) - self._ema(close, N2)
        HULLMA_long = self._ema(X2, int(np.sqrt(N2)))
        
        signals = np.where(self._cross_above(HULLMA_short, HULLMA_long), 1,
                          np.where(self._cross_below(HULLMA_short, HULLMA_long), -1, 0))
        return signals

    def ASI(self, N=20):
        """
        ASI指标 - 累积摆动指数
        信号: ASI上穿ASIMA买入(1), 下穿ASIMA卖出(-1)
        """
        df = self.data
        high, low, close, open_ = df['high'], df['low'], df['close'], df['open']
        ref_close_1 = df['ref_close_1']
        ref_open_1 = df['ref_open_1']
        
        A = abs(high - ref_close_1)
        B = abs(low - ref_close_1)
        C = abs(high - df['ref_low_1'])
        D = abs(ref_close_1 - ref_open_1)
        
        K = np.maximum(A, B)
        M = (high - low).rolling(N).max()
        
        R1 = A + 0.5*B + 0.25*D
        R2 = B + 0.5*A + 0.25*D
        R3 = C + 0.25*D
        R4 = np.where((A >= B) & (A >= C), R1, R2)
        R = np.where((C >= A) & (C >= B), R3, R4)
        
        SI = 50 * ((close - ref_close_1) + (ref_close_1 - ref_open_1) + 
                 0.5*(close - open_)) / (R + 1e-8) * K / (M + 1e-8)
        ASI = SI.cumsum()
        ASIMA = self._ma(ASI, N)
        
        signals = np.where(self._cross_above(ASI, ASIMA), 1,
                          np.where(self._cross_below(ASI, ASIMA), -1, 0))
        return signals

    def Arron(self, N=20):
        """
        Arron指标 - 衡量最高/最低价出现位置
        信号: ArronOs上穿20买入(1), 下穿-20卖出(-1)
        """
        df = self.data
        high, low = df['high'], df['low']
        
        high_len = high.rolling(N).apply(lambda x: N - np.argmax(x) - 1)
        low_len = low.rolling(N).apply(lambda x: N - np.argmin(x) - 1)
        
        ArronUp = (N - high_len) / N * 100
        ArronDown = (N - low_len) / N * 100
        ArronOs = ArronUp - ArronDown
        
        signals = np.where(self._cross_above(ArronOs, 20), 1,
                          np.where(self._cross_below(ArronOs, -20), -1, 0))
        return signals

    def KC(self, N=14):
        """
        KC指标 - 肯通纳通道
        信号: 收盘价上穿UPPER买入(1), 下穿LOWER卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TR = np.maximum(abs(high - low), 
                       np.maximum(abs(high - df['ref_close_1']), 
                                 abs(df['ref_close_1'] - low)))
        ATR = self._ma(TR, N)
        Middle = self._ema(close, 20)
        UPPER = Middle + 2 * ATR
        LOWER = Middle - 2 * ATR
        
        signals = np.where(self._cross_above(close, UPPER), 1,
                          np.where(self._cross_below(close, LOWER), -1, 0))
        return signals

    def VMA(self, N=20):
        """
        VMA指标 - 成交量加权移动平均
        信号: PRICE上穿VMA买入(1), 下穿VMA卖出(-1)
        """
        df = self.data
        high, low, open_, close = df['high'], df['low'], df['open'], df['close']
        PRICE = (high + low + open_ + close) / 4
        VMA = self._ma(PRICE, N)
        
        signals = np.where(self._cross_above(PRICE, VMA), 1,
                          np.where(self._cross_below(PRICE, VMA), -1, 0))
        return signals

    def BIAS(self, N1=6, N2=12, N3=24):
        """
        BIAS指标 - 乖离率
        信号: BIAS6>5且BIAS12>7且BIAS24>11买入(1), 
             BIAS6<-5且BIAS12<-7且BIAS24<-11卖出(-1)
        """
        df = self.data
        close = df['close']
        BIAS6 = (close - self._ma(close, N1)) / self._ma(close, N1) * 100
        BIAS12 = (close - self._ma(close, N2)) / self._ma(close, N2) * 100
        BIAS24 = (close - self._ma(close, N3)) / self._ma(close, N3) * 100
        
        buy_cond = (BIAS6 > 5) & (BIAS12 > 7) & (BIAS24 > 11)
        sell_cond = (BIAS6 < -5) & (BIAS12 < -7) & (BIAS24 < -11)
        signals = np.where(buy_cond, 1,
                          np.where(sell_cond, -1, 0))
        return signals

    def WMA(self, N=20):
        """
        WMA指标 - 加权移动平均
        信号: 收盘价上穿WMA买入(1), 下穿WMA卖出(-1)
        """
        df = self.data
        close = df['close']
        weights = np.arange(1, N+1)
        WMA = close.rolling(N).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True)
        
        signals = np.where(self._cross_above(close, WMA), 1,
                          np.where(self._cross_below(close, WMA), -1, 0))
        return signals

    def DDI(self, N=40):
        """
        DDI指标 - 方向偏差指标
        信号: DDI上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        high, low = df['high'], df['low']
        HL = high + low
        high_abs = abs(high - df['ref_high_1'])
        low_abs = abs(low - df['ref_low_1'])
        
        DMZ = np.where(HL > HL.shift(1), np.maximum(high_abs, low_abs), 0)
        DMF = np.where(HL < HL.shift(1), np.maximum(high_abs, low_abs), 0)
        DIZ = pd.Series(DMZ).rolling(N).sum() / (pd.Series(DMZ).rolling(N).sum() + 
                                               pd.Series(DMF).rolling(N).sum() + 1e-8)
        DIF = pd.Series(DMF).rolling(N).sum() / (pd.Series(DMZ).rolling(N).sum() + 
                                               pd.Series(DMF).rolling(N).sum() + 1e-8)
        DDI = DIZ - DIF
        
        signals = np.where(self._cross_above(DDI, 0), 1,
                          np.where(self._cross_below(DDI, 0), -1, 0))
        return signals

    def HMA(self, N=20):
        """
        HMA指标 - 最高价移动平均
        信号: 最高价上穿HMA买入(1), 下穿HMA卖出(-1)
        """
        df = self.data
        high = df['high']
        HMA = self._ma(high, N)
        
        signals = np.where(self._cross_above(high, HMA), 1,
                          np.where(self._cross_below(high, HMA), -1, 0))
        return signals

    def SROC(self, N=13, M=21):
        """
        SROC指标 - 平滑变化率指标
        信号: SROC上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close = df['close']
        EMAP = self._ema(close, N)
        SROC = (EMAP - EMAP.shift(M)) / EMAP.shift(M)
        
        signals = np.where(self._cross_above(SROC, 0), 1,
                          np.where(self._cross_below(SROC, 0), -1, 0))
        return signals

    def EXPMA(self, N1=12, N2=50):
        """
        EXPMA指标 - 指数移动平均
        信号: 短期EXPMA上穿长期EXPMA买入(1), 下穿卖出(-1)
        """
        df = self.data
        close = df['close']
        EMA1 = self._ema(close, N1)
        EMA2 = self._ema(close, N2)
        
        signals = np.where(self._cross_above(EMA1, EMA2), 1,
                          np.where(self._cross_below(EMA1, EMA2), -1, 0))
        return signals

    def DC(self, N=20):
        """
        DC指标 - 唐奇安通道
        信号: 收盘价上穿UPPER买入(1), 下穿LOWER卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        UPPER = high.rolling(N).max()
        LOWER = low.rolling(N).min()
        MIDDLE = (UPPER + LOWER) / 2
        
        signals = np.where(self._cross_above(close, MIDDLE), 1,
                          np.where(self._cross_below(close, MIDDLE), -1, 0))
        return signals

    def VIDYA(self, N=10):
        """
        VIDYA指标 - 可变指数动态移动平均
        信号: 收盘价上穿VIDYA买入(1), 下穿VIDYA卖出(-1)
        """
        df = self.data
        close = df['close']
        VI = abs(close - close.shift(N)) / (abs(close.diff()).rolling(N).sum() + 1e-8)
        VIDYA = VI * close + (1 - VI) * close.shift(1)
        
        signals = np.where(self._cross_above(close, VIDYA), 1,
                          np.where(self._cross_below(close, VIDYA), -1, 0))
        return signals

    def Qstick(self, N=20):
        """
        Qstick指标 - 衡量收盘价与开盘价差异
        信号: Qstick上穿0买入(1), 下穿0卖出(-1)
        """
        df = self.data
        close, open_ = df['close'], df['open']
        Qstick = self._ma(close - open_, N)
        
        signals = np.where(self._cross_above(Qstick, 0), 1,
                          np.where(self._cross_below(Qstick, 0), -1, 0))
        return signals

    def FB(self, N=20):
        """
        FB指标 - 斐波那契布林带
        信号: 收盘价上穿UPPER2买入(1), 下穿LOWER2卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        TR = np.maximum(high - low, 
                       np.maximum(abs(high - df['ref_close_1']), 
                                 abs(low - df['ref_close_1'])))
        ATR = self._ma(TR, N)
        MIDDLE = self._ma(close, N)
        UPPER2 = MIDDLE + 2.618 * ATR
        LOWER2 = MIDDLE - 2.618 * ATR
        
        signals = np.where(self._cross_above(close, UPPER2), 1,
                          np.where(self._cross_below(close, LOWER2), -1, 0))
        return signals

    def DEMA(self, N=60):
        """
        DEMA指标 - 双重指数移动平均
        信号: 收盘价上穿DEMA买入(1), 下穿DEMA卖出(-1)
        """
        df = self.data
        close = df['close']
        EMA = self._ema(close, N)
        DEMA = 2 * EMA - self._ema(EMA, N)
        
        signals = np.where(self._cross_above(close, DEMA), 1,
                          np.where(self._cross_below(close, DEMA), -1, 0))
        return signals

    def APZ(self, N=10, M=20, PARAM=2):
        """
        APZ指标 - 自适应价格区间
        信号: 收盘价上穿UPPER买入(1), 下穿LOWER卖出(-1)
        """
        df = self.data
        high, low, close = df['high'], df['low'], df['close']
        VOL = self._ema(self._ema(high - low, N), N)
        UPPER = self._ema(self._ema(close, M), M) + PARAM * VOL
        LOWER = self._ema(self._ema(close, M), M) - PARAM * VOL
        
        signals = np.where(self._cross_above(close, UPPER), 1,
                          np.where(self._cross_below(close, LOWER), -1, 0))
        return signals

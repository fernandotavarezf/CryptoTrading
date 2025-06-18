# ML-based Trading Bot adapted from Hybrid MA-EMA Bot V2
# ------------------------------------------------------
# This script keeps the same exchange and risk management logic as the
# `Hybrid MA-EMA Bot V2` but replaces the indicator based strategy with a
# machine learning model.  The model was trained using TALib derived
# features and is loaded from `model_final_5_1.h5`.
#
# Each section below mirrors the original bot (exchange connector, strategy,
# position management and trading loop) with additional comments to make it
# clearer how the ML predictions are used to decide when to buy or sell.

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from technical_analysis_lib import TecnicalAnalysis, BUY, HOLD, SELL
from compute_indicators_labels_lib import get_dataset
from config import RUN as RUN_CONFIG


# --------------------------------------------------
# Persistent StandardScaler used for feature scaling
# --------------------------------------------------
SCALER_PATH = Path("C:\\Users\\Fernando Fondeur\\OneDrive\\Desktop\\CryptoTrading_vf\\CryptoTrading\\artifacts").with_name("std_scaler.pkl")
if SCALER_PATH.exists():
    scaler = joblib.load(SCALER_PATH)
else:
    data = get_dataset(RUN_CONFIG)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    drop_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Asset_name", "label"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    scaler = StandardScaler().fit(data)
    SCALER_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)


class TradingLogger:
    """Simple stdout/file logger used across the project."""

    def __init__(self, name: str, log_level: int = 20, log_file: Optional[str] = None):
        """Create the logger.

        Parameters
        ----------
        name : str
            Name of the logger instance.
        log_level : int, optional
            Logging level (INFO=20 by default).
        log_file : Optional[str]
            If provided, logs will also be written into ``logs/<log_file>``.
        """
        import logging

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        if log_file:
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(f"logs/{log_file}")
            fh.setLevel(log_level)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        self.logger.addHandler(ch)

    def info(self, msg: str, extra: Optional[dict] = None) -> None:
        """Log an info level message."""
        if extra:
            self.logger.info(f"{msg} - {extra}")
        else:
            self.logger.info(msg)

    def error(self, msg: str, err: Any = None) -> None:
        """Log an error."""
        if err:
            self.logger.error(f"{msg}: {err}")
        else:
            self.logger.error(msg)

    def debug(self, msg: str, extra: Optional[dict] = None) -> None:
        """Log a debug message."""
        if extra:
            self.logger.debug(f"{msg} - {extra}")
        else:
            self.logger.debug(msg)


@dataclass
class OrderRequest:
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reduce_only: bool = False


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


class BybitConnector:
    """Minimal async connector for Bybit using ccxt."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False, logger: Optional[TradingLogger] = None):
        """Create an async Bybit connector using ccxt."""
        self.exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "linear", "adjustForTimeDifference": True, "recvWindow": 20000},
            "timeout": 30000,
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
        self.logger = logger or TradingLogger("BybitConnector")

    async def fetch_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Return OHLCV data for ``symbol`` as a DataFrame."""
        candles = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df

    async def create_order(self, order: OrderRequest) -> Dict:
        """Submit an order to the exchange."""
        params = {}
        if order.stop_loss:
            params["stopLoss"] = order.stop_loss
        if order.take_profit:
            params["takeProfit"] = order.take_profit
        if order.reduce_only:
            params["reduceOnly"] = True

        return await self.exchange.create_order(
            symbol=order.symbol,
            type=order.order_type,
            side=order.side,
            amount=order.quantity,
            price=order.price,
            params=params,
        )

    async def cancel_orders(self, symbol: str) -> bool:
        await self.exchange.cancel_all_orders(symbol)
        return True

    async def get_positions(self, symbol: str) -> List[Position]:
        """Fetch open positions for ``symbol`` and map them into ``Position`` objects."""
        positions = await self.exchange.fetch_positions([symbol])
        result = []
        for p in positions:
            if float(p.get("contracts", 0)) > 0:
                result.append(
                    Position(
                        symbol=p["symbol"],
                        side=p["side"],
                        size=float(p["contracts"]),
                        entry_price=float(p["entryPrice"]),
                        unrealized_pnl=float(p["unrealizedPnl"]),
                        leverage=float(p["leverage"]),
                    )
                )
        return result

    async def get_balance(self) -> Dict:
        """Return account balance dictionary."""
        bal = await self.exchange.fetch_balance()
        return {"total": bal["total"], "used": bal["used"], "free": bal["free"]}

    async def close(self) -> None:
        """Close the underlying ccxt connection."""
        await self.exchange.close()

@dataclass
class StrategySignal:
    """Container for a trading signal generated by the strategy."""
    timestamp: datetime
    symbol: str
    signal_type: str
    entry_price: float
    order_type: str = "market"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict] = None
    strategy_name: str = "MLStrategy"
    id: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Generate a unique ID so we can track signals across the bot."""
        self.id = SignalIDGenerator.generate_id(
            direction=self.signal_type,
            ticker=self.symbol,
            order_type=self.order_type,
            timestamp=self.timestamp,
            strategy_class_name=self.strategy_name,
        )


class SignalIDGenerator:
    """Utility class used to build reproducible signal identifiers."""
    @staticmethod
    def generate_strategy_code(name: str) -> str:
        """Return a short code derived from a CamelCase strategy name."""
        import re

        words = re.findall("[A-Z][^A-Z]*", name)
        code = "".join(w[0] for w in words)
        if len(code) == 1 and len(words[0]) > 1:
            code += words[0][1].upper()
        return code[:3]

    @staticmethod
    def generate_id(direction: str, ticker: str, order_type: str, timestamp: datetime, strategy_class_name: str) -> str:
        """Compose a human readable identifier for a signal."""
        dir_char = "B" if direction.lower() == "buy" else "S"
        type_char = "M" if order_type.lower() == "market" else "L"
        time_str = timestamp.strftime("%y%m%d%H%M")
        strategy_code = SignalIDGenerator.generate_strategy_code(strategy_class_name)
        return f"{dir_char}_{ticker}_{type_char}_{time_str}_{strategy_code}"


class MLStrategy:
    """Strategy that uses a pretrained ML model for predictions."""

    def __init__(self, model_path: str) -> None:
        """Load the Keras model and use the persistent scaler."""
        self.model = tf.keras.models.load_model(model_path)
        # Reuse the scaler computed at import time
        self.scaler = scaler
        self.logger: Optional[TradingLogger] = None

    def set_logger(self, logger: TradingLogger) -> None:
        """Attach logger from the bot instance."""
        self.logger = logger

    async def calculate_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate the technical indicator features expected by the model."""
        df = df.copy()
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "timestamp": "Date",
            }
        )
        df["Asset_name"] = symbol
        df = TecnicalAnalysis.compute_oscillators(df)
        df = TecnicalAnalysis.find_patterns(df)
        df = TecnicalAnalysis.add_timely_data(df)
        df = TecnicalAnalysis.add_timely_data(df)
        # Replace any infinite results from TA-Lib with NaN then drop
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna().reset_index(drop=True)
        return df

    async def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """Use the ML model to predict the next action.

        The model outputs probabilities for each class (Buy/Hold/Sell). ``np.argmax``
        selects the class with the highest probability.  Labels in the dataset are
        encoded as ``-1`` for buy, ``0`` for hold and ``1`` for sell; therefore the
        argmax result is shifted by ``-1`` to obtain that mapping.
        """
        df_feat = await self.calculate_features(df, symbol)
        if df_feat.empty:
            return None
        # Remove columns not used for training before scaling.
        # "symbol" must also be dropped to avoid strings during scaling.
        X = df_feat.drop(
            columns=[
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Asset_name",
                "symbol",
            ],
            errors="ignore",
        )
        # Guard against any unexpected inf or NaN values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X.isnull().any().any():
            if self.logger:
                self.logger.debug("Skipping signal due to NaN values in features")
            return None
        # Features are scaled using the precomputed scaler
        X_scaled = self.scaler.transform(X)
        # Show the raw and scaled feature values for the last row so it's
        # clear exactly what the model is receiving.
        raw_dict = X.iloc[-1].to_dict()
        scaled_dict = {c: float(v) for c, v in zip(X.columns, X_scaled[-1])}
        #print(f"[{symbol}] Raw features at {df_feat['Date'].iloc[-1]}: {raw_dict}")
        #print(f"[{symbol}] Scaled features: {scaled_dict}")

        # Predict using the last row of features
        pred = self.model.predict(X_scaled[-1].reshape(1, -1), verbose=0)
        # Display model output probabilities and the resulting action
        probs = {lbl: float(p) for lbl, p in zip(['BUY', 'HOLD', 'SELL'], pred[0])}
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} [{symbol}] Model probabilities: {probs}")
        # Convert network output to label in [-1, 0, 1]
        label = int(np.argmax(pred, axis=1)[0]) - 1
        action_text = {BUY: 'BUY', HOLD: 'HOLD', SELL: 'SELL'}[label]
        print(f"{ts} [{symbol}] Predicted action: {action_text}")
        price = df_feat["Close"].iloc[-1]
        if label == BUY:
            stype = "buy"
        elif label == SELL:
            stype = "sell"
        else:
            return None
        sig = StrategySignal(
            timestamp=pd.to_datetime(df_feat["Date"].iloc[-1]),
            symbol=symbol,
            signal_type=stype,
            entry_price=price,
            metadata={"ml_label": int(label)},
            strategy_name=self.__class__.__name__,
        )
        return sig

    async def calculate_position_size(self, signal: StrategySignal, equity: float, risk_percentage: float) -> float:
        """Basic position sizing using account equity and leverage."""
        risk_amount = equity * risk_percentage
        price = signal.entry_price
        leverage = 10
        if price <= 0:
            return 0.0
        return (risk_amount / price) * leverage


@dataclass
class ProfitMilestone:
    """Configuration of a PnL milestone used by ``PNLMilestoneTracker``."""
    target_percentage: Decimal
    trailing_stop_percentage: Decimal
    level: int
    triggered: bool = False
    highest_pnl: Decimal = Decimal("0")


class PNLMilestoneTracker:
    """Tracks profit milestones and evaluates when to close a position."""
    def __init__(self, initial_milestone: float = 25.0, trailing_stop: float = 5.0, max_milestones: int = 10):
        """Create a tracker with a series of incremental profit targets."""
        if initial_milestone <= 0 or trailing_stop <= 0:
            raise ValueError("Invalid milestone parameters")
        self.milestone_percentage = Decimal(str(initial_milestone))
        self.trailing_stop = Decimal(str(trailing_stop))
        self.max_milestones = max_milestones
        self.milestones: Dict[int, ProfitMilestone] = {}
        self._init_milestones()

    def _init_milestones(self) -> None:
        """Populate ``self.milestones`` with the configured levels."""
        for lvl in range(1, self.max_milestones + 1):
            target = self.milestone_percentage * lvl
            self.milestones[lvl] = ProfitMilestone(target, self.trailing_stop, lvl)

    def _get_current_milestone(self, pnl: Decimal) -> Optional[ProfitMilestone]:
        mstone = None
        for lvl in range(1, self.max_milestones + 1):
            ms = self.milestones[lvl]
            if pnl >= ms.target_percentage:
                if not ms.triggered:
                    ms.triggered = True
                    ms.highest_pnl = pnl
                mstone = ms
            elif ms.triggered:
                mstone = ms
            else:
                break
        return mstone

    def evaluate_position(self, current_pnl_percentage: float) -> Tuple[bool, Optional[str], Dict]:
        """Return True if drawdown exceeds the trailing stop for the current milestone."""
        pnl = Decimal(str(current_pnl_percentage))
        milestone = self._get_current_milestone(pnl)
        if not milestone:
            return False, None, {"status": "no_milestone", "current_pnl": float(pnl)}
        if pnl > milestone.highest_pnl:
            milestone.highest_pnl = pnl
        drawdown = milestone.highest_pnl - pnl
        state = {
            "current_pnl": float(pnl),
            "milestone_level": milestone.level,
            "milestone_target": float(milestone.target_percentage),
            "highest_pnl": float(milestone.highest_pnl),
            "drawdown": float(drawdown),
            "trailing_stop": float(milestone.trailing_stop_percentage),
        }
        if drawdown > milestone.trailing_stop_percentage:
            reason = f"Drawdown {float(drawdown)}% exceeded trailing stop {float(milestone.trailing_stop_percentage)}%"
            return True, reason, state
        return False, None, state

    def reset(self) -> None:
        """Clear milestone state when all positions are closed."""
        self._init_milestones()


class PositionManager:
    """Monitors open positions and applies exit rules."""
    def __init__(self, logger: TradingLogger, timeframe: str = "4h") -> None:
        """Initialise with sensible default hold times and PnL tracker."""
        self.logger = logger
        self.timeframe = timeframe
        self.pnl_tracker = PNLMilestoneTracker()
        self.max_hold_time = timedelta(hours=24)
        self.min_hold_time = timedelta(hours=1)

    async def update_monitoring(self, info: Dict, current_pnl: float) -> Dict:
        """Update position monitoring info with the latest PnL and time held."""
        now = datetime.now()
        info["time_held"] = now - info["entry_time"]
        info["last_update_time"] = now
        info["current_pnl"] = current_pnl
        if current_pnl > info.get("highest_pnl", float("-inf")):
            info["highest_pnl"] = current_pnl
            info["current_drawdown"] = 0
        else:
            drawdown = info.get("highest_pnl", 0) - current_pnl
            info["current_drawdown"] = max(0, drawdown)
        return info

    async def should_close_position(self, info: Dict) -> Tuple[bool, Optional[str]]:
        """Evaluate multiple criteria to determine if a position should be closed."""
        current_time = datetime.now()
        time_held = current_time - info["entry_time"]
        price_diff = info["current_price"] - info["entry_price"] if info["side"].lower() in ["buy", "long"] else info["entry_price"] - info["current_price"]
        raw_pnl = (price_diff / info["entry_price"]) * 100
        current_pnl = raw_pnl * info.get("leverage", 1)
        if current_pnl <= -25:
            return True, "Emergency stop loss"
        if time_held < self.min_hold_time:
            if current_pnl <= -20:
                return True, "Significant early loss"
            return False, None
        if time_held > self.max_hold_time:
            return True, "Max hold time exceeded"
        peak_pnl = info["highest_pnl"]
        drawdown = peak_pnl - current_pnl
        trailing_stop = 5.0
        if peak_pnl >= 25.0 and drawdown > trailing_stop:
            return True, f"Trailing stop hit: drawdown {drawdown:.2f}%"
        should_close, reason, _ = self.pnl_tracker.evaluate_position(current_pnl)
        if should_close:
            return True, reason
        return False, None


class OrderManager:
    """Utility helper for order size limits."""
    @staticmethod
    def get_min_size(ticker: str) -> float:
        """Return the minimum order size for the specified ticker."""
        min_sizes = {
            "BTCUSDT": 0.001,
            "SOLUSDT": 0.1,
            "DOGEUSDT": 100,
            "WIFUSDT": 1,
            "XRPUSDT": 1,
            "PNUTUSDT": 1,
            "TAOUSDT": 1,
            "1000PEPEUSDT": 100,
            "CHILLGUYUSDT": 1,
            "MOODENGUSDT": 1,
            "KOMAUSDT": 1,
            "RIFSOLUSDT": 1,
            "MOVEUSDT": 1,
            "GIGAUSDT": 1,
            "SUIUSDT": 1
        }
        return min_sizes.get(ticker, 1.0)


# ---------------------------------------------------------------------------
# Shared risk sizing helper and constants
# ---------------------------------------------------------------------------
COMMISSION_PCT = 0.0006
STOP_LOSS_PCT = 0.025  # 2.5 %
RISK_PCT = 0.25        # 25 % of equity
LEVERAGE = 10

def calc_size(equity: float, price: float, ticker: str,
              risk_pct: float = RISK_PCT,
              stop_loss_pct: float = STOP_LOSS_PCT,
              leverage: int = LEVERAGE) -> float:
    """Return order quantity using a fraction of available equity.

    The position size is derived from the margin allocated to the trade
    (``equity * risk_pct``) multiplied by the desired leverage and divided by
    the entry price.  This ensures we never request more margin than available
    while still applying the configured leverage.
    """
    qty = (equity * risk_pct * leverage) / price
    return max(qty, OrderManager.get_min_size(ticker))


class MLTradingBot:
    """Main bot class orchestrating data fetch, prediction and order execution."""
    def __init__(self, logger: TradingLogger) -> None:
        """Initialise the bot components and load the ML strategy."""
        self.logger = logger
        model_path = os.path.join(os.path.dirname(__file__), "model_final_5_1.h5")
        self.strategy = MLStrategy(model_path)
        self.strategy.set_logger(logger)
        self.position_manager = PositionManager(logger, timeframe="4h")
        self.active_positions: Dict[str, Dict] = {}
        self.closed_signals: set[str] = set()
        self.tickers = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "SUIUSDT"
        ]
        self.timeframe = "4h"
        self.risk_percentage = 0.25
        self.leverage = 10
        self.exchange = BybitConnector(
            api_key=os.getenv("BYBIT_API_KEY_LIVE_LSTM_MODEL"),
            api_secret=os.getenv("BYBIT_API_SECRET_LIVE_LSTM_MODEL"),
            testnet=False,
            logger=logger,
        )

    async def calculate_leveraged_pnl(self, current_price: float, entry_price: float, side: str, leverage: float) -> float:
        """Return the PnL percentage including leverage."""
        if entry_price <= 0:
            return 0.0
        is_long = side.lower().strip() in ["buy", "long"]
        diff = current_price - entry_price if is_long else entry_price - current_price
        pnl_decimal = diff / entry_price
        leveraged_pnl = pnl_decimal * leverage
        return round(leveraged_pnl * 100, 4)

    async def _open_long_position(self, ticker: str, signal: StrategySignal, equity: float) -> None:
        """Send a market order for a long position and track it."""
        if signal.signal_type != "buy" or signal.id in self.closed_signals:
            return
        position_size = calc_size(equity, signal.entry_price, ticker)
        order = OrderRequest(
            symbol=ticker,
            side="buy",
            order_type=signal.order_type,
            quantity=position_size,
            price=signal.entry_price,
        )
        try:
            await self.exchange.create_order(order)
        except Exception as e:  # ccxt throws various subclasses like InsufficientFunds
            self.logger.error(f"Failed to open position for {ticker}", e)
            return
        self.active_positions[ticker] = {
            "entry_time": datetime.now(),
            "entry_price": signal.entry_price,
            "position_size": position_size,
            "highest_pnl": 0.0,
            "current_price": signal.entry_price,
            "side": "Buy",
            "leverage": self.leverage,
            "signal_id": signal.id,
        }
        self.logger.info(f"Long position opened for {ticker}")

    async def _close_position(self, ticker: str, info: Dict, reason: str) -> None:
        """Close a position and reset tracking info."""
        await self.exchange.cancel_orders(ticker)
        close_side = "sell" if info["side"].lower() in ["buy", "long"] else "buy"
        order = OrderRequest(
            symbol=ticker,
            side=close_side,
            order_type="market",
            quantity=info["position_size"],
            reduce_only=True,
        )
        try:
            await self.exchange.create_order(order)
        except Exception as e:
            self.logger.error(f"Failed to close position for {ticker}", e)
            return
        self.logger.info(f"Position closed for {ticker}", {"reason": reason})
        self.closed_signals.add(info["signal_id"])
        self.active_positions.pop(ticker, None)
        self.position_manager.pnl_tracker.reset()

    async def process_ticker(self, ticker: str, equity: float) -> None:
        """Fetch data, generate a signal and handle open positions for one ticker."""
        try:
            df = await self.exchange.fetch_candles(ticker, self.timeframe, 150)
            df["symbol"] = ticker
            current_price = df["close"].iloc[-1]
            signal = await self.strategy.generate_signal(df, ticker)
            positions = await self.exchange.get_positions(ticker)
            current_position = positions[0] if positions else None
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}", e)
            return
        if current_position:
            info = self.active_positions.get(ticker)
            if info:
                info["current_price"] = current_price
                current_pnl = await self.calculate_leveraged_pnl(current_price, info["entry_price"], info["side"], info["leverage"])
                info = await self.position_manager.update_monitoring(info, current_pnl)
                should_close, reason, _ = self.position_manager.pnl_tracker.evaluate_position(current_pnl)
                if not should_close:
                    should_close, reason = await self.position_manager.should_close_position(info)
                if should_close:
                    await self._close_position(ticker, info, reason or "")
                    if signal and signal.signal_type == "buy":
                        await asyncio.sleep(2)
                        await self._open_long_position(ticker, signal, equity)
            else:
                self.active_positions[ticker] = {
                    "entry_time": datetime.now(),
                    "entry_price": current_position.entry_price,
                    "position_size": current_position.size,
                    "highest_pnl": 0.0,
                    "current_price": current_price,
                    "side": current_position.side,
                    "leverage": self.leverage,
                }
                self.position_manager.pnl_tracker.reset()
        elif signal and signal.signal_type == "buy":
            await self._open_long_position(ticker, signal, equity)
        elif signal and signal.signal_type == "sell" and ticker in self.active_positions:
            info = self.active_positions[ticker]
            await self._close_position(ticker, info, "Model sell signal")

    async def get_account_equity(self) -> float:
        """Return available account equity in USDT."""
        bal = await self.exchange.get_balance()
        # Use the free balance so position sizing reflects funds that can be
        # allocated to new trades.
        return float(bal["free"].get("USDT", 0.0))

    async def run_bot_iteration(self) -> None:
        """Process all tickers once using the latest equity value."""
        equity = await self.get_account_equity()
        for ticker in self.tickers:
            await self.process_ticker(ticker, equity)
            await asyncio.sleep(1)

    async def run(self) -> None:
        """Continuous loop which processes tickers every minute."""
        self.logger.info("Starting ML Trading Bot")
        try:
            while True:
                start = time.time()
                await self.run_bot_iteration()
                elapsed = time.time() - start
                await asyncio.sleep(max(60 - elapsed, 0))
        finally:
            # Ensure exchange connection is closed on exit or error.
            await self.exchange.close()


async def main() -> None:
    """Entry point for running the bot as a script."""
    load_dotenv()
    logger = TradingLogger(name="MLBot", log_file="ml_bot.log")
    bot = MLTradingBot(logger)
    await bot.run()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
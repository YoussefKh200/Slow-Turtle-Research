"""Connectors: turn RebalanceIntents into executions.

DryRunConnector is the default and only fully-implemented connector: it
journals what WOULD be sent. MT5Connector wraps the MetaTrader5 python
package behind the same interface; it activates only if the package is
installed and a terminal is running.
"""
from core import RebalanceIntent, Journal


class DryRunConnector:
    def __init__(self, journal: Journal):
        self.journal = journal

    def execute(self, intents: list[RebalanceIntent]) -> list[dict]:
        recs = []
        for it in intents:
            recs.append(self.journal.log("dry_run_order", {"intent": it.__dict__}))
            print(f"[DRY-RUN] {it.asset:12s} ({it.ticker:9s}) -> {it.target_weight:.2%}  {it.reason}")
        return recs


class MT5Connector:
    """Real-money path. Same interface as DryRunConnector.

    ponytail: rebalance-to-weight via market orders only, no partial fills,
    no requote retry loop - add when a funded account exists to test against.
    """

    def __init__(self, journal: Journal, symbol_map: dict | None = None):
        import MetaTrader5 as mt5  # noqa: F401 - fails fast if not installed
        self.mt5 = mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        self.journal = journal
        # FTMO CFD symbols; extend before going live on the rest of the universe
        self.symbol_map = symbol_map or {
            "SPY": "US500.cash", "SP500": "US500.cash", "NAS100": "US100.cash",
            "QQQ": "US100.cash", "DIA": "US30.cash", "GOLD": "XAUUSD",
            "SILVER": "XAGUSD", "OIL": "USOIL.cash", "EURUSD": "EURUSD",
            "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
        }

    def execute(self, intents: list[RebalanceIntent]) -> list[dict]:
        mt5 = self.mt5
        equity = mt5.account_info().equity
        recs = []
        for it in intents:
            symbol = self.symbol_map.get(it.asset)
            if symbol is None:
                recs.append(self.journal.log("mt5_skip", {"intent": it.__dict__,
                                                          "reason": "no MT5 symbol mapped"}))
                continue
            sym = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            target_lots = round(equity * it.target_weight / (tick.ask * sym.trade_contract_size), 2)
            held = sum(p.volume for p in (mt5.positions_get(symbol=symbol) or []))
            delta = target_lots - held
            if abs(delta) * tick.ask * sym.trade_contract_size < equity * 0.001:
                continue  # ignore dust rebalances < 0.1% of equity
            req = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                "volume": max(abs(delta), sym.volume_min),
                "type": mt5.ORDER_TYPE_BUY if delta > 0 else mt5.ORDER_TYPE_SELL,
                "price": tick.ask if delta > 0 else tick.bid,
                "comment": "slowturtle-engine",
            }
            result = mt5.order_send(req)
            recs.append(self.journal.log("mt5_order", {"intent": it.__dict__,
                                                       "result": str(result)}))
        return recs

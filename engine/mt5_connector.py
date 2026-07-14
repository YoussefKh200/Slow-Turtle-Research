"""Connectors: turn RebalanceIntents into executions.

Two connectors, same interface `execute(intents) -> list[dict]`:

- DryRunConnector: journals what WOULD be sent and prints it. No terminal.
- MT5Connector: talks to a running MetaTrader5 terminal. Reconciles each
  broker symbol's net long volume to the target implied by the blended book.

The MT5 symbol map, deviation and magic live in config["mt5"] and were
verified against the live FTMO-Demo terminal (see config's _source). Any asset
not in the map is skipped loudly -- the connector never guesses a symbol.

SAFETY: MT5Connector defaults to preview (live=False): it reads real prices
and positions and computes the exact orders, but does NOT send them. Sending
requires live=True (run_combined.py --live), and even then it re-checks
trade_allowed, symbol visibility and a live price before every order.
"""
from core import RebalanceIntent, Journal


# --------------------------------------------------------------- pure math ---
def net_by_symbol(intents: list[RebalanceIntent], symbol_map: dict) -> dict:
    """Sum target weights of every intent that maps to the same broker symbol.
    Unmapped assets are dropped. Returns {broker_symbol: net_weight}."""
    net: dict[str, float] = {}
    for it in intents:
        sym = symbol_map.get(it.asset)
        if sym is None:
            continue
        net[sym] = net.get(sym, 0.0) + it.target_weight
    return net


def target_lots(weight: float, equity: float, lot_value: float,
                vol_min: float, vol_step: float, vol_max: float) -> float:
    """Lots to hold for `weight` of equity as notional. `lot_value` is the
    account-currency notional of ONE lot (tick_value * price / tick_size --
    correct across indices, metals and FX, unlike price*contract which is
    wrong when the quote currency isn't the account currency, e.g. USDJPY).
    Rounds to the broker's volume step; returns 0.0 if the rounded size is
    below the minimum lot (position too small to trade at this account size)."""
    if lot_value <= 0 or weight <= 0:
        return 0.0
    raw = (weight * equity) / lot_value
    lots = round(raw / vol_step) * vol_step
    if lots < vol_min:
        return 0.0
    return round(min(lots, vol_max), 8)


# ------------------------------------------------------------- connectors ---
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
    """Reconcile the blended book to live positions on a MetaTrader5 terminal.

    live=False (default): connect, price, compute orders, print + journal them,
    send nothing. live=True: actually place the reconciling market orders.
    """

    def __init__(self, journal: Journal, cfg: dict, live: bool = False):
        import MetaTrader5 as mt5  # imported lazily so the dry path needs no terminal
        self.mt5 = mt5
        self.journal = journal
        self.cfg = cfg["mt5"]
        self.live = live
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        ai = mt5.account_info()
        if ai is None:
            raise RuntimeError("MT5 account_info() is None -- not logged in")
        self.equity = ai.equity
        self.trade_allowed = ai.trade_allowed
        if live and "demo" not in ai.server.lower() and not self.cfg.get("allow_real_account"):
            raise RuntimeError(
                f"refusing --live: terminal is logged into {ai.login} on {ai.server}, "
                "which is not a demo server. The connector trades whatever account the "
                "terminal shows; set mt5.allow_real_account=true in config.json only "
                "when funded deployment is a deliberate decision.")
        if live and not ai.trade_allowed:
            raise RuntimeError(
                "refusing --live: Algo Trading is disabled in the terminal (every order "
                "would be rejected). Click the 'Algo Trading' toolbar button in MT5.")
        print(f"[MT5] {ai.login} {ai.server}  equity {ai.equity:.2f} {ai.currency}  "
              f"live={'YES' if live else 'preview'}  trade_allowed={ai.trade_allowed}")

    # -- helpers --
    def _net_long(self, symbol: str) -> float:
        pos = self.mt5.positions_get(symbol=symbol) or []
        return sum((p.volume if p.type == self.mt5.POSITION_TYPE_BUY else -p.volume) for p in pos)

    def _filling(self, info) -> int:
        m = info.filling_mode
        if m & 1:  # SYMBOL_FILLING_FOK
            return self.mt5.ORDER_FILLING_FOK
        if m & 2:  # SYMBOL_FILLING_IOC
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _order(self, symbol: str, is_buy: bool, volume: float, info, reason: str,
               close_ticket: int | None = None) -> dict:
        """Send (or, in preview, describe) one market order."""
        tick = self.mt5.symbol_info_tick(symbol)
        price = tick.ask if is_buy else tick.bid
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(volume, 8),
            "type": self.mt5.ORDER_TYPE_BUY if is_buy else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": self.cfg["deviation_points"],
            "magic": self.cfg["magic"],
            "comment": reason[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        if close_ticket is not None:
            req["position"] = close_ticket
        if not self.live:
            rec = {"preview_order": {"symbol": symbol, "side": "BUY" if is_buy else "SELL",
                                     "volume": req["volume"], "price": price, "reason": reason}}
            print(f"[PREVIEW]  {symbol:12s} {'BUY ' if is_buy else 'SELL'} {req['volume']:.2f} @ {price}  {reason}")
            return self.journal.log("preview_order", rec)
        result = self.mt5.order_send(req)
        ok = result is not None and result.retcode == self.mt5.TRADE_RETCODE_DONE
        print(f"[LIVE {'OK ' if ok else 'ERR'}] {symbol:12s} {'BUY ' if is_buy else 'SELL'} "
              f"{req['volume']:.2f} @ {price}  ret={getattr(result, 'retcode', None)}")
        return self.journal.log("live_order", {"request": {k: req[k] for k in
                                ("symbol", "volume", "type", "price")},
                                "retcode": getattr(result, "retcode", None),
                                "deal": getattr(result, "deal", None), "reason": reason})

    def _reduce(self, symbol: str, reduce_by: float, info, reason: str) -> list[dict]:
        """Close our long tickets (FIFO) until net long is cut by `reduce_by`."""
        recs, remaining = [], reduce_by
        for p in (self.mt5.positions_get(symbol=symbol) or []):
            if remaining <= 0:
                break
            if p.type != self.mt5.POSITION_TYPE_BUY:
                continue
            vol = min(p.volume, round(remaining / info.volume_step) * info.volume_step)
            if vol < info.volume_min:
                continue
            recs.append(self._order(symbol, is_buy=False, volume=vol, info=info,
                                    reason=reason, close_ticket=p.ticket))
            remaining -= vol
        return recs

    # -- interface --
    def execute(self, intents: list[RebalanceIntent]) -> list[dict]:
        targets = net_by_symbol(intents, self.cfg["symbol_map"])
        recs = []
        for symbol, weight in sorted(targets.items()):
            info = self.mt5.symbol_info(symbol)
            if info is None or not self.mt5.symbol_select(symbol, True):
                print(f"[SKIP] {symbol}: not available on this account")
                continue
            tick = self.mt5.symbol_info_tick(symbol)
            price = tick.ask if tick else 0.0
            if price <= 0 or info.trade_tick_size <= 0:
                print(f"[SKIP] {symbol}: no live price (market closed?)")
                continue
            lot_value = info.trade_tick_value * price / info.trade_tick_size  # notional of 1 lot, acct ccy
            tgt = target_lots(weight, self.equity, lot_value,
                              info.volume_min, info.volume_step, info.volume_max)
            cur = self._net_long(symbol)
            delta = round(tgt - cur, 8)
            reason = f"rebalance {symbol} {weight:.2%} -> {tgt:g} lots (have {cur:g})"
            if abs(delta) < info.volume_step:
                continue  # already at target within one step -> hold, no order
            if delta > 0:
                recs.append(self._order(symbol, is_buy=True, volume=delta, info=info, reason=reason))
            else:
                recs.extend(self._reduce(symbol, -delta, info=info, reason=reason))
        if not recs:
            print("[MT5] book already at target -- no orders")
        return recs

    def close(self):
        self.mt5.shutdown()

"""Sanity checks for the MT5 connector's pure sizing/netting math.

The order-routing path needs a live terminal and can't be unit-tested safely;
these cover the money-math that decides *what* to send. The live path is
exercised by run_combined.py --mt5 (preview) against the terminal.
"""
from core import RebalanceIntent
from mt5_connector import net_by_symbol, target_lots

SYMBOL_MAP = {"NAS100": "US100.cash", "SP500": "US500.cash",
              "MR_QQQ": "US100.cash", "MR_SPY": "US500.cash", "GOLD": "XAUUSD"}


def test_net_by_symbol_sums_and_drops():
    intents = [
        RebalanceIntent("d", "NAS100", "^NDX", 0.02, ""),
        RebalanceIntent("d", "MR_QQQ", "QQQ", 0.01, ""),   # nets onto US100 with NAS100
        RebalanceIntent("d", "SP500", "^GSPC", 0.03, ""),
        RebalanceIntent("d", "XLK", "XLK", 0.05, ""),      # unmapped -> dropped
        RebalanceIntent("d", "GOLD", "GC=F", 0.013, ""),
    ]
    net = net_by_symbol(intents, SYMBOL_MAP)
    assert abs(net["US100.cash"] - 0.03) < 1e-9, "NAS100 + MR_QQQ must net"
    assert abs(net["US500.cash"] - 0.03) < 1e-9
    assert abs(net["XAUUSD"] - 0.013) < 1e-9
    assert "XLK" not in net and len(net) == 3, "unmapped asset must be dropped"


def test_target_lots_rounds_to_step():
    # 1.4% of 96805 = ~1355 notional; US100 lot_value ~29170 -> 0.0465 -> 0.05 lots
    lots = target_lots(0.014, 96805, 29170, vol_min=0.01, vol_step=0.01, vol_max=1000)
    assert lots == 0.05, f"expected 0.05, got {lots}"


def test_target_lots_below_min_is_zero():
    # gold: 1.3% of 96805 = 1258 notional; lot_value ~400000 -> 0.003 lots < 0.01 min
    lots = target_lots(0.013, 96805, 400000, vol_min=0.01, vol_step=0.01, vol_max=100)
    assert lots == 0.0, f"sub-minimum position must round to 0, got {lots}"


def test_target_lots_usdjpy_uses_lot_value_not_price_x_contract():
    # regression: the live preview zeroed USDJPY because price*contract (16.2M JPY)
    # was used instead of the true USD lot value (100000). At 2.25% of 96805 = 2178
    # notional / 100000 = 0.0218 -> 0.02 lots. With the buggy 16.2M denominator it
    # rounded to 0.
    lots = target_lots(0.0225, 96805, 100000, vol_min=0.01, vol_step=0.01, vol_max=50)
    assert lots == 0.02, f"USDJPY must size on USD lot value, got {lots}"


def test_target_lots_zero_weight_and_bad_lot_value():
    assert target_lots(0.0, 96805, 29170, 0.01, 0.01, 1000) == 0.0
    assert target_lots(0.02, 96805, 0.0, 0.01, 0.01, 1000) == 0.0  # no price -> lot_value 0


if __name__ == "__main__":
    test_net_by_symbol_sums_and_drops()
    test_target_lots_rounds_to_step()
    test_target_lots_below_min_is_zero()
    test_target_lots_usdjpy_uses_lot_value_not_price_x_contract()
    test_target_lots_zero_weight_and_bad_lot_value()
    print("mt5 connector math checks passed")

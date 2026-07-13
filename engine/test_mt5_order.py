"""DryRunConnector round-trip: intents are executed and journaled."""
import os
import tempfile
from core import RebalanceIntent, Journal
from mt5_connector import DryRunConnector


def test_dry_run_journals_orders():
    path = os.path.join(tempfile.gettempdir(), "test_journal.jsonl")
    if os.path.exists(path):
        os.remove(path)
    journal = Journal(path)
    intents = [RebalanceIntent("2026-07-17", "SPY", "SPY", 0.0269, "weekly rebalance"),
               RebalanceIntent("2026-07-17", "GOLD", "GC=F", 0.0164, "weekly rebalance")]
    recs = DryRunConnector(journal).execute(intents)
    assert len(recs) == 2
    logged = journal.read()
    assert len(logged) == 2
    assert logged[0]["kind"] == "dry_run_order"
    assert logged[1]["intent"]["target_weight"] == 0.0164
    os.remove(path)


if __name__ == "__main__":
    test_dry_run_journals_orders()
    print("mt5 connector dry-run checks passed")

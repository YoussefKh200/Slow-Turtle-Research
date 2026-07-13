"""Connectors: turn RebalanceIntents into executions.

DryRunConnector is the default and only implemented connector: it journals
what WOULD be sent and prints it. That's all the engine needs until there is
an actual MT5 account to test against.

ponytail: a real MT5Connector (order routing, lot sizing, a verified FTMO
symbol map) belongs here once a demo/funded account exists to test it
against — writing broker order logic against symbol names nobody has
confirmed is worse than not having it, since it would look done and misfire.
Add it then; the interface is just execute(intents) -> list[dict].
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

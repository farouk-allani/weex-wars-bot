#!/usr/bin/env python
"""ai-log compliance audit and backfill.

Every AI-driven order the competition sees must be accompanied by an ai-log whose
`output` matches the submitted order. A missing log is not a worse score, it is a
non-compliant order — so this is checked as an artifact-level invariant rather than
trusted to the emit path.

    python run_compliance_audit.py            # report only
    python run_compliance_audit.py --backfill # regenerate recoverable logs

Backfill is possible because the decision log is append-only: an order_link record
names its decision_id, and the decision itself (verbatim messages + full context) is
still on disk, so the ai-log can be rebuilt exactly. Anything whose decision record
is genuinely gone is reported as unrecoverable rather than filled with a guess — a
fabricated log is worse than a missing one.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ai import wars_log
from src.ai.logbook import DecisionLog


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/ai_decisions.jsonl")
    ap.add_argument("--ai-logs", default="data/ai_logs")
    ap.add_argument("--backfill", action="store_true",
                    help="rebuild ai-logs for linked orders that are missing one")
    args = ap.parse_args()

    dl = DecisionLog(args.log)
    status = dl.compliance_status(args.ai_logs)
    if status.get("error"):
        print(f"ERROR reading {args.log}: {status['error']}")
        return 2

    print("=== ai-log compliance ===")
    print(f"  orders linked          : {status['orders_linked']}")
    print(f"  ai-logs on disk        : {status['ai_logs_on_disk']}")
    print(f"  orders WITHOUT a log   : {status['orders_without_ai_log']}")
    print(f"  logs INCOMPLETE        : {status['ai_logs_incomplete']}")
    print(f"  compliant              : {status['compliant']}")

    if status["incomplete"]:
        print(f"\n--- {status['ai_logs_incomplete']} log(s) present but not usable ---")
        for c in status["incomplete"]:
            print(f"  {c['file'][:56]}")
            for pr in c["problems"]:
                print(f"      - {pr}")
        print("  NOTE: a log whose input has no verbatim message array cannot be")
        print("  repaired if the decision that produced it never captured one.")

    missing = status["missing"]
    if not missing:
        if status["compliant"]:
            print("\nAll linked orders have a complete ai-log.")
        return 0 if status["compliant"] else 1

    print(f"\n--- {len(missing)} order(s) missing an ai-log ---")
    for m in missing:
        print(f"  {str(m['timestamp'])[:19]}  {m['symbol']:<16} order={m['order_id']}")

    if not args.backfill:
        print("\nRe-run with --backfill to rebuild the recoverable ones.")
        return 1

    # Rebuild. The order params come from the order_link record, which is what was
    # actually submitted, so the regenerated log matches the trade by construction.
    links = {}
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("type") == "order_link":
                oid = str((row.get("order") or {}).get("order_id") or "")
                links[(row.get("decision_id"), oid)] = row

    done = unrecoverable = 0
    print("\n--- backfill ---")
    for m in missing:
        key = (m["decision_id"], m["order_id"])
        row = links.get(key)
        if not row:
            print(f"  {m['symbol']:<16} UNRECOVERABLE (no order_link record)")
            unrecoverable += 1
            continue
        entry = dl._read_decision(m["decision_id"])
        if entry is None:
            print(f"  {m['symbol']:<16} UNRECOVERABLE (decision {m['decision_id']} not on disk)")
            unrecoverable += 1
            continue
        try:
            path = wars_log.emit(entry, row["order"], out_dir=args.ai_logs)
        except Exception as e:
            print(f"  {m['symbol']:<16} FAILED: {e}")
            unrecoverable += 1
            continue
        # The emitted filename carries a fresh timestamp; the audit matches on the
        # decision_id + order_id prefix, so coverage is what actually closes.
        print(f"  {m['symbol']:<16} rebuilt -> {path.name}")
        done += 1

    after = dl.compliance_status(args.ai_logs)
    print(f"\n  rebuilt {done}, unrecoverable {unrecoverable}")
    print(f"  orders WITHOUT a log now : {after['orders_without_ai_log']}")
    print(f"  logs INCOMPLETE now      : {after['ai_logs_incomplete']}")
    print(f"  compliant                : {after['compliant']}")
    if after["ai_logs_incomplete"]:
        print("\n  A rebuilt log is only as complete as the decision behind it. Logs")
        print("  from before the logbook captured verbatim messages will stay")
        print("  incomplete — that is a fact about the record, not something to")
        print("  paper over by inventing a prompt.")
    return 0 if after["compliant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

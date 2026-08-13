"""Non-trading AI Wars allowlist probe using the latest authentic AI decision."""

import json
from pathlib import Path

from dotenv import load_dotenv

from src.ai.wars_log import WeexAILogUploader


def main() -> int:
    load_dotenv()
    source = Path("logs/ai_decisions.jsonl")
    last = None
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not row.get("type") and not row.get("error"):
                last = row
    if last is None:
        print("FAIL: no successful AI decision is available for the probe")
        return 1

    result = WeexAILogUploader(enabled=True).probe_allowlist(last)
    if result.get("verified"):
        print("PASS: WEEX AI-log allowlist verified (no order was placed)")
        return 0
    print(
        "FAIL: WEEX AI-log allowlist not verified: "
        + str(result.get("last_error") or "unknown error")
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

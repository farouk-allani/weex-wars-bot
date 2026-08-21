"""Focused tests for credential-bound UploadAiLog allowlist readiness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai.wars_log import WeexAILogUploader


class AllowlistBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.status_dir = self.root / "uploads"
        self.allowlist_path = self.root / "allowlist.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def uploader(
        self,
        *,
        api_key: str = "key-one",
        base_url: str = "https://api-contract.weex.com",
    ) -> WeexAILogUploader:
        uploader = WeexAILogUploader(
            enabled=True,
            status_dir=self.status_dir,
            allowlist_status_path=self.allowlist_path,
            base_url=base_url,
        )
        uploader.api_key = api_key
        uploader.api_secret = "secret-must-not-be-persisted"
        uploader.passphrase = "passphrase-must-not-be-persisted"
        return uploader

    @staticmethod
    def decision() -> dict:
        return {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "assess this snapshot"}],
            "context": {"markets": [{"symbol": "BTC/USDT:USDT", "price": 100}]},
            "raw_response": json.dumps(
                {
                    "market_assessment": "No edge.",
                    "decisions": [{"symbol": "BTC/USDT:USDT", "action": "hold"}],
                }
            ),
        }

    def verify(self, uploader: WeexAILogUploader) -> dict:
        with patch.object(uploader, "_post_payload", return_value=(True, 200, None)):
            return uploader.probe_allowlist(self.decision())

    def test_success_is_bound_without_persisting_credentials(self):
        uploader = self.uploader()

        result = self.verify(uploader)
        persisted_text = self.allowlist_path.read_text(encoding="utf-8")
        persisted = json.loads(persisted_text)

        self.assertTrue(result["verified"])
        self.assertTrue(persisted["binding_fingerprint"].startswith("sha256-v1:"))
        self.assertNotIn("key-one", persisted_text)
        self.assertNotIn("secret-must-not-be-persisted", persisted_text)
        self.assertNotIn("passphrase-must-not-be-persisted", persisted_text)
        self.assertTrue(uploader.allowlist_status()["binding_matches"])
        self.assertTrue(uploader.readiness()[0])

    def test_same_key_and_normalized_base_survive_process_restart(self):
        self.verify(self.uploader(base_url="https://api-contract.weex.com/"))

        restarted = self.uploader(base_url="https://api-contract.weex.com")

        self.assertTrue(restarted.allowlist_status()["verified"])
        self.assertTrue(restarted.readiness()[0])

    def test_api_key_rotation_invalidates_readiness(self):
        self.verify(self.uploader(api_key="key-one"))

        rotated = self.uploader(api_key="key-two")
        status = rotated.allowlist_status()

        self.assertFalse(status["verified"])
        self.assertFalse(status["binding_matches"])
        self.assertIn("re-probe", status["last_error"])
        self.assertFalse(rotated.readiness()[0])

    def test_base_url_rotation_invalidates_readiness(self):
        self.verify(self.uploader(base_url="https://api-contract.weex.com"))

        rotated = self.uploader(base_url="https://api-contract.weex.tech")
        status = rotated.allowlist_status()

        self.assertFalse(status["verified"])
        self.assertFalse(status["binding_matches"])
        self.assertIn("re-probe", status["last_error"])
        self.assertFalse(rotated.readiness()[0])

    def test_legacy_unbound_success_is_treated_as_unverified(self):
        self.allowlist_path.write_text(
            json.dumps(
                {
                    "verified": True,
                    "verified_at": "2026-02-01T00:00:00+00:00",
                    "http_status": 200,
                }
            ),
            encoding="utf-8",
        )

        uploader = self.uploader()
        status = uploader.allowlist_status()

        self.assertFalse(status["verified"])
        self.assertFalse(status["binding_matches"])
        self.assertIn("predates credential binding", status["last_error"])
        self.assertFalse(uploader.readiness()[0])


if __name__ == "__main__":
    unittest.main()

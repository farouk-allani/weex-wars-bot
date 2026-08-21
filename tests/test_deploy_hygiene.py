import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DeployHygieneTests(unittest.TestCase):
    def test_docker_context_excludes_credentials_and_runtime_state(self):
        patterns = {
            line.strip()
            for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for required in {".env*", "deploy_key*", "id_rsa*", "id_ed25519*", "*.pem", "*.key", "data", "logs"}:
            self.assertIn(required, patterns)

    def test_dashboard_is_loopback_only_and_ci_always_uses_ssh_loopback(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('"127.0.0.1:${DASHBOARD_PORT:-8787}:8787"', compose)

        workflow_path = ROOT / ".github" / "workflows" / "deploy.yml"
        if not workflow_path.exists():
            self.skipTest("CI metadata is intentionally excluded from the runtime image")
        workflow = workflow_path.read_text(encoding="utf-8")
        self.assertNotIn("DASHBOARD_PUBLIC_URL", workflow)
        self.assertIn("ssh -i ~/.ssh/deploy_key", workflow)
        self.assertIn("-o StrictHostKeyChecking=yes -o ConnectTimeout=10", workflow)
        self.assertIn("curl -fsS http://127.0.0.1:8787/api/health", workflow)
        self.assertIn('assert d.get("ok") is True', workflow)


if __name__ == "__main__":
    unittest.main()

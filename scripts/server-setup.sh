#!/usr/bin/env bash
# One-time setup on the project VPS as user farouk
#   ssh farouk@45.88.191.129
#   bash scripts/server-setup.sh
#
# Or as root (will clone into farouk's home):
#   DEPLOY_USER=farouk bash scripts/server-setup.sh

set -euo pipefail

# --- This project's VPS defaults ---
DEPLOY_USER="${DEPLOY_USER:-farouk}"
DEPLOY_HOME="${DEPLOY_HOME:-/home/farouk}"
DEPLOY_PATH="${DEPLOY_PATH:-/home/farouk/weex-wars-bot}"
REPO_URL="${REPO_URL:-https://github.com/farouk-allani/weex-wars-bot.git}"
VPS_IP="${VPS_IP:-45.88.191.129}"

echo "==> Target user:  $DEPLOY_USER"
echo "==> Home:         $DEPLOY_HOME"
echo "==> App path:     $DEPLOY_PATH"
echo "==> VPS IP:       $VPS_IP"

echo "==> Installing Docker (if needed)"
if ! command -v docker >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y ca-certificates curl git
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker "$DEPLOY_USER" || true
  else
    echo "Docker not installed. Run once as root:"
    echo "  curl -fsSL https://get.docker.com | sh"
    echo "  usermod -aG docker $DEPLOY_USER"
    exit 1
  fi
fi

# Ensure farouk can run docker
if id "$DEPLOY_USER" >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ]; then
    usermod -aG docker "$DEPLOY_USER" || true
  fi
fi

echo "==> Cloning / updating repo at $DEPLOY_PATH"
if [ ! -d "$DEPLOY_PATH/.git" ]; then
  mkdir -p "$(dirname "$DEPLOY_PATH")"
  if [ "$(id -u)" -eq 0 ]; then
    sudo -u "$DEPLOY_USER" git clone "$REPO_URL" "$DEPLOY_PATH"
  else
    git clone "$REPO_URL" "$DEPLOY_PATH"
  fi
else
  cd "$DEPLOY_PATH"
  git fetch origin main
  git reset --hard origin/main
fi

cd "$DEPLOY_PATH"

if [ ! -f .env ]; then
  echo "==> Creating .env from example (EDIT THIS)"
  cp .env.example .env
  echo "    Edit: nano $DEPLOY_PATH/.env"
fi

echo "==> First build"
docker compose build
docker compose up -d

echo ""
echo "================================================"
echo " Setup complete"
echo " User:      $DEPLOY_USER"
echo " Path:      $DEPLOY_PATH"
echo " Dashboard: http://$VPS_IP:8787"
echo " Logs:      cd $DEPLOY_PATH && docker compose logs -f bot"
echo ""
echo " GitHub secrets to set:"
echo "   DEPLOY_HOST = $VPS_IP"
echo "   DEPLOY_USER = $DEPLOY_USER"
echo "   DEPLOY_PATH = $DEPLOY_PATH"
echo "   DEPLOY_SSH_KEY = (private key for $DEPLOY_USER)"
echo "   DASHBOARD_PUBLIC_URL = http://$VPS_IP:8787"
echo "================================================"

#!/usr/bin/env bash
#
# Install MarketSwarm as a systemd service on a Debian/Ubuntu VPS.
#
#   sudo ./deploy/install.sh
#
# Idempotent: safe to re-run to upgrade an existing install.

set -euo pipefail

APP_USER="marketswarm"
APP_DIR="/opt/marketswarm"
DATA_DIR="/var/lib/marketswarm"
ENV_FILE="/etc/marketswarm/env"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root (sudo $0)"

log "Installing system packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip tzdata ca-certificates

log "Setting timezone data (the agent schedules in America/New_York)"
timedatectl set-timezone UTC 2>/dev/null || true   # container-safe; the app converts internally

if ! id -u "$APP_USER" >/dev/null 2>&1; then
    log "Creating service user $APP_USER"
    useradd --system --home-dir "$DATA_DIR" --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

log "Installing application to $APP_DIR"
mkdir -p "$APP_DIR"
cp -r "$REPO_DIR/marketswarm" "$REPO_DIR/pyproject.toml" "$REPO_DIR/requirements.txt" "$APP_DIR/"
[[ -f "$REPO_DIR/README.md" ]] && cp "$REPO_DIR/README.md" "$APP_DIR/"

log "Building virtualenv"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
"$APP_DIR/venv/bin/pip" install --quiet -e "$APP_DIR"

mkdir -p "$DATA_DIR/reports" "$(dirname "$ENV_FILE")"
chown -R "$APP_USER:$APP_USER" "$DATA_DIR" "$APP_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
    log "Writing $ENV_FILE — edit it before the first run"
    cat > "$ENV_FILE" <<'EOF'
# Required: SEC EDGAR rejects or throttles clients that do not identify themselves.
MARKETSWARM_CONTACT=you@example.com

# Optional: enables the written narrative and post-mortems.
#ANTHROPIC_API_KEY=sk-ant-...

# Optional: adds authoritative macro series to the economic calendar section.
#FRED_API_KEY=

# Optional: pushes the morning summary to Discord/Slack.
#MARKETSWARM_WEBHOOK=https://discord.com/api/webhooks/...

# Schedule (Eastern Time). The daemon handles DST itself.
MARKETSWARM_RUN_TIME=08:15

# Optional: override the watchlist.
#MARKETSWARM_UNIVERSE=SPY,QQQ,NVDA,AAPL,MSFT,AMZN,META,TSLA,AMD
EOF
    chmod 600 "$ENV_FILE"
    chown root:root "$ENV_FILE"
fi

log "Installing systemd unit"
cp "$REPO_DIR/deploy/marketswarm.service" /etc/systemd/system/marketswarm.service
systemctl daemon-reload
systemctl enable marketswarm.service

cat <<EOF

Installed.

  1. Edit credentials:      sudo nano $ENV_FILE
  2. Verify the setup:      sudo -u $APP_USER $APP_DIR/venv/bin/marketswarm status
  3. Start the daemon:      sudo systemctl start marketswarm
  4. Watch it:              sudo journalctl -u marketswarm -f

Reports are written to $DATA_DIR/reports (latest.html is always the most recent).
Run one manually at any time:

  sudo -u $APP_USER $APP_DIR/venv/bin/marketswarm run

EOF

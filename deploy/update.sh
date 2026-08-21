#!/usr/bin/env bash
#
# Update a running MarketSwarm install to the current checkout.
#
#   cd /path/to/repo && sudo ./deploy/update.sh
#
# What it does, in order:
#   1. notes which services are running, before anything can stop them
#   2. backs up every database, before anything else touches them
#   3. reinstalls the code and dependencies over the running install
#   4. removes a stale systemd drop-in if one is shadowing the unit
#   5. restarts what was running, and reports whether it stayed up
#
# Safe to re-run. It does not touch /etc/marketswarm/env, so credentials and
# the webhook survive an update.

set -euo pipefail

APP_USER="marketswarm"
APP_DIR="/opt/marketswarm"
DATA_DIR="/var/lib/marketswarm"
DROPIN_DIR="/etc/systemd/system/marketswarm.service.d"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root (sudo $0)"
[[ -d "$APP_DIR" ]] || die "$APP_DIR not found — this is an update, run deploy/install.sh first"

# --- 1. which services are actually running ------------------------------
# Recorded first, because the backup's plain-copy fallback stops them. Noted
# afterwards, a stopped service looks like one that was never running and never
# gets started again.
WAS_RUNNING=()
for unit in marketswarm marketswarm-bot; do
    if systemctl is-active --quiet "$unit" 2>/dev/null; then
        WAS_RUNNING+=("$unit")
    fi
done
log "Running before update: ${WAS_RUNNING[*]:-none}"

# --- 2. back up the databases before anything else touches them ----------
# The learning history is the part that cannot be regenerated: re-running the
# swarm gives you today's report back, but not months of resolved predictions.
# backup-db.sh aborts on an empty backup, so a failure here stops the update.
"$REPO_DIR/deploy/backup-db.sh" "$DATA_DIR" "$APP_USER"

# --- 3. reinstall the code -----------------------------------------------
# install.sh is idempotent and does the copy, the venv and the units. Reusing
# it means there is one definition of a correct install rather than two that
# drift apart.
log "Reinstalling from $REPO_DIR"
"$REPO_DIR/deploy/install.sh" >/dev/null

# --- 4. clear a stale drop-in --------------------------------------------
# A drop-in that was added to work around a bug in the packaged unit will keep
# overriding the fixed unit forever, which is how a fix gets applied and has no
# effect. Removed only if it exists.
if [[ -d "$DROPIN_DIR" ]]; then
    log "Removing systemd drop-in at $DROPIN_DIR (the packaged unit is correct now)"
    rm -rf "$DROPIN_DIR"
    systemctl daemon-reload
fi

# --- 5. restart and check it stayed up -----------------------------------
FAILED=0
if (( ${#WAS_RUNNING[@]} )); then
    for unit in "${WAS_RUNNING[@]}"; do
        log "Restarting $unit"
        systemctl restart "$unit"
    done

    # One look is not enough: a service that starts and dies looks identical to
    # a healthy one for the first few seconds. One look after the dust settles
    # is, though — the unit's RestartSec is 30, so anything that died on start
    # is still down here rather than quietly back up and hiding it.
    sleep 20
    for unit in "${WAS_RUNNING[@]}"; do
        if systemctl is-active --quiet "$unit"; then
            log "$unit is running"
        else
            warn "$unit is NOT running after the update"
            journalctl -u "$unit" -n 25 --no-pager --full || true
            FAILED=1
        fi
    done
fi

echo
if [[ $FAILED -eq 0 ]]; then
    log "Update complete — everything that was running is still running."
else
    warn "Update finished with problems. The database backup above is intact."
fi

cat <<EOF

Check it did what you expect:

  marketswarm status        sudo marketswarm-cli status
  live logs                 sudo journalctl -u marketswarm -f
  force a run now           sudo systemctl start marketswarm

If the interactive bot is configured but not yet enabled:

  sudo systemctl enable --now marketswarm-bot

EOF

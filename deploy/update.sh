#!/usr/bin/env bash
#
# Update a running MarketSwarm install to the current checkout.
#
#   cd /path/to/repo && sudo ./deploy/update.sh
#
# What it does, in order:
#   1. backs up the database (before anything else touches it)
#   2. reinstalls the code and dependencies over the running install
#   3. removes a stale systemd drop-in if one is shadowing the unit
#   4. restarts what is running, and reports whether it stayed up
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
# Recorded first, because the fallback backup path below stops them. Noted
# afterwards, a stopped service looks like one that was never running and never
# gets started again.
WAS_RUNNING=()
for unit in marketswarm marketswarm-bot; do
    if systemctl is-active --quiet "$unit" 2>/dev/null; then
        WAS_RUNNING+=("$unit")
    fi
done
log "Running before update: ${WAS_RUNNING[*]:-none}"

# --- 2. back up the database before anything else touches it -------------
# The learning history is the part that cannot be regenerated: re-running the
# swarm gives you today's report back, but not months of resolved predictions.
#
# Every *.db in the data directory is backed up rather than one name spelled out
# here. A script that guesses the filename and misses prints a reassuring
# "nothing to back up" while protecting nothing, which is worse than having no
# backup step at all — that is not hypothetical, this script looked for
# marketswarm.db for a while and the application writes memory.db.
shopt -s nullglob
DBS=("$DATA_DIR"/*.db)
shopt -u nullglob

if (( ${#DBS[@]} )); then
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$DATA_DIR/backups"
    for DB in "${DBS[@]}"; do
        BASE="$(basename "$DB" .db)"
        BACKUP="$DATA_DIR/backups/$BASE-$STAMP.db"
        # sqlite3 .backup is safe against a live writer; cp is not. Fall back
        # only if the sqlite3 binary is absent, and say so rather than pretending.
        if command -v sqlite3 >/dev/null 2>&1; then
            sqlite3 "$DB" ".backup '$BACKUP'"
        else
            warn "sqlite3 not installed — falling back to a plain copy, which is"
            warn "only safe while nothing is writing. Stopping services first."
            systemctl stop marketswarm marketswarm-bot 2>/dev/null || true
            cp "$DB" "$BACKUP"
        fi
        # A backup that silently produced nothing is not a backup. Refuse to go
        # on rather than update with an imaginary safety net behind us.
        [[ -s "$BACKUP" ]] || die "backup of $DB came out empty — stopping before the update"
        log "Backed up $(basename "$DB") -> $BACKUP"
    done
    chown -R "$APP_USER:$APP_USER" "$DATA_DIR/backups"
else
    # Show what was actually searched, so "nothing to back up" is something the
    # reader can check rather than has to trust.
    log "No .db file in $DATA_DIR — nothing to back up. That directory holds:"
    ls -A "$DATA_DIR" 2>/dev/null | sed 's/^/      /' | head -10 || log "      (nothing)"
fi

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
for unit in "${WAS_RUNNING[@]:-}"; do
    [[ -n "$unit" ]] || continue
    log "Restarting $unit"
    systemctl restart "$unit"
done

sleep 8
FAILED=0
for unit in "${WAS_RUNNING[@]:-}"; do
    [[ -n "$unit" ]] || continue
    if systemctl is-active --quiet "$unit"; then
        log "$unit is running"
    else
        warn "$unit is NOT running after the update"
        systemctl status "$unit" --no-pager -n 15 || true
        FAILED=1
    fi
done

# A service that starts and dies looks identical to a healthy one for the first
# few seconds, so check again rather than trusting the first look.
sleep 12
for unit in "${WAS_RUNNING[@]:-}"; do
    [[ -n "$unit" ]] || continue
    if ! systemctl is-active --quiet "$unit"; then
        warn "$unit died after starting — likely a crash loop"
        journalctl -u "$unit" -n 25 --no-pager --full || true
        FAILED=1
    fi
done

echo
if [[ $FAILED -eq 0 ]]; then
    log "Update complete — everything that was running is still running."
else
    warn "Update finished with problems. The database backup above is intact."
fi

# The service unit sets the data directory explicitly; the CLI's own default is
# ~/.marketswarm, which for this user resolves to a *second*, empty directory
# beside the real one. A hand-run "marketswarm status" without these reports
# cheerfully on the wrong place, so every command printed below carries them.
RUN_CLI="sudo -u $APP_USER env MARKETSWARM_DATA_DIR=$DATA_DIR MARKETSWARM_REPORT_DIR=$DATA_DIR/reports $APP_DIR/venv/bin/marketswarm"

cat <<EOF

Check it did what you expect:

  marketswarm status        $RUN_CLI status
  live logs                 sudo journalctl -u marketswarm -f
  force a run now           sudo systemctl start marketswarm

If the interactive bot is configured but not yet enabled:

  sudo systemctl enable --now marketswarm-bot

EOF

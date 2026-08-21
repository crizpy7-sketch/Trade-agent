#!/usr/bin/env bash
#
# Back up every database in a MarketSwarm data directory.
#
#   deploy/backup-db.sh <data-dir> [owner]
#
# Split out of update.sh so it can be run — and tested — on its own. The bug
# this exists to prevent is not a crash: it is a backup step that looks for a
# filename the application does not use, finds nothing, prints "nothing to back
# up", and lets the update proceed over an unprotected database. A step that
# fails that way can only be caught by running it against a real directory, so
# it lives in a file a test can execute.
#
# Every *.db is backed up rather than one name written out here, because a name
# repeated by hand drifts from marketswarm/config.py and the drift is silent.

set -euo pipefail

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

DATA_DIR="${1:?usage: $0 <data-dir> [owner]}"
OWNER="${2:-}"

[[ -d "$DATA_DIR" ]] || die "$DATA_DIR does not exist"

shopt -s nullglob
DBS=("$DATA_DIR"/*.db)
shopt -u nullglob

if (( ${#DBS[@]} == 0 )); then
    # Say what was searched, so "nothing to back up" is checkable rather than
    # something the reader has to take on trust.
    log "No .db file in $DATA_DIR — nothing to back up. That directory holds:"
    ls -A "$DATA_DIR" | sed 's/^/      /' | head -10
    exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DATA_DIR/backups"

for DB in "${DBS[@]}"; do
    BACKUP="$DATA_DIR/backups/$(basename "$DB" .db)-$STAMP.db"
    # sqlite3 .backup is safe against a live writer; cp is not. Fall back only
    # if the sqlite3 binary is absent, and say so rather than pretending.
    if command -v sqlite3 >/dev/null 2>&1; then
        sqlite3 "$DB" ".backup '$BACKUP'"
    else
        warn "sqlite3 not installed — copying instead, which is only safe while"
        warn "nothing is writing. Stopping the services first."
        systemctl stop marketswarm marketswarm-bot 2>/dev/null || true
        cp "$DB" "$BACKUP"
    fi
    # A backup that silently produced nothing is not a backup. Refuse to go on
    # rather than let an update proceed behind an imaginary safety net.
    [[ -s "$BACKUP" ]] || die "backup of $DB came out empty"
    [[ -z "$OWNER" ]] || chown "$OWNER:$OWNER" "$BACKUP"
    log "Backed up $(basename "$DB") -> $BACKUP"
done

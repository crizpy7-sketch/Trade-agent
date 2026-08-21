"""The redeploy script's safety properties.

A backup step that looks for the wrong filename is worse than no backup step at
all: it prints "nothing to back up" and the reader believes it. That is exactly
what happened once — the script looked for ``marketswarm.db`` while the
application writes ``memory.db`` — so what is checked here is the agreement
between the script and the application, not the script on its own.

These are text assertions over the shell source. They cannot prove the script
runs correctly on a live host; they prove the two halves still agree about the
things that go silently wrong when they drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

from marketswarm.config import Config

SCRIPT = (Path(__file__).resolve().parent.parent / "deploy" / "update.sh").read_text()


def test_the_backup_covers_the_database_the_application_actually_writes():
    """The one that was wrong. config.py names the file; the script must find it."""
    name = Config().db_path.name
    assert name.endswith(".db"), \
        f"the app writes {name!r}, which the *.db backup glob would not match"
    assert '"$DATA_DIR"/*.db' in SCRIPT, \
        "update.sh no longer globs for databases — it will miss a renamed one"


def test_no_database_filename_is_hard_coded():
    """A name repeated by hand drifts from config.py, and the drift is silent."""
    hardcoded = re.findall(r'DATA_DIR[/"]*/?([A-Za-z0-9_-]+\.db)', SCRIPT)
    assert not hardcoded, f"hard-coded database name(s) in update.sh: {hardcoded}"


def test_a_backup_that_produced_nothing_stops_the_update():
    """An empty backup file is not a backup. Better to refuse than to proceed."""
    assert '-s "$BACKUP"' in SCRIPT, \
        "update.sh does not verify the backup is non-empty before continuing"


def test_running_services_are_recorded_before_anything_can_stop_them():
    """The fallback path stops services. Recorded after, they never come back."""
    assert "WAS_RUNNING=(" in SCRIPT
    assert SCRIPT.index("WAS_RUNNING=(") < SCRIPT.index("systemctl stop"), \
        "services are stopped before the script notes which ones to restart"


def test_the_no_database_message_shows_what_it_looked_at():
    """'Nothing to back up' must be checkable, not taken on trust."""
    # Anchored on the log call, not the phrase — the phrase also appears in
    # the comment explaining why this branch has to be self-evidencing.
    idx = SCRIPT.index('log "No .db file in')
    assert 'ls -A "$DATA_DIR"' in SCRIPT[idx:idx + 400], \
        "the empty case does not show the directory it searched"


# ------------------------------------------------- what the scripts print

INSTALL = (Path(__file__).resolve().parent.parent / "deploy" / "install.sh").read_text()


def _cli_invocations(script: str) -> list[str]:
    return [ln.strip() for ln in script.splitlines()
            if "venv/bin/marketswarm" in ln and not ln.strip().startswith("#")]


def test_every_printed_cli_command_sets_the_data_directory():
    """The service sets MARKETSWARM_DATA_DIR; the CLI's default is elsewhere.

    A printed ``marketswarm status`` without it reads ~/.marketswarm — for this
    system user, an empty directory sitting right beside the real one — and
    reports a healthy, empty swarm. Wrong place, reassuring answer.
    """
    for script, name in ((INSTALL, "install.sh"), (SCRIPT, "update.sh")):
        for line in _cli_invocations(script):
            assert "MARKETSWARM_DATA_DIR" in line, \
                f"{name} invokes the CLI without a data directory: {line}"


def test_the_cli_prefix_is_defined_once_per_script():
    """Spelled out at each use, one copy gets fixed and the others do not."""
    for script, name in ((INSTALL, "install.sh"), (SCRIPT, "update.sh")):
        defs = [ln for ln in script.splitlines() if ln.startswith("RUN_CLI=")]
        assert len(defs) == 1, f"{name} defines RUN_CLI {len(defs)} times"

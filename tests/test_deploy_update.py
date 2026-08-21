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


def test_the_scripts_never_invoke_the_venv_binary_directly():
    """Every hand-run route must go through the wrapper.

    The venv binary on its own misses the data directory (reading an empty
    ~/.marketswarm beside the real one) and the credentials file (a configured
    webhook reporting as "not set"). Both answer confidently about the wrong
    environment. The wrapper is the only place that path may appear.
    """
    docs = (Path(__file__).resolve().parent.parent / "INSTALL.txt").read_text()
    for script, name in ((INSTALL, "install.sh"), (SCRIPT, "update.sh"),
                         (docs, "INSTALL.txt")):
        offenders = [ln.strip() for ln in script.splitlines()
                     if "venv/bin/marketswarm" in ln and not ln.strip().startswith("#")]
        assert not offenders, \
            f"{name} runs the CLI without the wrapper: {offenders}"


def test_the_cli_prefix_is_defined_once_per_script():
    """Spelled out at each use, one copy gets fixed and the others do not."""
    for script, name in ((INSTALL, "install.sh"), (SCRIPT, "update.sh")):
        defs = [ln for ln in script.splitlines() if ln.startswith("RUN_CLI=")]
        assert len(defs) == 1, f"{name} defines RUN_CLI {len(defs)} times"


# ------------------------------------------------------- the CLI wrapper

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
WRAPPER = (DEPLOY / "marketswarm-cli").read_text()
UNIT = (DEPLOY / "marketswarm.service").read_text()


def _unit_env(name: str) -> str:
    for line in UNIT.splitlines():
        if line.startswith(f"Environment={name}="):
            return line.split("=", 2)[2]
    raise AssertionError(f"{name} is not set in marketswarm.service")


def test_the_wrapper_uses_the_same_directories_as_the_service():
    """The whole point of the wrapper is to be the service's environment.

    If these drift, a hand-run check reports on a different directory than the
    daemon writes to — and reports it confidently.
    """
    for var in ("MARKETSWARM_DATA_DIR", "MARKETSWARM_REPORT_DIR"):
        assert f"{var}={_unit_env(var)}\n" in WRAPPER, \
            f"{var} in marketswarm-cli does not match marketswarm.service"


def test_the_wrapper_runs_as_the_same_user_and_binary_as_the_service():
    user = next(l.split("=", 1)[1] for l in UNIT.splitlines() if l.startswith("User="))
    exec_start = next(l for l in UNIT.splitlines() if l.startswith("ExecStart="))
    binary = exec_start.split("=", 1)[1].split()[0]
    assert f"-u {user} {binary}" in WRAPPER, \
        "marketswarm-cli runs a different user or binary than the service does"


def test_the_wrapper_loads_the_credentials_file():
    """Without it, a configured webhook reports as 'not set'."""
    assert ". /etc/marketswarm/env" in WRAPPER


def test_the_wrapper_keeps_secrets_out_of_the_process_list():
    """/proc/<pid>/cmdline is world-readable; the environment is not.

    Passing the env file through `env $(cat ...)` or xargs would put the Discord
    token where any local account can read it.
    """
    for bad in ("xargs", "env $(", "$(cat /etc/marketswarm/env)", "$(grep"):
        assert bad not in WRAPPER, f"marketswarm-cli exposes secrets via {bad!r}"


def test_the_installer_installs_the_wrapper():
    assert "marketswarm-cli" in INSTALL, "install.sh never installs the wrapper"

"""The deploy scripts' safety properties.

Two incidents shaped what is checked here, and both were the same shape: a step
that looked in the wrong place and reported confidently about it.

The backup step looked for ``marketswarm.db`` while the application writes
``memory.db``, so it found nothing, printed "nothing to back up", and let the
update proceed over an unprotected database. That one is covered by *running*
deploy/backup-db.sh against a real directory — a test written to match the fixed
text would have passed against the bug just as happily.

``marketswarm status`` run by hand read a different data directory and a
different credentials file than the daemon, so it described a healthy empty
swarm sitting beside the real one. That one is covered by pinning the wrapper to
the unit, which is a drift detector rather than a proof: it compares two files in
the repo and cannot see the installed unit or a value overridden in
/etc/marketswarm/env.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess

import pytest

from pathlib import Path

from marketswarm.config import Config

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
BACKUP_SH = DEPLOY / "backup-db.sh"
UPDATE = (DEPLOY / "update.sh").read_text()
INSTALL = (DEPLOY / "install.sh").read_text()
WRAPPER = (DEPLOY / "marketswarm-cli").read_text()
UNIT = (DEPLOY / "marketswarm.service").read_text()
DOCS = (ROOT / "INSTALL.txt").read_text()


def run_backup(data_dir: Path, *, sqlite3_available: bool = True):
    """Run the real backup script against a real directory."""
    env = {"PATH": "/usr/bin:/bin" if sqlite3_available else str(data_dir / "empty-path")}
    return subprocess.run(
        ["bash", str(BACKUP_SH), str(data_dir)],
        capture_output=True, text=True, env=env,
    )


def make_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("create table predictions (id integer primary key, symbol text)")
    conn.execute("insert into predictions (symbol) values ('NVDA')")
    conn.commit()
    conn.close()
    return path


# ------------------------------------------------------- backing up, for real

def test_the_database_the_application_writes_is_the_one_backed_up(tmp_path):
    """The original bug, reproduced end to end.

    The name comes from marketswarm.config, so this fails if the script ever
    goes back to looking for a filename spelled out by hand.
    """
    make_db(tmp_path / Config().db_path.name)

    result = run_backup(tmp_path)

    assert result.returncode == 0, result.stderr
    backups = list((tmp_path / "backups").glob("*.db"))
    assert len(backups) == 1, f"expected one backup, got {backups}"
    assert backups[0].stat().st_size > 0
    assert "nothing to back up" not in result.stdout


def test_a_backup_is_a_readable_copy_not_just_a_file_of_the_right_size(tmp_path):
    make_db(tmp_path / Config().db_path.name)
    run_backup(tmp_path)

    backup = next((tmp_path / "backups").glob("*.db"))
    conn = sqlite3.connect(backup)
    assert conn.execute("select symbol from predictions").fetchall() == [("NVDA",)]
    conn.close()


def test_every_database_is_backed_up_not_only_the_first(tmp_path):
    """A rename or a second store must not fall out of the backup silently."""
    make_db(tmp_path / Config().db_path.name)
    make_db(tmp_path / "experiments.db")

    run_backup(tmp_path)

    assert len(list((tmp_path / "backups").glob("*.db"))) == 2


def test_an_empty_directory_reports_what_it_actually_looked_at(tmp_path):
    """'Nothing to back up' has to be checkable, not taken on trust."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "cache").mkdir()

    result = run_backup(tmp_path)

    assert result.returncode == 0
    assert "nothing to back up" in result.stdout
    # The directory listing is the evidence for the claim.
    assert "reports" in result.stdout and "cache" in result.stdout


def test_a_missing_data_directory_is_an_error_not_a_quiet_success(tmp_path):
    result = run_backup(tmp_path / "does-not-exist")
    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_an_empty_backup_aborts_rather_than_reporting_success(tmp_path):
    """The safety net has to fail loudly, or it is not a safety net.

    An empty source file stands in for any way the copy can come out empty; the
    script must refuse rather than let an update proceed behind it.
    """
    (tmp_path / Config().db_path.name).touch()

    result = run_backup(tmp_path)

    assert result.returncode != 0
    assert "empty" in result.stderr


@pytest.mark.skipif(shutil.which("sqlite3") is not None,
                    reason="sqlite3 present, so the fallback path is not taken")
def test_without_sqlite3_the_copy_fallback_still_produces_a_usable_backup(tmp_path):
    """The fallback is the path this host actually takes — it has to work."""
    make_db(tmp_path / Config().db_path.name)

    result = run_backup(tmp_path)

    assert result.returncode == 0
    assert "sqlite3 not installed" in result.stdout + result.stderr
    backup = next((tmp_path / "backups").glob("*.db"))
    conn = sqlite3.connect(backup)
    assert conn.execute("select symbol from predictions").fetchall() == [("NVDA",)]
    conn.close()


# ------------------------------------------------ the wrapper against the unit

def _unit_field(prefix: str) -> str:
    for line in UNIT.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    raise AssertionError(f"no {prefix!r} line in marketswarm.service")


@pytest.mark.parametrize("var", ["MARKETSWARM_DATA_DIR", "MARKETSWARM_REPORT_DIR"])
def test_the_wrapper_uses_the_same_directories_as_the_service(var):
    """Drift here means a hand-run check reads a directory the daemon does not."""
    value = _unit_field(f"Environment={var}=")
    assert f"export {var}={value}\n" in WRAPPER, \
        f"{var} in marketswarm-cli does not match marketswarm.service"


def test_the_wrapper_runs_as_the_same_user_and_binary_as_the_service():
    user = _unit_field("User=")
    binary = _unit_field("ExecStart=").split()[0]
    assert f"-u {user} {binary}" in WRAPPER, \
        "marketswarm-cli runs a different user or binary than the service does"


def test_the_credentials_file_can_override_the_wrapper_defaults():
    """Precedence has to match the unit, which sets Environment= before
    EnvironmentFile= and so lets /etc/marketswarm/env win.

    Reversed, relocating the data directory would move the daemon and leave the
    wrapper reporting on the abandoned one — the bug the wrapper exists to stop,
    rebuilt inside it.
    """
    defaults = WRAPPER.index("export MARKETSWARM_DATA_DIR=")
    sourced = WRAPPER.index(". /etc/marketswarm/env")
    assert defaults < sourced, \
        "marketswarm-cli overwrites the credentials file's values instead of " \
        "defaulting beneath them"


def test_the_wrapper_keeps_secrets_out_of_the_process_list():
    """/proc/<pid>/cmdline is world-readable; the environment is not.

    Passing the env file through `env $(cat ...)` or xargs would put the Discord
    token where any local account can read it.
    """
    for bad in ("xargs", "env $(", "$(cat /etc/marketswarm/env)", "$(grep"):
        assert bad not in WRAPPER, f"marketswarm-cli exposes secrets via {bad!r}"


# --------------------------------------------- what the scripts and docs print

def test_nothing_tells_the_operator_to_run_the_venv_binary_directly():
    """Every hand-run route must go through the wrapper.

    The check is for the directory, not the full binary path: INSTALL.txt had a
    line ending at ``venv/bin/`` with the command name on the line above, and a
    narrower pattern walked straight past it.
    """
    for text, name in ((INSTALL, "install.sh"), (UPDATE, "update.sh"),
                       (DOCS, "INSTALL.txt")):
        offenders = [ln.strip() for ln in text.splitlines()
                     if "/opt/marketswarm/venv/bin" in ln
                     and not ln.strip().startswith("#")]
        assert not offenders, f"{name} bypasses the wrapper: {offenders}"


def test_the_installer_installs_the_wrapper():
    assert "marketswarm-cli" in INSTALL, "install.sh never installs the wrapper"


def test_the_update_delegates_the_backup_rather_than_repeating_it():
    """Two copies of the backup logic is how one of them gets fixed."""
    assert "backup-db.sh" in UPDATE
    assert "sqlite3" not in UPDATE, "update.sh has its own copy of the backup"


def test_services_are_recorded_before_anything_can_stop_them():
    """The plain-copy fallback stops them. Recorded after, they never restart."""
    assert UPDATE.index("WAS_RUNNING=(") < UPDATE.index("backup-db.sh")

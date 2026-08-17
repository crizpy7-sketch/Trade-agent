# Security

## Threat model

MarketSwarm ingests text from RSS feeds, SEC filings, news aggregators and
option chains — all attacker-influenceable by anyone who can publish a
headline — and some reaches an LLM prompt. It also writes files and posts to
webhooks.

## Controls

| Threat | Control | Test |
|---|---|---|
| Command injection | **No shell path exists.** `assert_no_shell_execution()` scans the package for `subprocess`, `os.system`, `eval`, `exec` and fails the build | `test_no_shell_execution_exists_anywhere` |
| Prompt injection | 10 patterns detected and *defused* (not deleted — a real headline may contain the phrase); role markup stripped; length capped; content fenced as untrusted DATA | `test_prompt_injection_attempts_are_neutralised` |
| Secret exfiltration | 7 credential patterns redacted from logs, reports and prompts; `RedactingFilter` on every log handler | `test_secrets_never_survive_redaction` |
| Path traversal | `safe_output_path` rejects `..`, absolute paths, backslashes and null bytes, then confines to the base directory | `test_path_traversal_is_blocked` |
| Unsafe symbols | `safe_symbol` allows only `[A-Z0-9.\-^=]{1,12}` | `test_hostile_symbols_are_rejected` |
| SSRF | `check_outbound_url` rejects non-HTTP schemes and private/loopback/link-local/unresolvable hosts | `test_ssrf_targets_are_refused` |
| Privilege creep | Tool allowlist; unknown tools denied by default | `test_unknown_tools_are_denied_by_default` |
| Financial actions | `broker_order`, `broker_cancel`, `funds_transfer`, `portfolio_allocate` permanently forbidden | `test_forbidden_capabilities_are_permanently_denied` |

## Secrets

Environment variables only. Never written to `config.yaml`, the database, or a
report. `Config.to_yaml()` excludes them — test-enforced.

## Read-only dashboard

`api.open_api()` opens SQLite with `mode=ro`. A dashboard bug cannot write to
production state — `test_api_read_only_connection_rejects_writes`.

## Hard boundary

MarketSwarm is a research and decision-support system. It has no brokerage
integration, no order path and no funds access, and the allowlist makes that
structural rather than incidental.

## Residual risks

- Prompt-injection defence is layered mitigation, not proof. A sufficiently
  novel injection could still influence a narrative. The narrative is
  non-load-bearing by design; the numbers come from code.
- Webhook URLs are validated but unsigned.
- Dependency vulnerabilities are not scanned in-repo; run `pip-audit` on the
  VPS.

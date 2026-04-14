"""Secret, sensitive-path and exfiltration patterns shared by all security hooks.

Keep this file dependency-free (stdlib only) so /usr/bin/python3 can run it.
All regexes are pre-compiled; the public surface is a handful of inspect helpers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Secret signatures (substring / regex fingerprints of high-entropy tokens)
# ---------------------------------------------------------------------------

SECRET_SIGNATURES: list[tuple[str, re.Pattern[str]]] = [
    ("Anthropic API key",           re.compile(r"sk-ant-(?:api|admin|sid)\d{0,3}-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI API key",              re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("AWS access key id",           re.compile(r"\b(?:AKIA|ASIA|AIDA|AGPA|ANPA|AROA)[0-9A-Z]{16}\b")),
    ("AWS secret access key",       re.compile(r"(?i)aws(.{0,20})?(secret|access)[_\-]?key[\"' :=]+[A-Za-z0-9/+=]{40}")),
    ("GitHub personal token",       re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("GitLab personal token",       re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b")),
    ("Slack token",                 re.compile(r"\bxox[abpsr]-[A-Za-z0-9\-]{10,}\b")),
    ("Stripe key",                  re.compile(r"\b(?:sk|rk|pk)_(?:test|live)_[A-Za-z0-9]{20,}\b")),
    ("Google API key",              re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Google OAuth client secret",  re.compile(r"GOCSPX-[A-Za-z0-9_\-]{20,}")),
    ("JWT",                         re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("Private key header",          re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----")),
    ("SSH private key body",        re.compile(r"\b(?:b3BlbnNzaC1rZXktdjE|ssh-rsa-priv|PuTTY-User-Key-File)")),
    ("NPM auth token",              re.compile(r"//[A-Za-z0-9.\-]+/:_authToken=[A-Za-z0-9\-]{8,}")),
    ("Generic hi-entropy assign",   re.compile(r"(?i)(?:api[_\-]?key|apikey|api[_\-]?secret|access[_\-]?token|auth[_\-]?token|secret[_\-]?key|client[_\-]?secret|bearer)[\"' :=]{1,5}[\"']?[A-Za-z0-9_\-]{24,}[\"']?")),
]


# ---------------------------------------------------------------------------
# Sensitive filesystem paths (substring match, case-insensitive where noted)
# ---------------------------------------------------------------------------
# Split in two tiers:
#   BLOCK_READ_PATHS     — reading the *contents* of these is dangerous and
#                          should be blocked unless the user explicitly named
#                          the file in this session.  The pre-tool guard is
#                          conservative here: it blocks only if combined with
#                          an outbound step OR with base64/gzip/tar encoding.
#   PRIVATE_KEY_PATHS    — absolute black-list; contents must never leave the
#                          machine or be echoed back to the model.
# ---------------------------------------------------------------------------

PRIVATE_KEY_PATH_RES = [
    # ~/.ssh/id_rsa, $HOME/.ssh/id_ed25519, /Users/x/.ssh/my_rsa, ./.ssh/id_rsa, .ssh/id_rsa
    re.compile(
        r"(?:(?:~|\$HOME|/Users/[^/\s\"']+|/home/[^/\s\"']+)/)?"
        r"\.ssh/"
        r"(?:id_[A-Za-z0-9_]+|[A-Za-z0-9_\-]+_(?:rsa|ed25519|ecdsa|dsa))"
        r"(?!\.pub)\b",
        re.IGNORECASE,
    ),
    # Shell-glob / indirection forms that target private keys without naming one literally:
    #   cat ~/.ssh/id_*, cat ~/.ssh/*_ed25519, cat ~/.ssh/*rsa, $(ls ~/.ssh/id_*)
    re.compile(r"\.ssh/(?:id_\*|\*(?:_(?:rsa|ed25519|ecdsa|dsa)|rsa))\b", re.IGNORECASE),
    re.compile(r"\.ssh/\*(?!\.pub)"),
    re.compile(r"\.ssh/id_(?!.*\.pub)\S+"),
    re.compile(r"\.aws/credentials\b"),
    re.compile(r"\.aws/config\b"),
    re.compile(r"\.config/gcloud/(?:credentials|application_default_credentials)"),
    re.compile(r"\.kube/config\b"),
    re.compile(r"\.gnupg/(?:secring|private-keys|trustdb)"),
    re.compile(r"\.netrc\b"),
    re.compile(r"\.pgpass\b"),
    re.compile(r"\.npmrc\b"),
    re.compile(r"\.pypirc\b"),
    re.compile(r"\.dockercfg\b"),
    re.compile(r"docker/config\.json\b"),
    re.compile(r"Library/Keychains/"),
    re.compile(r"/etc/(?:shadow|sudoers|master\.passwd)\b"),
]

# Broader "sensitive but context-dependent" paths.
SENSITIVE_CONTENT_PATH_RES = PRIVATE_KEY_PATH_RES + [
    re.compile(r"(?:^|[\s:=\"'/])\.env(?:\.[A-Za-z0-9_.\-]+)?\b"),
    re.compile(r"(?:^|[\s:=\"'/])secrets?(?:\.(?:json|yaml|yml|env|txt|ini|conf|toml))\b", re.IGNORECASE),
    re.compile(r"\.(?:pem|key|p12|pfx|jks|keystore)\b"),
    re.compile(r"\.claude/(?:settings(?:\.local)?\.json|config\.json|history\.jsonl|sessions/|projects/|session-env/)"),
    re.compile(r"(?:\.bash_history|\.zsh_history|\.python_history|\.psql_history|\.mysql_history|\.sqlite_history)\b"),
    re.compile(r"Library/Mail/"),
    re.compile(r"Library/Messages/"),
    re.compile(r"Library/Application Support/(?:Google/Chrome|Firefox|Arc|BraveSoftware|Microsoft Edge)/"),
    re.compile(r"Library/Cookies/"),
    re.compile(r"Library/Safari/"),
]


# ---------------------------------------------------------------------------
# Environment dumps
# ---------------------------------------------------------------------------

ENV_DUMP_RES = [
    re.compile(r"\benv\b(?![\w/.\-])"),           # bare `env`
    re.compile(r"/usr/bin/env\b(?!\s+[A-Za-z_])"),
    re.compile(r"\bcommand\s+env\b"),
    re.compile(r"\bprintenv\b"),
    re.compile(r"\bexport\s+-p\b"),
    re.compile(r"\bset\s*(?:\||$)"),
    re.compile(r"\blaunchctl\s+dumpstate\b"),
    re.compile(r"\bsecurity\s+(?:dump-keychain|find-(?:generic|internet)-password)\b"),
]

# `$ANTHROPIC_API_KEY`, `${AWS_SECRET_ACCESS_KEY}`, etc. — direct dereference of
# a suspiciously-named env var.  Co-occurrence with a network-send indicates
# targeted exfil without a full `env` dump.
ENV_REF_RES = [
    re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*(?:_KEY|_TOKEN|_SECRET|_PASSWORD|_PASS|_PWD)\b"),
    re.compile(r"\$\{?(?:ANTHROPIC|OPENAI|AWS|GITHUB|GH|GOOGLE|GCP|AZURE|SLACK|STRIPE|DATABASE|DB|REDIS|MONGO|SMTP|JWT)_[A-Z0-9_]{2,}\b"),
]


# ---------------------------------------------------------------------------
# Outbound / network send commands
# ---------------------------------------------------------------------------

NETWORK_SEND_RES = [
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),
    re.compile(r"\bhttpie?\b"),
    re.compile(r"\bhttp\s+(?:post|put|patch)\b", re.IGNORECASE),
    re.compile(r"\bnc\b(?!\w)"),
    re.compile(r"\bncat\b"),
    re.compile(r"\bnetcat\b"),
    re.compile(r"\bbusybox\s+nc\b"),
    re.compile(r"\bsocat\b"),
    re.compile(r"\bscp\b"),
    re.compile(r"\brsync\b.*(?:::|@[^\s]+:|rsync://|/Volumes/)"),
    # Remote-shell exec (ssh + pipeline to file/shell) AND raw tunnel flags
    re.compile(r"\bssh\b\s+[^\s]+\s+['\"]?(?:cat|tee|sh|bash|python|nc)"),
    re.compile(r"\bssh\b[^\n]*\s-[RDLW]\b"),
    re.compile(r"\bautossh\b"),
    re.compile(r"\bsftp\b"),
    re.compile(r"\bftp\b"),
    re.compile(r"\baws\s+s3\s+(?:cp|sync|mv)\b"),
    re.compile(r"\baws\s+s3api\s+(?:put-object|copy-object|upload-part)\b"),
    re.compile(r"\bgsutil\s+(?:cp|rsync|mv)\b"),
    re.compile(r"\bgcloud\s+storage\s+(?:cp|rsync)\b"),
    re.compile(r"\baz\s+storage\s+(?:blob|file)\s+(?:upload|copy)\b"),
    # Allowed-egress dev tooling that can carry secrets off the machine.
    # These are dual-use; `pretool_guard` treats them as outbound only when
    # combined with a sensitive source or secret fingerprint.
    re.compile(r"\bgit\s+push\b"),
    re.compile(r"\bgh\s+(?:gist|pr|issue|release)\s+(?:create|edit)\b"),
    re.compile(r"\bglab\s+(?:snippet|mr|issue|release)\s+(?:create|edit)\b"),
    re.compile(r"\bhub\s+pull-request\b"),
    re.compile(r"\btwine\s+upload\b"),
    re.compile(r"\bnpm\s+publish\b"),
    re.compile(r"\bpip\s+(?:install|download)\s+[^#\n]*-i\b"),
    re.compile(r"\brclone\b"),
    re.compile(r"\bcroc\s+(?:send|s)\b"),
    re.compile(r"\bwormhole\s+send\b"),
    re.compile(r"\bmagic-wormhole\b"),
    # macOS clipboard + mail
    re.compile(r"\bpbcopy\b"),
    re.compile(r"\bmail\b\s+-s"),
    re.compile(r"\bmailx\b"),
    re.compile(r"\bsendmail\b"),
    re.compile(r"\bmsmtp\b"),
    re.compile(r"\bmutt\b\s+-s"),
    re.compile(r"\bopen\s+(?:mailto|https?|x-[\w\-]+|shortcuts)://"),
    re.compile(r"\bshortcuts\s+run\b"),
    # Side-channels
    re.compile(r"\bscreencapture\b"),
    re.compile(r"\bpbpaste\b"),
    # Raw TCP / UDP via bash /dev/tcp
    re.compile(r"/dev/(?:tcp|udp)/[A-Za-z0-9.\-]+/\d+"),
    # Generic interpreter one-liners that speak HTTP/TCP.  These are powerful —
    # rule fires only when co-occurring with sensitive source in pretool_guard.
    re.compile(r"\bpython[23]?\s+-c\b[^\n]*\b(?:urllib|http\.client|requests|socket|httpx|aiohttp)\b"),
    re.compile(r"\bperl\s+-[MeE][^\n]*\b(?:HTTP::Tiny|LWP::|IO::Socket::INET|Net::HTTP)\b"),
    re.compile(r"\bruby\s+-[eEr][^\n]*\b(?:open-uri|net/http|httparty|faraday)\b"),
    re.compile(r"\bnode\s+-e\b[^\n]*\b(?:https?|net|fetch\(|axios|got\()"),
    re.compile(r"\b(?:deno|bun)\s+(?:run|-e)\b[^\n]*\bfetch\("),
    re.compile(r"\bphp\s+-r\b[^\n]*\b(?:file_get_contents|fsockopen|curl_exec)\("),
    re.compile(r"\bosascript\b[^\n]*\bdo\s+shell\s+script\b"),
]

# DNS-ish exfil.  Threshold lowered from 40→20 so chunked leaks don't slip.
DNS_EXFIL_RES = [
    re.compile(r"\bdig\s+(?:@[\w.\-]+|[\w.\-]{20,})"),
    re.compile(r"\bhost\s+[\w.\-]{20,}"),
    re.compile(r"\bnslookup\s+[\w.\-]{20,}"),
]

# Persistence / autostart / background-execution surfaces.  A Write/Edit or
# Bash touching these is how a payload survives after the session ends.
PERSISTENCE_PATH_RES = [
    re.compile(r"Library/LaunchAgents/[^\s\"']+\.plist"),
    re.compile(r"Library/LaunchDaemons/[^\s\"']+\.plist"),
    re.compile(r"/etc/(?:cron\.d/|crontab)"),
    re.compile(r"(?:~|/Users/[^/\s\"']+|/home/[^/\s\"']+)/\.config/launchd/"),
    # shell rc files — match both `~/.zshrc` and `/Users/x/.zshrc`
    re.compile(r"(?:^|/)\.(?:zshrc|zshenv|zprofile|zlogin|bashrc|bash_profile|profile|bash_logout)\b"),
    re.compile(r"(?:^|/)\.config/fish/config\.fish\b"),
    re.compile(r"Library/LoginItems/"),
]

PERSISTENCE_CMD_RES = [
    re.compile(r"\bcrontab\s+-\b"),
    re.compile(r"\bcrontab\s+<"),
    re.compile(r"\bat\s+(?:now|-f|\d)"),
    re.compile(r"\blaunchctl\s+(?:load|bootstrap|submit|enable)\b"),
    re.compile(r"\bnohup\b"),
    re.compile(r"\bdisown\b"),
    re.compile(r"\bsetsid\b"),
    re.compile(r"\bscreen\s+-dm\b"),
    re.compile(r"\btmux\s+new\s+-d\b"),
    re.compile(r"\bcaffeinate\s+-[a-z]*s\b"),
]

# Cloud-sync folders (iCloud, Dropbox, OneDrive, Google Drive, Box, pCloud,
# Mega, Sync.com).  Writing a secret here exfiltrates via the sync client.
SYNC_DIR_RES = [
    re.compile(r"/Library/Mobile Documents/"),
    re.compile(r"/iCloud Drive/", re.IGNORECASE),
    re.compile(r"(?:/|^|~)Dropbox/", re.IGNORECASE),
    re.compile(r"(?:/|^|~)OneDrive(?:\s*-\s*[^/]+)?/", re.IGNORECASE),
    re.compile(r"(?:/|^|~)Google Drive/", re.IGNORECASE),
    re.compile(r"(?:/|^|~)Box(?:\s+Sync)?/", re.IGNORECASE),
    re.compile(r"(?:/|^|~)pCloud Drive/", re.IGNORECASE),
    re.compile(r"(?:/|^|~)MEGAsync/", re.IGNORECASE),
    re.compile(r"/Library/CloudStorage/"),
]

# Symlink-from-sensitive evasion: `ln -s ~/.ssh/id_rsa /tmp/k`.
SYMLINK_SENSITIVE_RES = re.compile(
    r"\bln\s+-s[A-Za-z]*\s+\S*(?:\.ssh/id_|\.aws/credentials|\.gnupg/|/Library/Keychains)"
)

# Known canary / paste / webhook / tunnel domains; these are *always*
# suspicious outbound targets — even a GET is enough to leak data via URL.
EXFIL_DOMAIN_SUBSTRINGS = [
    "requestbin.com", "requestbin.net", "requestcatcher.com",
    "pipedream.net", "webhook.site", "hookb.in",
    "ngrok.io", "ngrok-free.app", "ngrok.app",
    "oastify.com", "interact.sh", "burpcollaborator.net",
    "canarytokens.com", "beeceptor.com", "mocky.io",
    "termbin.com", "transfer.sh", "0x0.st", "paste.ee",
    "file.io", "tmpfiles.org", "gofile.io",
    "pastebin.com", "dpaste.com", "ix.io", "sprunge.us",
    "hastebin.com", "rentry.co",
    "discord.com/api/webhooks",
    "hooks.slack.com",
    "trycloudflare.com", "localtunnel.me", "serveo.net",
    "attacker.com",  # sentinel test
]

# Remote fetch → execute.  Split into two tiers:
#   *_SHELL  — ALWAYS block.  Piping downloaded content into a shell
#              interpreter treats every byte as code; no trusted domain
#              is worth that (CDN compromise / TLS MITM).
#   *_INTERP — block by default, but downgradable when the source host
#              is on the allowlist (see ALLOWLIST_HOSTS).  Covers
#              `curl … | python3 -m json.tool`, `| node -e "…"`, etc.
REMOTE_EXEC_SHELL_RES = [
    re.compile(r"(?:curl|wget|fetch|http)\s[^|;&\n]{0,400}\|\s*(?:sh|bash|zsh|ksh|dash|fish)\b"),
    re.compile(r"\beval\s*[\"'`]?\$?\(\s*(?:curl|wget|fetch)\b"),
    re.compile(r"\bsource\s*[\"'`]?<\(\s*(?:curl|wget)\b"),
    re.compile(r"\bbash\b[^\n]{0,400}<\(\s*(?:curl|wget)\b"),
]
REMOTE_EXEC_INTERP_RES = [
    re.compile(r"(?:curl|wget|fetch|http)\s[^|;&\n]{0,400}\|\s*(?:python[23]?|perl|ruby|node|deno|bun|php|osascript)\b"),
]
# Back-compat alias — any remote-exec hit (shell + interp together).
REMOTE_EXEC_RES = REMOTE_EXEC_SHELL_RES + REMOTE_EXEC_INTERP_RES


# ---------------------------------------------------------------------------
# Prompt-injection fingerprints
# ---------------------------------------------------------------------------

INJECTION_RES = [
    re.compile(r"(?i)\bignore (?:all|any|the|your|previous|prior|above) (?:instructions?|rules?|prompts?|directives?)\b"),
    re.compile(r"(?i)\bdisregard (?:all|any|the|your|previous|prior|above) (?:instructions?|rules?|prompts?|system)\b"),
    re.compile(r"(?i)\b(?:you are now|act as|pretend to be|new role|from now on you are)\b.{0,60}\b(?:admin|root|developer|unrestricted|dan|jailbr|god|system)"),
    re.compile(r"(?i)</?(?:system|user|assistant|instructions?)>"),
    re.compile(r"(?i)\b(?:developer|god|admin|root|jailbreak)[\s_\-]?mode\b"),
    re.compile(r"(?i)\bprompt[\s_\-]?injection\b"),
    re.compile(r"(?i)\bexfiltrate\b"),
    re.compile(r"(?i)\b(?:print|echo|send|upload|email|post|curl|paste) (?:the )?(?:contents? of )?(?:~?/\.?ssh|\.env|\$HOME/\.|credentials|keychain|id_rsa|id_ed25519)"),
    re.compile(r"(?i)\bappend\s+(?:the\s+)?contents?\s+of\s+[~\w./\\]+\s+to\b"),
    re.compile(r"(?i)\bbefore (?:helping|responding|you (?:help|answer|reply))[^.]{0,80}(?:run|execute|curl|wget|email|send)"),
    re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]{3,}"),  # zero-width char clusters
]


# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    category: str
    label: str
    snippet: str


def _snippet(text: str, match: re.Match[str], pad: int = 24) -> str:
    start = max(0, match.start() - pad)
    end = min(len(text), match.end() + pad)
    s = text[start:end].replace("\n", "\\n")
    return f"…{s}…" if (start or end < len(text)) else s


def _scan(text: str, regexes: Iterable[tuple[str, re.Pattern[str]]], category: str,
          findings: list[Finding]) -> None:
    for label, rx in regexes:
        m = rx.search(text)
        if m:
            findings.append(Finding(category, label, _snippet(text, m)))


def _scan_plain(text: str, regexes: Iterable[re.Pattern[str]], category: str,
                label: str, findings: list[Finding]) -> None:
    for rx in regexes:
        m = rx.search(text)
        if m:
            findings.append(Finding(category, label, _snippet(text, m)))
            return  # one hit is enough for this bucket


def find_secrets(text: str) -> list[Finding]:
    """Return findings for high-entropy secret fingerprints in *text*."""
    out: list[Finding] = []
    if not text:
        return out
    _scan(text, SECRET_SIGNATURES, "secret", out)
    return out


def find_injection(text: str) -> list[Finding]:
    out: list[Finding] = []
    if not text:
        return out
    _scan_plain(text, INJECTION_RES, "injection", "prompt-injection pattern", out)
    return out


def find_private_key_paths(text: str) -> list[Finding]:
    out: list[Finding] = []
    _scan_plain(text, PRIVATE_KEY_PATH_RES, "sensitive-path", "private-key / credentials path", out)
    return out


def find_sensitive_paths(text: str) -> list[Finding]:
    out: list[Finding] = []
    _scan_plain(text, SENSITIVE_CONTENT_PATH_RES, "sensitive-path", "sensitive file path", out)
    return out


def find_network_send(text: str) -> list[Finding]:
    out: list[Finding] = []
    _scan_plain(text, NETWORK_SEND_RES, "network-send", "outbound network command", out)
    for sub in EXFIL_DOMAIN_SUBSTRINGS:
        if sub in text:
            out.append(Finding("network-send", f"paste/webhook/tunnel domain ({sub})", sub))
            break
    _scan_plain(text, DNS_EXFIL_RES, "network-send", "dns-exfil pattern", out)
    return out


def find_env_dump(text: str) -> list[Finding]:
    out: list[Finding] = []
    _scan_plain(text, ENV_DUMP_RES, "env-dump", "environment / keychain dump", out)
    return out


def find_remote_exec(text: str) -> list[Finding]:
    out: list[Finding] = []
    _scan_plain(text, REMOTE_EXEC_RES, "remote-exec", "curl|sh-style remote execution", out)
    return out


def find_remote_exec_shell(text: str) -> list[Finding]:
    out: list[Finding] = []
    _scan_plain(text, REMOTE_EXEC_SHELL_RES, "remote-exec", "curl|sh-style remote execution", out)
    return out


def find_remote_exec_interp(text: str) -> list[Finding]:
    out: list[Finding] = []
    _scan_plain(text, REMOTE_EXEC_INTERP_RES, "remote-exec", "curl|interpreter pipeline", out)
    return out


# ---------------------------------------------------------------------------
# Trusted-host allowlist (loaded from ~/.claude/hooks/security/allowlist.txt)
# ---------------------------------------------------------------------------

_URL_HOST_RE = re.compile(r"https?://([A-Za-z0-9._\-]+)")


def _load_allowlist() -> list[str]:
    """Read allowlist.txt once at import; tolerate missing file.

    Looks first next to this file (plugin-relative), then falls back to the
    legacy ~/.claude/hooks/security/ path so unpacked copies still work."""
    import os, pathlib
    candidates = [
        pathlib.Path(__file__).resolve().parent / "allowlist.txt",
        pathlib.Path(os.path.expanduser("~/.claude/hooks/security/allowlist.txt")),
    ]
    p = next((c for c in candidates if c.exists()), candidates[0])
    try:
        raw = p.read_text()
    except Exception:
        return []
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.lower())
    return out


ALLOWLIST_HOSTS: list[str] = _load_allowlist()


def _host_matches(host: str, pattern: str) -> bool:
    host = host.lower()
    if pattern.startswith("*."):
        suffix = pattern[1:]  # ".googleapis.com"
        return host.endswith(suffix) or host == suffix[1:]
    return host == pattern


def extract_hosts(text: str) -> list[str]:
    """Return every https?://host found in *text* (case-preserved, deduped)."""
    if not text:
        return []
    seen: dict[str, None] = {}
    for m in _URL_HOST_RE.finditer(text):
        seen.setdefault(m.group(1), None)
    return list(seen)


def all_hosts_allowlisted(text: str) -> bool:
    """True iff at least one URL is present AND every URL host is allow-listed."""
    hosts = extract_hosts(text)
    if not hosts:
        return False
    if not ALLOWLIST_HOSTS:
        return False
    for h in hosts:
        if not any(_host_matches(h, p) for p in ALLOWLIST_HOSTS):
            return False
    return True


def redact_secrets(text: str) -> tuple[str, list[Finding]]:
    """Return (redacted_text, findings) — replace matched secrets with ▇▇▇REDACTED▇▇▇."""
    out: list[Finding] = []
    if not text:
        return text, out
    redacted = text
    for label, rx in SECRET_SIGNATURES:
        def _sub(m: re.Match[str], _label: str = label) -> str:
            out.append(Finding("secret", _label, _snippet(text, m)))
            return "▇▇▇REDACTED-" + _label.upper().replace(" ", "-") + "▇▇▇"
        redacted = rx.sub(_sub, redacted)
    return redacted, out

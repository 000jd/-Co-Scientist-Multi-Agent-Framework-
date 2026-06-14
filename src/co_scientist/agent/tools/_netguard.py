"""SSRF guard shared by web/computer-use tools.

Model- or injection-supplied URLs must never be allowed to reach loopback,
private, link-local, reserved, or multicast ranges, nor the cloud metadata
endpoint (169.254.169.254). The hostname is resolved and EVERY resolved IP is
checked, so a public hostname that resolves to an internal address is also
blocked (DNS-rebinding style).
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse

_METADATA_IP = "169.254.169.254"


def is_blocked_url(url: str) -> Optional[str]:
    """Return a human-readable reason if ``url`` must be blocked, else ``None``."""
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return "url is not parseable"

    if parsed.scheme not in ("http", "https"):
        return f"scheme {parsed.scheme!r} not allowed (http/https only)"

    host = parsed.hostname
    if not host:
        return "url has no host"

    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except Exception as e:  # noqa: BLE001 — DNS failure is treated as blocked
        return f"could not resolve host {host!r}: {e}"

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return f"unresolvable address for host {host!r}"
        if str(ip) == _METADATA_IP:
            return "blocked cloud metadata endpoint 169.254.169.254"
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return f"host {host!r} resolves to blocked address {ip}"

    return None

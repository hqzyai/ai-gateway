import asyncio
import ipaddress
import os
import socket
import ssl
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend
from httpcore._backends.base import SOCKET_OPTION


def enabled() -> bool:
    return os.getenv("AGENTOS_MCP_EGRESS_ENABLED", "false").lower() == "true"


def validate_addresses(host: str, port: int, addresses: tuple[str, ...], allowed: frozenset[str]) -> None:
    if not addresses:
        raise ValueError("mcp_url_not_allowed")
    for raw in addresses:
        parsed = ipaddress.ip_address(raw)
        address = parsed.ipv4_mapped or parsed if isinstance(parsed, ipaddress.IPv6Address) else parsed
        if (
            address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            raise ValueError("mcp_url_not_allowed")
        if isinstance(address, ipaddress.IPv6Address) and (address.sixtofour or address.teredo):
            raise ValueError("mcp_url_not_allowed")
        private_allowed = f"{host.lower().rstrip('.')}:{port}" in allowed and any(
            address in ipaddress.ip_network(network)
            for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
        )
        if not address.is_global and not private_allowed:
            raise ValueError("mcp_url_not_allowed")


def validate_origin(url: str, allowed: frozenset[str]) -> tuple[str, int]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("mcp_url_not_allowed")
    host, port = parsed.hostname.lower().rstrip("."), parsed.port or 443
    if host in ("localhost", "metadata.google.internal") or parsed.port == 0:
        raise ValueError("mcp_url_not_allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host, port
    validate_addresses(host, port, (str(address),), allowed)
    return host, port


class PinnedBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, allowed: frozenset[str], backend: httpcore.AsyncNetworkBackend | None = None) -> None:
        self.allowed = allowed
        self.backend = backend or AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        async with asyncio.timeout(timeout if timeout is not None else 30):
            records = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
            addresses = tuple(sorted(frozenset(str(record[4][0]) for record in records)))
            validate_addresses(host, port, addresses, self.allowed)
            return await self.backend.connect_tcp(addresses[0], port, timeout, local_address, socket_options)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise ValueError("MCP Unix sockets are not permitted")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class PinnedTransport(httpx.AsyncHTTPTransport):
    def __init__(self, allowed: frozenset[str], verify: ssl.SSLContext) -> None:
        super().__init__(verify=verify, trust_env=False, retries=0)
        self._pool._network_backend = PinnedBackend(allowed)  # pyright: ignore[reportPrivateUsage]  # HTTPX exposes no public backend injection while retaining its TLS and response adapters.


def client(
    url: str,
    *,
    headers: httpx.Headers | Mapping[str, str] | None = None,
    timeout: httpx.Timeout | float | None = None,
    auth: httpx.Auth | None = None,
    verify: ssl.SSLContext | bool | str = True,
) -> httpx.AsyncClient:
    allowed = frozenset(
        part.strip().lower() for part in os.getenv("AGENTOS_MCP_ALLOWED_PRIVATE_ORIGINS", "").split(",") if part.strip()
    )
    origin = validate_origin(url, allowed)
    if verify is False or (
        isinstance(verify, ssl.SSLContext) and (verify.verify_mode != ssl.CERT_REQUIRED or not verify.check_hostname)
    ):
        raise ValueError("MCP requires TLS certificate verification")
    context = (
        verify
        if isinstance(verify, ssl.SSLContext)
        else ssl.create_default_context(cafile=verify if isinstance(verify, str) else os.getenv("SSL_CERT_FILE"))
    )

    async def check_request(request: httpx.Request) -> None:
        if validate_origin(str(request.url.copy_with(query=None, fragment=None)), allowed) != origin:
            raise ValueError("MCP cross-origin request rejected")

    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout if timeout is not None else 30,
        auth=auth,
        transport=PinnedTransport(allowed, context),
        trust_env=False,
        follow_redirects=False,
        event_hooks={
            "request": [check_request]
        },  # mutable-ok: HTTPX copies and exposes hook lists through its public mutable event_hooks API.
    )

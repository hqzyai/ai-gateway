import ssl
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from litellm.experimental_mcp_client.egress import PinnedBackend, client, validate_addresses, validate_origin


@pytest.mark.parametrize("address", ("127.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "0.0.0.0", "224.0.0.1"))
def test_private_allowlist_never_allows_local_or_metadata(address):
    with pytest.raises(ValueError):
        validate_addresses("internal.example", 443, (address,), frozenset(("internal.example:443",)))


def test_private_network_requires_exact_origin():
    with pytest.raises(ValueError):
        validate_addresses("internal.example", 443, ("10.1.2.3",), frozenset())
    validate_addresses("internal.example", 443, ("10.1.2.3",), frozenset(("internal.example:443",)))
    with pytest.raises(ValueError):
        validate_addresses("internal.example", 444, ("10.1.2.3",), frozenset(("internal.example:443",)))


def test_mixed_dns_answers_fail_closed():
    with pytest.raises(ValueError):
        validate_addresses("external.example", 443, ("8.8.8.8", "127.0.0.1"), frozenset())


@pytest.mark.asyncio
@pytest.mark.parametrize("addresses,allowed", [(("8.8.8.8",), True), (("8.8.8.8", "127.0.0.1"), False)])
async def test_backend_dials_only_validated_dns_result(monkeypatch, addresses, allowed):
    backend = AsyncMock()
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443)) for address in addresses
    ])
    transport = PinnedBackend(frozenset(), backend=backend)
    if allowed:
        result = await transport.connect_tcp("mcp.example", 443, timeout=2)
        assert result is backend.connect_tcp.return_value
        backend.connect_tcp.assert_awaited_once_with("8.8.8.8", 443, 2, None, None)
    else:
        with pytest.raises(ValueError, match="mcp_url_not_allowed"):
            await transport.connect_tcp("mcp.example", 443, timeout=2)
        backend.connect_tcp.assert_not_awaited()


@pytest.mark.parametrize("url", ("http://example.com", "https://user:secret@example.com", "https://example.com?token=1", "https://localhost", "https://example.com:0"))
def test_invalid_origin_rejected(url):
    with pytest.raises(ValueError):
        validate_origin(url, frozenset())


def test_tls_verification_cannot_be_disabled():
    with pytest.raises(ValueError):
        client("https://example.com", verify=False)
    context = ssl.create_default_context()
    context.check_hostname = False
    with pytest.raises(ValueError):
        client("https://example.com", verify=context)


@pytest.mark.asyncio
async def test_session_query_allowed_but_cross_origin_rejected():
    async with client("https://example.com/mcp") as session:
        assert not session.follow_redirects
        hook = session.event_hooks["request"][0]
        await hook(httpx.Request("POST", "https://example.com/messages?session=1"))
        with pytest.raises(ValueError, match="cross-origin"):
            await hook(httpx.Request("POST", "https://other.example/messages"))

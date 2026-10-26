# LiteLLM MCP Client

LiteLLM MCP Client is a client that allows you to use MCP tools with LiteLLM.

## Restricted outbound connections

Set `AGENTOS_MCP_EGRESS_ENABLED=true` to enable HTTPS-only, DNS-pinned connections for MCP HTTP/SSE clients and client-credentials token requests. This is disabled by default to preserve existing installations

`AGENTOS_MCP_ALLOWED_PRIVATE_ORIGINS` accepts comma-separated exact `host:port` entries, for example `mcp.internal.example:443`. These entries permit private network addresses, but never loopback, link-local, multicast or reserved addresses. Keep this allowlist consistent with the control plane

Restricted connections require certificate and hostname verification, ignore environment proxies, reject cross-origin requests and do not follow redirects. Set `SSL_CERT_FILE` to a trusted CA bundle when required. Session query parameters on same-origin MCP requests remain supported



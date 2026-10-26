# AgentOS native gateway integration

These modules implement the AgentOS gateway behavior in the gateway source tree
and use the existing LiteLLM callback, router, pricing and management interfaces

## Configuration

Merge `litellm/proxy/example_config_yaml/agentos_native_callbacks.yaml` into the
gateway configuration. Do not replace the existing configuration or remove its
model definitions, compression callback, Prometheus callback or other integrations

The native callback import paths are

```yaml
litellm_settings:
  callbacks:
    - litellm.integrations.agentos.identity.agentos_identity_callback
    - litellm.integrations.agentos.live_counters.agentos_live_counters_callback
    - litellm.integrations.agentos.usage_callback.agentos_usage_callback
```

Set `AGENTOS_SPEND_ATTRIBUTION_ENABLED=true` to store the public model group,
catalog provider and routing hops in spend logs. Without this flag the original
spend-log assembly stays unchanged. This changes attribution, not the calculated
spend or token counts

Set `AGENTOS_GUARDRAIL_MODE=true`, `AGENTOS_MCP_EGRESS_ENABLED=true` and
`AGENTOS_ROUTING_OBSERVATIONS_ENABLED=true` for the managed guardrail behavior,
validated MCP transport and decision recording. The decision endpoint returns at
most 100 records. MCP network configuration is documented in
`litellm/experimental_mcp_client/Readme.md`

Usage delivery requires `AGENTOS_BFF_INTERNAL_TOKEN` and a persistent writable
`AGENTOS_USAGE_OUTBOX_PATH`. Configure `AGENTOS_USAGE_CALLBACK_URL` and
`AGENTOS_USAGE_PREFLIGHT_URL` for the BFF internal endpoints; their defaults use
`http://agentos-server:8080/internal/ai-gateway/usage` and `/internal/ai-gateway/preflight`
respectively. Do not expose the internal token in logs or client configuration

The queue preserves the original SQLite schema, signed attribution format,
attempt identity, retry schedule and dead-letter behavior. The default flush
interval is 5 seconds, configured with `AGENTOS_USAGE_OUTBOX_FLUSH_SECONDS`

```bash
python -m litellm.integrations.agentos.usage_outbox_admin --database /data/usage-outbox.sqlite3 status
python -m litellm.integrations.agentos.usage_outbox_admin --database /data/usage-outbox.sqlite3 requeue --call-id '<call-id>'
```

## Native entry points

`POST /agentos/usage/query`, `POST /agentos/usage/daily`,
`GET /agentos/routing-decisions` and `PUT /model/cost_map/overrides/import`
require the gateway master key. The BFF supplies the authorized user scope;
these are not browser-facing APIs

Pricing import uses the existing configuration repository and cluster reload
mechanism, rolling back persisted overrides and the local cost map if reload
fails. Price replacement restores served deployment registrations. DashScope
image generation retains catalog prices, honors deployment-specific prices and
falls back to the actual returned image count when upstream usage omits it

Department labels use `metadata.display_name` while permission identifiers stay
unchanged. Guardrail monitoring classifies masking separately and retains the
original bounded action-filter scan and first-page blocked-record prioritization

The source branch already implements BM25 scoring in Python, so the old overlay's
`rank-bm25` installation is not required. No deployment-time source replacement
scripts or generated dashboard bundle edits are needed for this integration

## Regression checks

The tests under `tests/test_litellm/integrations/agentos/` cover attribution,
identity, live counters and durable delivery. The management endpoint tests
include a snapshot of 88 original SQL/parameter combinations. PostgreSQL query
execution tests additionally require an isolated `AGENTOS_USAGE_TEST_DATABASE_URL`;
they use a temporary shadow table and roll back their transaction

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

## Managed content rules

The `generic_api_key` preset recognizes Chinese API/access credential labels and
case-insensitive English API key/access key/access token labels. Values contain
at least 8 characters with digits or credential symbols, or at least 20 letters.
Hyphens, underscores, dotted tokens and base64-style symbols are supported.
These are lexical detection criteria, not verification that a credential is valid.
Shorter or differently formatted secrets require a custom pattern

In managed post-call filtering, a credential identified in request messages is
also protected when the response repeats its exact value without the label.
BLOCK and MASK apply to response text and reasoning. Streams using this preset
are buffered until completion to avoid releasing a prefix of a long credential;
this trades first-content latency for safe redaction. Unlabeled arbitrary strings
without credential context are not automatically classified as API keys

Managed-mode topic dictionaries originate from the 17 Vocabulary files from
[Sensitive-lexicon](https://github.com/konsheng/Sensitive-lexicon), pinned to
`d967c30b053fa40b06c5a0dddf0be493f2dfae46`. The unmodified text files and MIT
license are bundled under `litellm_content_filter/lexicons`; no network is used
during loading or filtering. Blank lines and duplicates within a file are
removed. Reviewed neutral standalone entries are excluded across all dictionaries
using `lexicons/excluded_terms.txt`; internal spaces and punctuation are retained.
Empty curated dictionaries are omitted from new selections but their IDs remain
readable. Upstream originals are preserved; preview includes a curation revision

Curation revision `2026-10-27-r2` also removes standalone numbers, formatted
contact numbers and legacy regex/template or replacement-rule fragments across
all 17 dictionaries. `exclusion_reason()` provides a reason for each omission.
These checks operate on whole entries: numeric risk phrases and domains remain.
Concrete historical phone numbers do not replace personal-information format
rules. This structural pass does not certify the meaning of every retained term

Each `lexicon_*` category is a literal, case-insensitive keyword list. There are
no subject/action combinations, regex interpretation, exceptions or user-facing
severity levels. Preview YAML and runtime use the same curated terms. A selected
dictionary applies all its words; enablement and BLOCK/MASK remain configurable.
The high severity in native detection records is an API compatibility value,
not an upstream risk assessment. All matches are masked, overlapping spans merge,
and BLOCK wins over MASK across selected dictionaries. Streams buffer content
and reasoning until detection completes to prevent partial-keyword leakage;
this delays first-content delivery. Keyword patterns are cached by dictionary
and indexed by initial character rather than recompiled per request

The initial curation removes common places, civic terms and everyday vocabulary,
not entire messages containing those words. Longer explicit risk phrases remain.
Remaining terms still require business review; selection is opt-in and never
automatically enables all dictionaries. Quotation, negation and paraphrases are
not interpreted semantically. Review the actual terms against business samples

Existing non-lexicon categories retain their Chinese rules from
`chinese_categories.py` and remain readable/editable, but no longer appear in
the managed new-category catalog. There is no automatic semantic mapping or
migration between old categories and upstream files. Users can explicitly
remove an old category and choose a new dictionary. Custom files and non-managed
deployments keep their existing behavior

### Deferred tool permissions

In managed mode, post-call tool permission checks evaluate both the outer
`tool_call` dispatch and its targets using the same ordered rules. Both the
legacy `name`/`arguments` envelope and the `calls` array are supported. Every
batch entry is checked; one denied or malformed entry rejects the whole dispatch.
Empty batches and mixed legacy/batch envelopes are rejected. Allowing the bridge
does not allow every target; allowing a target does
not override a denial of the bridge. Inner argument patterns use paths relative
to the actual tool arguments, not the dispatch envelope. Object arguments and
JSON-encoded objects are supported; malformed envelopes and recursive bridge
dispatch are denied. Ordinary tools are not unwrapped just because their
arguments include a `name` field

Both normal and buffered streaming responses are checked before releasing a
tool call. Rewrites preserve correlation with the original dispatch call ID.
Managed rewrites use Chinese explanations and identify the denied target while
preserving the original diagnostic reason internally.
Pre-call checks still inspect only the model-facing tool catalog; configure
post-call enforcement to protect deferred targets proposed by the model.
Patterns still match the full tool name (for example
`mcp__guardrail_mock__drop_database`), with no implicit MCP prefix removal.
Non-managed deployments retain the original behavior

## Regression checks

The tests under `tests/test_litellm/integrations/agentos/` cover attribution,
identity, live counters and durable delivery. The management endpoint tests
include a snapshot of 88 original SQL/parameter combinations. PostgreSQL query
execution tests additionally require an isolated `AGENTOS_USAGE_TEST_DATABASE_URL`;
they use a temporary shadow table and roll back their transaction

"""
Headroom Interception Module

Server-side, transparent prompt compression + retrieval via the external
Headroom compression service, registered as a plain litellm_settings.callbacks
entry (no guardrails: config required).
"""

from litellm.integrations.headroom_interception.handler import HeadroomInterceptionLogger

__all__ = [
    "HeadroomInterceptionLogger",
]

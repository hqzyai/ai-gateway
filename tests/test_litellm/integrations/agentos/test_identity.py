from litellm.integrations.agentos import identity as callback


def test_native_callback_example_loads_without_patch_modules():
    from pathlib import Path

    import yaml

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.proxy.types_utils.utils import get_instance_fn
    from litellm.proxy import proxy_server

    path = Path(proxy_server.__file__).parent / "example_config_yaml" / "agentos_native_callbacks.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["litellm_settings"]["timezone"] == "Asia/Shanghai"
    for module_path in config["litellm_settings"]["callbacks"]:
        assert module_path.startswith("litellm.integrations.agentos.")
        assert isinstance(get_instance_fn(module_path, str(path)), CustomLogger)
    assert set(config["environment_variables"].values()) == {"true"}


def test_gateway_identity_callback_prepends_product_identity_once() -> None:
    request = {"messages": [{"role": "user", "content": "你是谁？"}]}

    first = callback.apply_agentos_identity(request)
    second = callback.apply_agentos_identity(first)

    assert first is request
    assert second is request
    assert len(request["messages"]) == 2
    assert request["messages"][0]["role"] == "system"
    assert "AgentOS" in request["messages"][0]["content"]
    assert "underlying model provider" in request["messages"][0]["content"]


def test_gateway_identity_callback_merges_existing_system_message() -> None:
    request = {
        "messages": [
            {"role": "system", "content": "Hermes runtime instructions."},
            {"role": "user", "content": "Reply with the marker."},
        ]
    }

    callback.apply_agentos_identity(request)
    callback.apply_agentos_identity(request)

    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert request["messages"][0]["content"].count(callback.AGENTOS_IDENTITY_MESSAGE) == 1
    assert request["messages"][0]["content"].endswith("Hermes runtime instructions.")


def test_gateway_identity_callback_merges_structured_system_message() -> None:
    request = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Runtime context."}]},
            {"role": "user", "content": "Reply with the marker."},
        ]
    }

    callback.apply_agentos_identity(request)
    callback.apply_agentos_identity(request)

    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert request["messages"][0]["content"][0] == {
        "type": "text",
        "text": callback.AGENTOS_IDENTITY_MESSAGE,
    }
    assert len(request["messages"][0]["content"]) == 2


def test_gateway_identity_callback_persists_sanitized_request_snapshot() -> None:
    request = {
        "model": "meituan/longcat-2.0",
        "messages": [{"role": "user", "content": "request-marker"}],
        "stream": False,
        "api_key": "must-not-be-logged",
        "provider_api_key": "must-not-be-logged",
        "extra_body": {
            "trace": "trace-marker",
            "authorization": "must-not-be-logged",
            "token": "must-not-be-logged",
            "secret": "must-not-be-logged",
            "secret_key": "must-not-be-logged",
            "provider_api_key": "must-not-be-logged",
            "providerApiKey": "must-not-be-logged",
            "nested": {
                "oauth_token": "must-not-be-logged",
                "model_key": "model-key-is-an-identifier",
                "token_id": "token-hash-is-an-identifier",
                "safe": True,
            },
        },
        "metadata": {"existing": "kept"},
    }

    result = callback.prepare_agentos_request(request)

    snapshot = result["metadata"]["agentos_request"]
    assert result["metadata"]["existing"] == "kept"
    assert snapshot["model"] == "meituan/longcat-2.0"
    assert snapshot["messages"][-1]["content"] == "request-marker"
    assert snapshot["extra_body"]["trace"] == "trace-marker"
    assert snapshot["extra_body"]["authorization"] == "[REDACTED]"
    for key in ("token", "secret", "secret_key", "provider_api_key", "providerApiKey"):
        assert snapshot["extra_body"][key] == "[REDACTED]"
    assert snapshot["extra_body"]["nested"] == {
        "oauth_token": "[REDACTED]",
        "model_key": "model-key-is-an-identifier",
        "token_id": "token-hash-is-an-identifier",
        "safe": True,
    }
    assert "api_key" not in snapshot
    assert "provider_api_key" not in snapshot

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy.management_endpoints import routing_decision_endpoints as endpoints
from litellm.router_strategy.decision_observations import Decision


@pytest.mark.asyncio
async def test_observation_endpoint_requires_master_and_filters_config(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "test-master")
    first = Decision(model_group="group", strategy="least-busy", config_version=1, selected_deployment_id="a", candidates=())
    second = first.model_copy(update={"config_version": 2})
    storage = Mock()
    storage.lrange.return_value = [second.model_dump_json(), first.model_dump_json()]
    monkeypatch.setattr(endpoints, "recorder", SimpleNamespace(client=storage))
    app = FastAPI()
    app.include_router(endpoints.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for headers in ({}, {"Authorization": "Bearer regular-key"}):
            response = await client.get("/agentos/routing-decisions?model_group=group", headers=headers)
            assert response.status_code == 403
        storage.lrange.assert_not_called()
        headers = {"Authorization": "Bearer test-master"}
        response = await client.get("/agentos/routing-decisions?model_group=group&config_version=2", headers=headers)
        assert response.status_code == 200
        assert [row["config_version"] for row in response.json()["decisions"]] == [2]
        storage.lrange.assert_called_once_with("agentos:routing-decisions:group", 0, 99)
        response = await client.get("/agentos/routing-decisions?model_group=group", headers=headers)
        assert [row["config_version"] for row in response.json()["decisions"]] == [1, 2]
        storage.lrange.side_effect = ConnectionError("offline")
        response = await client.get("/agentos/routing-decisions?model_group=group", headers=headers)
        assert response.json() == {"decisions": [], "available": False}

from __future__ import annotations

from services.langflow_client import _extract_flow_payload


def test_extract_flow_payload_direct_shape() -> None:
    payload = {
        "data": {
            "nodes": [{"id": "n1"}],
            "edges": [{"source": "n1", "target": "n1"}],
        }
    }
    result = _extract_flow_payload(payload)
    assert "data" in result
    assert "nodes" in result["data"]


def test_extract_flow_payload_nested_shape() -> None:
    payload = {
        "data": {
            "data": {
                "nodes": [{"id": "n1"}],
                "edges": [{"source": "n1", "target": "n1"}],
            }
        }
    }
    result = _extract_flow_payload(payload)
    assert result["data"]["nodes"][0]["id"] == "n1"


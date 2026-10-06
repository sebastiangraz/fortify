import base64

import pytest
import torch

from fortify.imageio import decode_delta, encode_delta, load_png, save_png


def test_delta_png_roundtrip():
    levels = torch.randint(-12, 13, (1, 3, 20, 30)).float()
    delta = levels / 255
    assert torch.allclose(decode_delta(encode_delta(delta)), delta, atol=1e-6)


def test_png_roundtrip():
    x = torch.randint(0, 256, (1, 3, 9, 11)).float() / 255
    assert torch.allclose(load_png(save_png(x)), x, atol=1e-6)


def test_service_contract(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("FORTIFY_TOKEN", "t")
    from fastapi.testclient import TestClient

    from fortify import service

    monkeypatch.setattr(service, "SURROGATES", "toy")
    monkeypatch.setattr(service, "_ensemble", None)
    client = TestClient(service.app)
    crop = base64.b64encode(save_png(torch.rand((1, 3, 40, 60)))).decode()
    body = {
        "strength": "low",
        "groups": [
            {"id": "s0", "logo": {"x": 10, "y": 8, "w": 40, "h": 24}, "frames": [crop, crop]}
        ],
    }

    assert client.post("/v1/vaccinate", json=body).status_code == 401
    res = client.post("/v1/vaccinate", json=body, headers={"Authorization": "Bearer t"})
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["offset"] == 128 and out["groups"][0]["id"] == "s0" and out["groups"][0]["eps"] == 4
    delta = decode_delta(base64.b64decode(out["groups"][0]["delta"]))
    assert delta.shape == (1, 3, 40, 60) and float(delta.abs().max()) <= 4 / 255 + 1e-6

    body["groups"][0]["logo"]["w"] = 999
    assert (
        client.post("/v1/vaccinate", json=body, headers={"Authorization": "Bearer t"}).status_code
        == 400
    )

    body["groups"][0]["logo"]["w"] = 40
    body["groups"][0]["view"] = {"frame": {"w": 320, "h": 180}, "at": {"x": 200, "y": 100}}
    ok = client.post("/v1/vaccinate", json=body, headers={"Authorization": "Bearer t"})
    assert ok.status_code == 200, ok.text
    body["groups"][0]["view"]["at"]["x"] = 300  # crop would stick out of the frame
    assert (
        client.post("/v1/vaccinate", json=body, headers={"Authorization": "Bearer t"}).status_code
        == 400
    )

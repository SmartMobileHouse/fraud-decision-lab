"""Test-wide safety: no test may reach the network unless marked `live`."""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return

    def guard(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)

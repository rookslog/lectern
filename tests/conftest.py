"""Default-deny network access for the unit suite.

Lectern's privacy posture says unit tests do not touch the network. Until now
that was enforced by two tests that patched `socket.socket` for their own
duration -- which proves those two paths are clean and says nothing about the
rest of the suite. A guarantee that holds only where someone remembered to
assert it is a guarantee about the assertions, not about the program.

So the block is applied to every test automatically, and the `integration`
marker is the only way out. `pyproject.toml` already deselects that marker from
`make verify`, so the default path is both network-free and enforced rather than
network-free and hoped for.

The block is a fixture rather than an import-time patch on purpose: patching at
collection would also cover module imports, where a library opening a socket
would fail the run in a way that has nothing to do with what any test does.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any, cast

import pytest

INTEGRATION_MARKER = "integration"


class NetworkAccessDenied(AssertionError):
    """Raised when an unmarked test attempts to open a socket.

    An `AssertionError` rather than a custom exception type, so it reads as a
    test failure -- which is what it is -- instead of as a crash in the code
    under test.
    """


def may_use_network(request: pytest.FixtureRequest) -> bool:
    """Whether this test is allowed to open sockets.

    Named as a predicate and kept trivially small because it is the whole
    policy: exactly one marker opens the door, and everything else is denied.
    """

    node = cast(pytest.Item, cast(Any, request).node)
    return node.get_closest_marker(INTEGRATION_MARKER) is not None


@pytest.fixture(autouse=True)
def deny_network_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if may_use_network(request):
        yield
        return

    def refuse(*args: Any, **kwargs: Any) -> socket.socket:
        raise NetworkAccessDenied(
            "this test attempted to open a network socket; unit tests are network-free. "
            f"Mark it with @pytest.mark.{INTEGRATION_MARKER} if it genuinely needs the network."
        )

    # `create_connection` is patched alongside the constructor because it is a
    # separate entry point into the same capability, and a guard that covers one
    # door is the failure this fixture replaces.
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


@pytest.fixture(autouse=True)
def deny_real_asr_models(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never construct a real model, even when the ASR extra is installed.

    Network denial alone would still allow cached weights to run. Acceptance
    tests may replace this unavailable factory with a synthetic backend.
    """

    if may_use_network(request):
        return

    def unavailable(model: str, *, allow_model_download: bool = False) -> None:
        raise ImportError("real ASR models are disabled in unit tests; inject a fake backend")

    monkeypatch.setattr("lectern.asr.create_asr_backend", unavailable)

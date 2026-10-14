"""Structural pins for the automation package's layering.

These assert boundaries rather than behavior. They exist because the costs the
layering buys — a state store that carries no transport, an import surface
consumers can rely on, and an ingest pipeline that sits above the store rather
than inside it — are all invisible to a behavioral suite, and so are exactly the
kind of property a later change can dissolve without any test noticing.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from _pytest.monkeypatch import MonkeyPatch

from lectern import automation, provenance, records, state
from lectern.automation import (
    AutomationError,
    LocalFolderAdapter,
    SourceKind,
    SourcePolicy,
    SourceRecord,
    YouTubePlaylistAdapter,
    default_source_adapter,
)
from lectern.sources import local, youtube

PACKAGE = Path(automation.__file__).parent


def module_source(module: object) -> str:
    return Path(inspect.getfile(module)).read_text(encoding="utf-8")  # type: ignore[arg-type]


def imported_modules(module: object) -> set[str]:
    tree = ast.parse(module_source(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.add(node.module)
    return names


def defined_classes(module: object) -> set[str]:
    tree = ast.parse(module_source(module))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def test_records_is_a_leaf_of_the_package() -> None:
    # Everything else in the spine depends on the vocabulary, so the vocabulary
    # may depend on nothing here: that is what keeps the graph acyclic and lets
    # the store, the adapters, and the provenance writer stay independent.
    assert {name for name in imported_modules(records) if name.startswith("lectern")} == set()


def test_state_store_defines_no_transport_or_adapter_of_its_own() -> None:
    """What the store's *own source* contains, which is a weaker claim.

    Kept alongside the import test below rather than replaced by it, because the
    two pin different things: that test measures what loading the store costs,
    while this one catches an adapter or transport being written directly into
    the persistence layer — which no `sys.modules` check would notice, since a
    hand-rolled client here adds no new import edge. On its own this assertion
    is not enough: it greps for a symptom, and a store that merely *imports* the
    transport passes it, which is exactly the gap review found.
    """

    source = module_source(state)

    assert "urllib" not in source
    assert not [name for name in defined_classes(state) if name.endswith("Adapter")]


def test_only_explicit_provider_and_model_boundaries_open_the_network() -> None:
    modules_with_transport = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "urllib.request" in path.read_text(encoding="utf-8")
    }

    # S8 adds an explicit-per-invocation, pinned model fetch boundary. Media
    # processing and the state store still must not implement transport.
    assert modules_with_transport == {"sources/youtube.py", "diarization.py"}


def test_orchestration_sits_above_the_store() -> None:
    assert issubclass(automation.AutomationState, state.AutomationStateStore)

    # The pipeline is defined in the composition root, not in the store: this is
    # what makes `attach_provenance_to_bundle` resolvable through
    # `lectern.automation`'s globals, which the ingest-rollback tests rely on.
    assert automation.AutomationState.ingest_queue_item.__module__ == "lectern.automation"
    assert automation.AutomationState.ingest_one_shot.__module__ == "lectern.automation"
    assert state.AutomationStateStore.scan_source.__module__ == "lectern.state"

    assert "attach_provenance_to_bundle" in vars(automation)
    assert automation.attach_provenance_to_bundle is provenance.attach_provenance_to_bundle


# Every name a consumer imported from `lectern.automation` before the package
# split. `cli.py` and the existing tests import from this module and are not
# adjusted for the new layout, so narrowing this surface is a consumer break
# even when the suite stays green.
CONSUMED_NAMES = (
    "AutomationError",
    "AutomationState",
    "DEFAULT_STATE_PATH",
    "DEFAULT_YOUTUBE_API_KEY_ENV",
    "QueueItem",
    "QueueState",
    "STATE_SCHEMA_VERSION",
    "SourceKind",
    "SourcePolicy",
    "SourceRecord",
    "YOUTUBE_METADATA_ONLY_ERROR",
    "YouTubeAPIError",
    "YouTubePlaylistAdapter",
    "attach_provenance_to_bundle",
    "normalize_youtube_playlist_id",
    "open_state",
    "preflight_local_folder",
    "preflight_state_store",
    "preflight_youtube_playlist",
)


def test_automation_remains_the_single_import_surface() -> None:
    missing = [name for name in CONSUMED_NAMES if not hasattr(automation, name)]
    assert missing == []
    assert set(CONSUMED_NAMES) <= set(automation.__all__)


def test_every_split_module_is_reachable_through_the_facade() -> None:
    # A name that lives in a part module but is absent from the facade is a name
    # a consumer would have to learn the new layout to reach.
    for module in (records, state, local, youtube, provenance):
        exported = {
            name
            for name in vars(module)
            if not name.startswith("_") and name in getattr(module, "__all__", vars(module))
        }
        reachable = {name for name in exported if hasattr(automation, name)}
        # Only the names the facade deliberately publishes need to be reachable;
        # this asserts the facade is not empty for any part, which is the
        # failure mode worth catching.
        assert reachable, f"{module.__name__} contributes nothing to the facade"


def test_importing_the_state_store_does_not_pull_in_a_transport() -> None:
    """The store must not *transitively* acquire the HTTP transport either.

    Grepping this module's own source for `urllib` is too weak a check: an
    import of the source package reaches `sources.youtube`, and through it
    `urllib.request`, without the string ever appearing in `state.py`. The
    boundary this split claims is about what loading the persistence layer
    actually costs, so it is checked in a fresh interpreter.
    """

    probe = (
        "import sys; import lectern.state; "
        "print('urllib.request' in sys.modules, 'lectern.sources' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False False", result.stdout


def test_default_adapter_selection_covers_every_scannable_kind(
    monkeypatch: MonkeyPatch,
) -> None:
    """The composition root's kind-to-adapter mapping, exercised directly.

    Every other test injects an adapter explicitly, so this seam — the one thing
    that decides which provider a scan actually talks to — was reached by no
    test at all. It is also where the store/composition-root boundary is
    enforced, so a regression here would silently reattach the store to a
    concrete provider.
    """

    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-secret")

    folder = SourceRecord(
        id="src_local",
        kind=SourceKind.LOCAL_FOLDER,
        name="talks",
        root_path="/tmp/talks",
        policy=SourcePolicy.REVIEW,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    playlist = replace(folder, id="src_yt", kind=SourceKind.YOUTUBE_PLAYLIST, root_path="PL_SYNTH")
    one_shot = replace(folder, id="src_one", kind=SourceKind.ONE_SHOT)

    assert isinstance(default_source_adapter(folder), LocalFolderAdapter)
    assert isinstance(default_source_adapter(playlist), YouTubePlaylistAdapter)

    # A kind with no adapter must be refused, not silently scanned as something
    # else; the store's unimplemented hook raises the same message.
    with pytest.raises(AutomationError, match="unsupported source kind for scan"):
        default_source_adapter(one_shot)
    with pytest.raises(AutomationError, match="unsupported source kind for scan"):
        state.AutomationStateStore._default_adapter(  # pyright: ignore[reportPrivateUsage]
            cast(state.AutomationStateStore, object()), folder
        )

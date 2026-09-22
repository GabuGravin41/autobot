"""
Offline tests for autobot/knowledge/project_registry.py.

Uses a real temp directory (tmp_path) for storage rather than mocking the
filesystem — this module IS a thin filesystem wrapper, so testing it
against a fake filesystem would mostly test the mock, not the code. Mirrors
how skill_distiller.py's own behavior would naturally be tested (no
existing test file for it in this repo to match against directly, so this
follows the same real-tmp-dir approach used elsewhere in this suite, e.g.
tests/test_kaggle_tool.py's tmp_path fixtures for pull_kernel/kernel_output).
"""
from __future__ import annotations

import pytest

from autobot.knowledge.project_registry import KNOWN_BACKENDS, ProjectRegistry


@pytest.fixture
def registry(tmp_path):
    return ProjectRegistry(projects_dir=tmp_path / "projects")


class TestRegister:
    def test_register_creates_project(self, registry):
        p = registry.register(
            name="seqoy",
            working_dir=r"C:\Users\User 1\...\seqoy",
            backend="claude_code",
            intent="Referral tracking dashboard for a health facility. Need clerk auth, "
                   "Twilio, Africa's Talking SMS webhooks wired up.",
        )
        assert p.name == "seqoy"
        assert p.backend == "claude_code"
        assert p.session_id is None
        assert p.created_at

    def test_register_persists_to_disk(self, registry):
        registry.register("seqoy", "/work/seqoy", "claude_code", "build the thing")
        fresh = ProjectRegistry(projects_dir=registry.projects_dir)
        loaded = fresh.get("seqoy")
        assert loaded is not None
        assert loaded.intent == "build the thing"

    def test_intent_stored_verbatim_not_altered(self, registry):
        weird_intent = "  messy    spacing, weird caps DASHBOARD -- keep AS-IS!!  "
        p = registry.register("proj", "/work/proj", "antigravity", weird_intent)
        assert p.intent == weird_intent  # exactly as given, no normalization

    def test_unknown_backend_rejected(self, registry):
        with pytest.raises(ValueError, match="Unknown backend"):
            registry.register("x", "/work/x", "gpt_engineer_gui_clicking", "goal")

    def test_empty_name_rejected(self, registry):
        with pytest.raises(ValueError):
            registry.register("", "/work/x", "claude_code", "goal")

    def test_empty_intent_rejected(self, registry):
        with pytest.raises(ValueError):
            registry.register("x", "/work/x", "claude_code", "")

    def test_known_backends_are_cli_only(self):
        # Pin the actual values — a change here silently changing what
        # backends exist should fail a test, not just ship.
        assert set(KNOWN_BACKENDS) == {"claude_code", "antigravity"}

    def test_re_register_same_name_replaces_intent_but_keeps_session(self, registry):
        registry.register("seqoy", "/work/seqoy", "claude_code", "original goal")
        registry.update_session("seqoy", "sess-abc")
        updated = registry.register("seqoy", "/work/seqoy-v2", "claude_code", "revised goal")
        assert updated.intent == "revised goal"
        assert updated.working_dir == "/work/seqoy-v2"
        assert updated.session_id == "sess-abc"  # NOT wiped by re-registering


class TestGetAndList:
    def test_get_missing_project_returns_none(self, registry):
        assert registry.get("does-not-exist") is None

    def test_list_all_empty_initially(self, registry):
        assert registry.list_all() == []

    def test_list_all_returns_registered_projects(self, registry):
        registry.register("a", "/work/a", "claude_code", "goal a")
        registry.register("b", "/work/b", "antigravity", "goal b")
        names = {p.name for p in registry.list_all()}
        assert names == {"a", "b"}

    def test_list_all_orders_most_recently_checked_first(self, registry):
        registry.register("a", "/work/a", "claude_code", "goal a")
        registry.register("b", "/work/b", "claude_code", "goal b")
        registry.record_check_in("a", "a is on track")
        ordered = registry.list_all()
        assert ordered[0].name == "a"  # checked-in project sorts before never-checked one


class TestUpdateSessionAndCheckIn:
    def test_update_session_persists(self, registry):
        registry.register("p", "/work/p", "antigravity", "goal")
        registry.update_session("p", "conv-99")
        assert registry.get("p").session_id == "conv-99"

    def test_record_check_in_sets_timestamp_and_summary(self, registry):
        registry.register("p", "/work/p", "claude_code", "goal")
        assert registry.get("p").last_checked_at is None
        registry.record_check_in("p", "70% done, blocked on API key")
        updated = registry.get("p")
        assert updated.last_checked_at is not None
        assert updated.last_status_summary == "70% done, blocked on API key"

    def test_update_session_on_unregistered_project_raises(self, registry):
        with pytest.raises(KeyError):
            registry.update_session("ghost", "sess-1")

    def test_record_check_in_on_unregistered_project_raises(self, registry):
        with pytest.raises(KeyError):
            registry.record_check_in("ghost", "summary")


class TestIntentNotes:
    def test_add_intent_note_appends(self, registry):
        registry.register("p", "/work/p", "claude_code", "original intent")
        registry.add_intent_note("p", "also needs dark mode")
        registry.add_intent_note("p", "and mobile responsive")
        updated = registry.get("p")
        assert updated.intent == "original intent"  # unchanged
        assert updated.intent_notes == ["also needs dark mode", "and mobile responsive"]

    def test_full_intent_context_joins_intent_and_notes(self, registry):
        registry.register("p", "/work/p", "claude_code", "build a dashboard")
        registry.add_intent_note("p", "use dark theme")
        context = registry.get("p").full_intent_context()
        assert "build a dashboard" in context
        assert "use dark theme" in context

    def test_add_intent_note_on_unregistered_project_raises(self, registry):
        with pytest.raises(KeyError):
            registry.add_intent_note("ghost", "note")

    def test_blank_note_not_appended(self, registry):
        registry.register("p", "/work/p", "claude_code", "goal")
        registry.add_intent_note("p", "   ")
        assert registry.get("p").intent_notes == []


class TestForget:
    def test_forget_removes_project(self, registry):
        registry.register("p", "/work/p", "claude_code", "goal")
        assert registry.forget("p") is True
        assert registry.get("p") is None

    def test_forget_nonexistent_returns_false(self, registry):
        assert registry.forget("never-existed") is False


class TestFilesystemSafety:
    def test_name_with_illegal_windows_chars_does_not_raise(self, registry):
        # Windows forbids : * ? " < > | in filenames — a project name
        # containing them (plausible: "Q&A: dashboard?") must not crash
        # register() the way an unslugged filename would.
        p = registry.register('weird: name? "quoted" <tag>', "/work/x", "claude_code", "goal")
        assert p.name == 'weird: name? "quoted" <tag>'  # display name preserved
        # And it round-trips correctly through get() using the same slug.
        assert registry.get('weird: name? "quoted" <tag>') is not None

    def test_names_that_collide_after_slugging_do_not_overwrite_each_other(self, registry):
        # "Foo Bar" and "Foo-Bar" are different project names, but the
        # human-readable half of _safe_name() alone collapses both the
        # space and the hyphen to "_", producing the identical slug
        # "foo_bar" for both. Regression test for a real bug: without a
        # disambiguating hash, registering the second name resolved to the
        # SAME underlying JSON file as the first, silently overwriting an
        # unrelated project's stored session_id/intent/history.
        registry.register("Foo Bar", "/work/foobar", "claude_code", "project A intent")
        registry.update_session("Foo Bar", "sess-A")
        registry.register("Foo-Bar", "/work/foo-bar-v2", "antigravity", "project B intent")

        a = registry.get("Foo Bar")
        b = registry.get("Foo-Bar")
        assert a is not None and b is not None
        assert a.name == "Foo Bar"
        assert b.name == "Foo-Bar"
        assert a.intent == "project A intent"
        assert b.intent == "project B intent"
        assert a.session_id == "sess-A"
        assert b.session_id is None  # must NOT have inherited A's session
        assert {p.name for p in registry.list_all()} == {"Foo Bar", "Foo-Bar"}

    def test_names_colliding_to_empty_slug_do_not_overwrite_each_other(self, registry):
        # Two different all-punctuation names both reduce to the "project"
        # fallback slug before hashing — same collision risk, different
        # trigger (empty slug rather than a shared separator).
        registry.register("!!!", "/work/a", "claude_code", "intent A")
        registry.register("???", "/work/b", "antigravity", "intent B")
        assert registry.get("!!!").intent == "intent A"
        assert registry.get("???").intent == "intent B"

    def test_case_variation_of_same_name_still_updates_same_project(self, registry):
        # Preserve the pre-existing (intentional) behavior that slugging
        # is case-insensitive: re-registering under a different case of the
        # SAME name updates the same project rather than creating a new one.
        registry.register("Seqoy", "/work/seqoy", "claude_code", "original goal")
        registry.register("SEQOY", "/work/seqoy-v2", "claude_code", "revised goal")
        assert len(registry.list_all()) == 1
        assert registry.get("seqoy").intent == "revised goal"

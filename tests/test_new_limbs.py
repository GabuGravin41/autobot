"""
Tests for new Autobot Computer limbs: Cleaner, Quant, VSCode, and Eris.
Uses sync functions with asyncio.run to avoid requiring pytest-asyncio plugin.
"""
import pytest
import asyncio
from autobot.computer.computer import Computer
from autobot.computer.dispatch import dispatch_computer_call


@pytest.fixture
def comp():
    return Computer()


def test_cleaner_limb(comp):
    async def _run():
        ok, res = await dispatch_computer_call(comp, "computer.cleaner.get_disk_free()")
        assert ok is True
        assert "free_gb" in res
    asyncio.run(_run())


def test_quant_limb(comp):
    async def _run():
        ok, res = await dispatch_computer_call(comp, "computer.quant.list_strategies()")
        assert ok is True
        assert "strategies" in res
    asyncio.run(_run())


def test_vscode_limb(comp):
    async def _run():
        ok, res = await dispatch_computer_call(comp, "computer.vscode.is_available()")
        assert ok is True
    asyncio.run(_run())


def test_eris_limb(comp):
    async def _run():
        ok, res = await dispatch_computer_call(comp, "computer.eris.list_challenges()")
        assert ok is True
        assert "challenges" in res

        ok, res = await dispatch_computer_call(comp, "computer.eris.fetch_challenge('test-quest-01')")
        assert ok is True
        assert "test-quest-01" in res
    asyncio.run(_run())


def test_tool_catalog_includes_all_limbs(comp):
    catalog = comp.get_tool_catalog()
    assert "cleaner" in catalog
    assert "quant" in catalog
    assert "vscode" in catalog
    assert "eris" in catalog

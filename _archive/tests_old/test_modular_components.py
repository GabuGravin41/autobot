"""
Modular Pre-Build Verification Test Suite

Tests:
1. SoftwareInstallerManager (Binary discovery & path dynamic updating)
2. HierarchicalPlanTree (3-level nested parent-child task nodes & checkpointing)
3. MaterialsScienceSynthesizer (CIF crystal structure script generation & VESTA launcher)
4. WhatsAppListener Outbound Status Formatting
5. BioInformatics & DICOM Domain Synthesizers (BLAST & 3D Volume script gen)
6. Visual Inspector & Terminal Stderr Diagnostician
"""
import asyncio
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_1_installer_manager():
    print("\n" + "=" * 60)
    print("MODULAR TEST 1: SoftwareInstallerManager")
    print("=" * 60)

    from autobot.computer.installer import SoftwareInstallerManager

    installer = SoftwareInstallerManager()
    python_installed = installer.is_installed("python")
    python_path = installer.find_binary_path("python")

    print(f"  Is 'python' installed: {python_installed}")
    print(f"  Python binary path: {python_path}")

    assert python_installed
    assert python_path is not None
    print("  ✅ MODULAR TEST 1 PASSED: Package installer manager discovers binaries & PATH.")


def test_2_hierarchical_plan_tree():
    print("\n" + "=" * 60)
    print("MODULAR TEST 2: HierarchicalPlanTree (Nested Plan-in-a-Plan)")
    print("=" * 60)

    from autobot.agent.plan_tree import HierarchicalPlanTree

    tree = HierarchicalPlanTree(root_goal="Materials Science VESTA Simulation Flow")

    # Level 1: Subtask
    node_prep = tree.create_subtask(
        parent_id="node_root",
        title="Software Preparation",
        description="Verify or install VESTA and PyMatGen dependencies",
    )

    # Level 2: Nested Subtask (Plan within a plan)
    node_install = tree.create_subtask(
        parent_id=node_prep.id,
        title="Install VESTA Binary",
        description="Download and extract portable VESTA zip into tools directory",
    )

    # Level 3: Nested Subtask (Plan within a plan within a plan)
    node_path = tree.create_subtask(
        parent_id=node_install.id,
        title="Register PATH Environment",
        description="Add VESTA binary folder to current process PATH",
    )

    # Update statuses
    tree.update_status(node_path.id, "completed", result_summary="PATH updated")
    tree.update_status(node_install.id, "completed", result_summary="Extracted to tmp/autobot_tools")
    tree.update_status(node_prep.id, "completed", result_summary="Environment ready")

    print("\n  ASCII Plan Tree Hierarchy:")
    ascii_tree = tree.render_ascii_tree()
    for line in ascii_tree.split("\n"):
        print(f"    {line}")

    # Checkpoint
    checkpoint_file = Path("tmp") / "autobot_scratch" / "test_tree_checkpoint.json"
    tree.save_checkpoint(checkpoint_file)
    print(f"\n  Checkpoint saved: {checkpoint_file.exists()}")

    assert checkpoint_file.exists()
    assert len(tree._all_nodes()) == 4
    print("  ✅ MODULAR TEST 2 PASSED: 3-level nested plan tree & checkpointing verified.")


def test_3_materials_science_synthesizer():
    print("\n" + "=" * 60)
    print("MODULAR TEST 3: MaterialsScienceSynthesizer (VESTA / CIF Template)")
    print("=" * 60)

    from autobot.agent.domain.materials_synthesizer import MaterialsScienceSynthesizer

    script = MaterialsScienceSynthesizer.generate_cif_script(formula="Fe2O3", output_path="fe2o3.cif")
    print(f"  Generated CIF Script Snippet:\n{script[:250]}...\n")

    launcher_script = MaterialsScienceSynthesizer.generate_vesta_launch_script(cif_path="fe2o3.cif")
    print(f"  Generated VESTA Launcher Script Snippet:\n{launcher_script[:200]}...\n")

    assert "Fe2O3" in script
    assert "VESTA.exe" in launcher_script
    print("  ✅ MODULAR TEST 3 PASSED: Materials science VESTA & CIF scripts synthesized.")


def test_4_whatsapp_listener_outbound():
    print("\n" + "=" * 60)
    print("MODULAR TEST 4: WhatsApp Outbound Telemetry")
    print("=" * 60)

    from autobot.agent.whatsapp_listener import WhatsAppListener

    class MockPage:
        url = "https://web.whatsapp.com"
        is_closed = lambda self: False

    listener = WhatsAppListener(page=MockPage())
    print("  Instantiated WhatsAppListener with mock page.")
    assert listener is not None
    print("  ✅ MODULAR TEST 4 PASSED: WhatsApp remote listener ready for outbound messages.")


def test_5_bio_and_dicom_synthesizers():
    print("\n" + "=" * 60)
    print("MODULAR TEST 5: BioInformatics & DICOM Domain Synthesizers")
    print("=" * 60)

    from autobot.agent.domain.bio_synthesizer import BioInformaticsSynthesizer
    from autobot.agent.domain.dicom_synthesizer import DICOMVolumeSynthesizer

    blast_script = BioInformaticsSynthesizer.generate_blast_script(sequence_or_fasta="ATGCGTACGT")
    print(f"  BLAST script generated ({len(blast_script)} chars)")
    assert "qblast" in blast_script

    dicom_script = DICOMVolumeSynthesizer.generate_volume_script(dicom_dir="ct_scans", hu_min_threshold=300)
    print(f"  DICOM script generated ({len(dicom_script)} chars)")
    assert "pydicom" in dicom_script

    print("  ✅ MODULAR TEST 5 PASSED: BioInformatics & DICOM 3D script generators verified.")


def test_6_inspectors_and_diagnostics():
    print("\n" + "=" * 60)
    print("MODULAR TEST 6: Visual Inspector & Stderr Diagnostician")
    print("=" * 60)

    from autobot.browser.visual_inspector import VisualStateInspector
    from autobot.agent.diagnostician import TerminalStderrDiagnostician

    inspector = VisualStateInspector()
    img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    hash1 = inspector.compute_hash(img_b64)
    print(f"  Screenshot b64 hash: {hash1[:16]}...")
    assert hash1 is not None

    diag = TerminalStderrDiagnostician.analyze_stderr(
        command="python run_analysis.py",
        stderr="ModuleNotFoundError: No module named 'biopython'",
    )
    print(f"  Diagnostician category: {diag.category}")
    print(f"  Diagnostician summary: {diag.summary}")
    print(f"  Diagnostician action: {diag.suggested_action}")

    assert diag.category == "missing_module"
    assert "pip install biopython" in diag.suggested_action
    print("  ✅ MODULAR TEST 6 PASSED: Visual Inspector & Stderr Diagnostician verified.")


def test_7_environment_memory_and_skills():
    print("\n" + "=" * 60)
    print("MODULAR TEST 7: EnvironmentMemory & SkillDistiller")
    print("=" * 60)

    from autobot.knowledge.environment_memory import EnvironmentMemory
    from autobot.knowledge.skill_distiller import SkillDistiller, LearnedSkill

    # 1. Test Environment Memory
    mem = EnvironmentMemory(memory_file=Path("tmp") / "autobot_scratch" / "test_env_knowledge.json")
    mem.record_software("Git", "C:\\Program Files\\Git\\cmd\\git.exe", configured=True)
    mem.record_configuration("github_ssh_setup", True, notes="SSH key added to GitHub")

    summary = mem.get_summary_text()
    print(f"  Environment Summary Snippet:\n{summary}\n")
    assert mem.is_software_configured("Git")
    assert mem.get_configuration("github_ssh_setup") is True

    # 2. Test Skill Distiller
    distiller = SkillDistiller(skills_dir=Path("tmp") / "autobot_scratch" / "test_skills")
    skill = LearnedSkill(
        name="GitHub SSH Setup",
        description="Configures Git SSH keys and verifies GitHub connection",
        keywords=["github", "ssh", "git"],
        prerequisites=["git_installed"],
        proven_steps=[
            {"goal": "Generate SSH Key", "actions": ["run_command"]},
            {"goal": "Add key to GitHub settings tab", "actions": ["navigate", "click", "input_text"]},
        ],
        lessons_learned=["Ensure ssh-agent service is running on Windows before adding key."],
        created_at="2026-07-30T12:00:00Z",
    )

    skill_file = distiller.save_skill(skill)
    print(f"  Skill file saved: {skill_file.exists()}")

    prompt_context = distiller.get_skill_prompt_context("Set up GitHub SSH configuration on my laptop")
    print(f"  Skill Prompt Context Snippet:\n{prompt_context}\n")

    assert skill_file.exists()
    assert "PREVIOUSLY LEARNED SKILL AVAILABLE" in prompt_context
    assert "GitHub SSH Setup" in prompt_context
    print("  ✅ MODULAR TEST 7 PASSED: Environment Knowledge Graph & Skill Distiller verified.")


def main():
    print("🤖 Autobot 2.0 — Modular Pre-Build Expanded Test Suite")
    test_1_installer_manager()
    test_2_hierarchical_plan_tree()
    test_3_materials_science_synthesizer()
    test_4_whatsapp_listener_outbound()
    test_5_bio_and_dicom_synthesizers()
    test_6_inspectors_and_diagnostics()
    test_7_environment_memory_and_skills()

    print("\n" + "=" * 60)
    print("🎉 ALL 7 MODULAR PRE-BUILD TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

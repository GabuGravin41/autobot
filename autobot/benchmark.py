from __future__ import annotations

from dataclasses import dataclass

from .engine import AutomationEngine, TaskStep, WorkflowPlan


@dataclass
class BenchmarkCase:
    name: str
    description: str
    plan: WorkflowPlan


def default_benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="quick_logging",
            description="Verify workflow runner executes simple logging steps.",
            plan=WorkflowPlan(
                name="benchmark_quick_logging",
                description="Simple logging benchmark",
                steps=[
                    TaskStep(action="log", args={"message": "benchmark start"}, description="log start"),
                    TaskStep(action="wait", args={"seconds": 0.1}, description="wait briefly"),
                    TaskStep(action="log", args={"message": "benchmark end"}, description="log end"),
                ],
            ),
        ),
        BenchmarkCase(
            name="adapter_registry",
            description="Verify adapter library loads without runtime errors.",
            plan=WorkflowPlan(
                name="benchmark_adapter_registry",
                description="Adapter registry benchmark",
                steps=[TaskStep(action="adapter_list_actions", save_as="adapter_library", description="load adapters")],
            ),
        ),
    ]


def run_benchmarks(logger=None) -> list[dict]:
    logger = logger or (lambda _msg: None)
    results: list[dict] = []
    for case in default_benchmark_cases():
        engine = AutomationEngine(logger=logger)
        try:
            result = engine.run_plan(case.plan)
            results.append(
                {
                    "name": case.name,
                    "description": case.description,
                    "success": result.success,
                    "completed_steps": result.completed_steps,
                    "total_steps": result.total_steps,
                    "run_history_path": result.state.get("last_run_history_path", ""),
                }
            )
        except Exception as error:  # noqa: BLE001
            results.append(
                {
                    "name": case.name,
                    "description": case.description,
                    "success": False,
                    "error": str(error),
                }
            )
        finally:
            engine.close()
    return results

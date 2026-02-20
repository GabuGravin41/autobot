import threading
import tkinter as tk
from tkinter import messagebox, ttk
import json
import os
import time

from .autonomy import AutonomousConfig, AutonomousRunner
from .engine import AutomationEngine, TaskStep, WorkflowPlan
from .llm_brain import LLMBrain, PlanDraft
from .planner import build_plan_from_text
from .workflows import (
    builtin_workflows,
    console_fix_assist_workflow,
    research_paper_workflow,
    tool_call_stress_workflow,
    website_builder_workflow,
)


class AutobotUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Autobot Autonomous Controller")
        self.root.geometry("1240x760")
        self.root.minsize(1080, 680)

        self.task_var = tk.StringVar()
        self.topic_var = tk.StringVar()
        self.workflow_var = tk.StringVar(value="website_builder")
        self.adapter_var = tk.StringVar(value="whatsapp_web")
        self.adapter_action_var = tk.StringVar()
        self.adapter_params_var = tk.StringVar(value="{}")
        self.adapter_confirm_var = tk.BooleanVar(value=False)
        self.adapter_policy_var = tk.StringVar(value="balanced")
        self.adapter_prepare_token_var = tk.StringVar()
        self.browser_mode_var = tk.StringVar(value=os.getenv("AUTOBOT_BROWSER_MODE", "auto"))
        self.goal_var = tk.StringVar()
        self.autonomous_url_var = tk.StringVar(value="http://localhost:3000")
        self.autonomous_diag_cmd_var = tk.StringVar(value="pytest -q")
        self.autonomous_loops_var = tk.StringVar(value="5")
        self.autonomous_steps_var = tk.StringVar(value="5")
        self.autonomous_allow_desktop_var = tk.BooleanVar(value=False)
        self.autonomous_allow_sensitive_adapter_var = tk.BooleanVar(value=False)
        self.engine: AutomationEngine | None = None
        self.autonomous_runner: AutonomousRunner | None = None
        self._is_running = False
        self._last_adapter_run_key = ""
        self._last_adapter_run_time = 0.0
        self._ai_plan_steps: list[TaskStep] = []
        self._ai_brain = LLMBrain(logger=self._log)
        self._build_layout()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Autobot Desktop Automation", font=("Segoe UI", 13, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        ttk.Label(
            main,
            text=(
                "Quick task command (examples: search climate policy 2026, open overleaf, "
                "run python -m autobot.main). You can also paste multiple lines."
            ),
        ).pack(anchor="w")

        task_row = ttk.Frame(main)
        task_row.pack(fill="x", pady=(6, 8))
        self.task_entry = ttk.Entry(task_row, textvariable=self.task_var)
        self.task_entry.pack(side="left", fill="x", expand=True)
        self.task_entry.focus_set()
        self.task_entry.bind("<Return>", self._on_run_task)

        self.run_button = ttk.Button(task_row, text="Run Task", command=self._run_task)
        self.run_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(task_row, text="Stop", command=self._stop_task, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(task_row, text="Browser mode").pack(side="left", padx=(12, 6))
        self.browser_mode_picker = ttk.Combobox(
            task_row,
            values=["auto", "human_profile", "devtools"],
            state="readonly",
            textvariable=self.browser_mode_var,
            width=14,
        )
        self.browser_mode_picker.pack(side="left")
        ttk.Button(task_row, text="Apply Mode", command=self._apply_browser_mode).pack(side="left", padx=(8, 0))

        workflow_frame = ttk.LabelFrame(main, text="Preset Workflows", padding=8)
        workflow_frame.pack(fill="x", pady=(0, 8))

        options = list(builtin_workflows().keys())
        self.workflow_picker = ttk.Combobox(
            workflow_frame,
            values=options,
            state="readonly",
            textvariable=self.workflow_var,
            width=24,
        )
        self.workflow_picker.pack(side="left")
        ttk.Label(workflow_frame, text="Topic / Context").pack(side="left", padx=(12, 6))
        ttk.Entry(workflow_frame, textvariable=self.topic_var).pack(side="left", fill="x", expand=True)
        self.run_workflow_button = ttk.Button(workflow_frame, text="Run Workflow", command=self._run_selected_workflow)
        self.run_workflow_button.pack(side="left", padx=8)
        self.run_benchmarks_button = ttk.Button(workflow_frame, text="Run Benchmarks", command=self._run_benchmarks)
        self.run_benchmarks_button.pack(side="left", padx=8)

        ai_frame = ttk.LabelFrame(main, text="AI Planner Chat", padding=8)
        ai_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(ai_frame, text="Describe what you want Autobot to do").grid(row=0, column=0, sticky="w")
        self.ai_prompt_text = tk.Text(ai_frame, height=4, wrap="word")
        self.ai_prompt_text.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(6, 6))
        self.generate_plan_button = ttk.Button(ai_frame, text="Generate Plan", command=self._generate_ai_plan)
        self.generate_plan_button.grid(row=2, column=0, sticky="w")
        self.execute_plan_button = ttk.Button(
            ai_frame, text="Execute AI Plan", command=self._execute_ai_plan, state="disabled"
        )
        self.execute_plan_button.grid(row=2, column=1, sticky="w", padx=(8, 0))
        self.ai_plan_label = ttk.Label(ai_frame, text="No AI plan generated yet.", foreground="#666666")
        self.ai_plan_label.grid(row=2, column=2, columnspan=4, sticky="w", padx=(8, 0))
        ai_frame.columnconfigure(0, weight=1)

        adapter_frame = ttk.LabelFrame(main, text="Stateful App Adapters", padding=8)
        adapter_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(adapter_frame, text="Adapter").grid(row=0, column=0, sticky="w")
        self.adapter_picker = ttk.Combobox(
            adapter_frame,
            values=["whatsapp_web", "instagram_web", "overleaf_web", "google_docs_web", "grok_web", "vscode_desktop"],
            state="readonly",
            textvariable=self.adapter_var,
            width=20,
        )
        self.adapter_picker.grid(row=0, column=1, sticky="w", padx=(8, 8))
        self.adapter_picker.bind("<<ComboboxSelected>>", lambda _e: self._refresh_adapter_actions())
        ttk.Label(adapter_frame, text="Action").grid(row=0, column=2, sticky="w")
        self.adapter_action_picker = ttk.Combobox(
            adapter_frame,
            values=[],
            state="readonly",
            textvariable=self.adapter_action_var,
            width=26,
        )
        self.adapter_action_picker.grid(row=0, column=3, sticky="w", padx=(8, 8))
        self.adapter_action_picker.bind("<<ComboboxSelected>>", lambda _e: self._update_adapter_docs())
        ttk.Checkbutton(
            adapter_frame,
            text="I confirm sensitive action",
            variable=self.adapter_confirm_var,
        ).grid(row=0, column=4, sticky="w")
        ttk.Label(adapter_frame, text="Policy").grid(row=0, column=5, sticky="w", padx=(8, 0))
        self.adapter_policy_picker = ttk.Combobox(
            adapter_frame,
            values=["strict", "balanced", "trusted"],
            state="readonly",
            textvariable=self.adapter_policy_var,
            width=10,
        )
        self.adapter_policy_picker.grid(row=0, column=6, sticky="w")
        self.set_policy_button = ttk.Button(adapter_frame, text="Set Policy", command=self._set_adapter_policy)
        self.set_policy_button.grid(row=0, column=7, sticky="w", padx=(8, 0))

        ttk.Label(adapter_frame, text="Params (JSON)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(adapter_frame, textvariable=self.adapter_params_var).grid(
            row=1, column=1, columnspan=4, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        self.run_adapter_button = ttk.Button(adapter_frame, text="Run Adapter Action", command=self._run_adapter_action)
        self.run_adapter_button.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.adapter_docs = ttk.Label(adapter_frame, text="", foreground="#666666")
        self.adapter_docs.grid(row=2, column=2, columnspan=3, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(adapter_frame, text="Prepared token").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(adapter_frame, textvariable=self.adapter_prepare_token_var).grid(
            row=3, column=1, columnspan=4, sticky="ew", padx=(8, 0), pady=(8, 0)
        )
        self.prepare_sensitive_button = ttk.Button(adapter_frame, text="Prepare Sensitive", command=self._prepare_sensitive_action)
        self.prepare_sensitive_button.grid(
            row=3, column=5, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self.confirm_token_button = ttk.Button(adapter_frame, text="Confirm Token", command=self._confirm_sensitive_action)
        self.confirm_token_button.grid(
            row=3, column=6, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        adapter_frame.columnconfigure(1, weight=1)
        adapter_frame.columnconfigure(3, weight=1)

        autonomous = ttk.LabelFrame(main, text="Autonomous Multi-Loop Mode", padding=8)
        autonomous.pack(fill="x", pady=(0, 8))

        ttk.Label(autonomous, text="Goal").grid(row=0, column=0, sticky="w")
        ttk.Entry(autonomous, textvariable=self.goal_var).grid(row=0, column=1, columnspan=5, sticky="ew", padx=(8, 0))

        ttk.Label(autonomous, text="Target URL").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(autonomous, textvariable=self.autonomous_url_var).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(6, 0)
        )
        ttk.Label(autonomous, text="Diagnostics Command").grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Entry(autonomous, textvariable=self.autonomous_diag_cmd_var).grid(
            row=1, column=4, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0)
        )

        ttk.Label(autonomous, text="Max Loops").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(autonomous, textvariable=self.autonomous_loops_var, width=8).grid(
            row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(autonomous, text="Max Steps/Loop").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(autonomous, textvariable=self.autonomous_steps_var, width=8).grid(
            row=2, column=3, sticky="w", padx=(8, 0), pady=(6, 0)
        )
        ttk.Checkbutton(
            autonomous,
            text="Allow desktop input actions (pyautogui)",
            variable=self.autonomous_allow_desktop_var,
        ).grid(row=2, column=4, columnspan=2, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Checkbutton(
            autonomous,
            text="Allow sensitive adapter actions",
            variable=self.autonomous_allow_sensitive_adapter_var,
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(8, 0))

        self.run_autonomous_button = ttk.Button(autonomous, text="Run Autonomous Mode", command=self._run_autonomous_mode)
        self.run_autonomous_button.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        autonomous.columnconfigure(1, weight=1)
        autonomous.columnconfigure(4, weight=1)

        ttk.Label(main, text="Activity").pack(anchor="w")
        self.log_box = tk.Text(main, height=14, wrap="word")
        self.log_box.pack(fill="both", expand=True)
        self.log_box.insert(
            "end",
            (
                "Autobot ready.\n"
                "- Uses your installed Chrome profile (cookies/session).\n"
                "- If Chrome is running and profile is locked, close Chrome and retry.\n\n"
                "- Browser mode can be set to auto/human_profile/devtools from the top row.\n\n"
                "Supported quick commands:\n"
                "- search <query>\n"
                "- open <url|target>\n"
                "- run <os command>\n"
                "- list adapters\n"
                "- adapter <name> <action> <json_params>\n"
                "- wait <seconds>\n\n"
                "Autonomous mode:\n"
                "- Repeats diagnose -> plan -> execute -> retest loops.\n"
                "- Uses LLM if GOOGLE_API_KEY is set, otherwise manual handoff fallback.\n\n"
            ),
        )
        self.log_box.configure(state="disabled")
        self._adapter_library = self._load_adapter_library()
        self._refresh_adapter_actions()
        self._apply_browser_mode(silent=True)

    def _on_run_task(self, _event) -> None:
        self._run_task()

    def _run_task(self) -> None:
        raw = self.task_var.get().strip()
        if not raw:
            messagebox.showinfo("Autobot", "Enter a task first.")
            return
        plan = self._build_plan_from_text_block(raw)
        self._run_plan(plan)

    def _run_selected_workflow(self) -> None:
        key = self.workflow_var.get().strip()
        topic = self.topic_var.get().strip()

        if key == "website_builder":
            plan = website_builder_workflow(topic)
        elif key == "research_paper":
            plan = research_paper_workflow(topic)
        elif key == "console_fix_assist":
            plan = console_fix_assist_workflow(topic or "http://localhost:3000")
        elif key == "tool_call_stress":
            # Topic/context format:
            # <phone>|<docs_existing_url>|<download_check_path>|<message>
            parts = [item.strip() for item in topic.split("|")] if topic else []
            while len(parts) < 4:
                parts.append("")
            plan = tool_call_stress_workflow(
                whatsapp_phone=parts[0],
                docs_existing_url=parts[1],
                download_check_path=parts[2],
                outgoing_message=parts[3] or "Autobot tool-calling test message",
            )
        else:
            messagebox.showerror("Autobot", f"Unknown workflow '{key}'.")
            return

        self._run_plan(plan)

    def _run_benchmarks(self) -> None:
        self._run_plan(
            WorkflowPlan(
                name="ui_benchmark_run",
                description="Run benchmark suite from UI.",
                steps=[TaskStep(action="benchmark_run", save_as="benchmark_results", description="Run benchmarks")],
            )
        )

    def _generate_ai_plan(self) -> None:
        if self._is_running:
            self._log("A task is already running. Wait or press Stop.")
            return
        prompt = self.ai_prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showinfo("Autobot", "Enter an AI prompt first.")
            return
        draft = self._ai_brain.generate_plan_draft(
            user_prompt=prompt,
            allowed_actions=self._allowed_actions_for_ai(),
            max_steps=12,
        )
        self._apply_plan_draft(draft)

    def _execute_ai_plan(self) -> None:
        if not self._ai_plan_steps:
            messagebox.showinfo("Autobot", "Generate an AI plan first.")
            return
        self._run_plan(
            WorkflowPlan(
                name="ai_generated_plan",
                description="Execute AI generated plan.",
                steps=self._ai_plan_steps,
            )
        )

    def _run_plan(self, plan: WorkflowPlan) -> None:
        if self._is_running:
            self._log("A task is already running. Wait or press Stop.")
            return
        self._apply_browser_mode(silent=True)
        self._is_running = True
        self._set_running_ui_state(True)
        self.engine = AutomationEngine(logger=self._log)
        self.autonomous_runner = None

        def runner() -> None:
            try:
                result = self.engine.run_plan(plan)
                self._log(
                    f"Result: success={result.success}, completed_steps={result.completed_steps}/{result.total_steps}"
                )
                run_path = str(result.state.get("last_run_history_path", "")).strip()
                run_dir = str(result.state.get("run_dir", "")).strip()
                if run_dir:
                    self._log(f"Run folder: {run_dir}")
                if run_path:
                    self._log(f"Run history: {run_path}")
                token_payload = result.state.get("sensitive_token_payload")
                if isinstance(token_payload, dict):
                    token = str(token_payload.get("token", "")).strip()
                    if token:
                        self.adapter_prepare_token_var.set(token)
                self._log_known_state_outputs(result.state)
            except Exception as error:  # noqa: BLE001
                self._log(f"Error: {error}")
            finally:
                self.engine = None
                self._is_running = False
                self.root.after(
                    0,
                    lambda: self._set_running_ui_state(False),
                )

        threading.Thread(target=runner, daemon=True).start()

    def _run_autonomous_mode(self) -> None:
        if self._is_running:
            self._log("A task is already running. Wait or press Stop.")
            return
        self._apply_browser_mode(silent=True)
        goal = self.goal_var.get().strip()
        if not goal:
            messagebox.showinfo("Autobot", "Enter a goal for autonomous mode.")
            return

        if _looks_like_mass_message(goal):
            messagebox.showwarning(
                "Autobot",
                (
                    "High-risk messaging automation detected. Use explicit, consent-based one-to-one tasks only. "
                    "Bulk or unsolicited messaging is blocked."
                ),
            )
            return

        loops = _safe_int(self.autonomous_loops_var.get(), default=5)
        steps = _safe_int(self.autonomous_steps_var.get(), default=5)
        config = AutonomousConfig(
            max_loops=max(1, min(loops, 30)),
            max_steps_per_loop=max(1, min(steps, 10)),
            diagnostics_command=self.autonomous_diag_cmd_var.get().strip(),
            target_url=self.autonomous_url_var.get().strip(),
            allow_desktop_actions=self.autonomous_allow_desktop_var.get(),
            allow_sensitive_adapter_actions=self.autonomous_allow_sensitive_adapter_var.get(),
        )

        self._is_running = True
        self._set_running_ui_state(True)
        self.engine = AutomationEngine(logger=self._log)
        self.autonomous_runner = AutonomousRunner(engine=self.engine, logger=self._log)

        def runner() -> None:
            try:
                result = self.autonomous_runner.run(goal=goal, config=config)
                self._log(
                    "Autonomous mode finished: "
                    f"success={result.success}, loops={result.completed_steps}/{result.total_steps}"
                )
                run_path = str(result.state.get("last_run_history_path", "")).strip()
                run_dir = str(result.state.get("run_dir", "")).strip()
                if run_dir:
                    self._log(f"Run folder: {run_dir}")
                if run_path:
                    self._log(f"Run history: {run_path}")
            except Exception as error:  # noqa: BLE001
                self._log(f"Autonomous mode error: {error}")
            finally:
                self.engine = None
                self.autonomous_runner = None
                self._is_running = False
                self.root.after(
                    0,
                    lambda: self._set_running_ui_state(False),
                )

        threading.Thread(target=runner, daemon=True).start()

    def _run_adapter_action(self) -> None:
        if self._is_running:
            self._log("A task is already running. Wait or press Stop.")
            return
        adapter_name = self.adapter_var.get().strip()
        adapter_action = self.adapter_action_var.get().strip()
        if not adapter_name or not adapter_action:
            messagebox.showerror("Autobot", "Select adapter and action.")
            return
        try:
            params = json.loads(self.adapter_params_var.get().strip() or "{}")
        except json.JSONDecodeError as error:
            messagebox.showerror("Autobot", f"Invalid JSON params: {error}")
            return
        if not isinstance(params, dict):
            messagebox.showerror("Autobot", "Params JSON must be an object.")
            return
        # Debounce repeated accidental triggers (key repeat / focus glitches).
        run_key = f"{adapter_name}:{adapter_action}:{json.dumps(params, sort_keys=True)}"
        now = time.time()
        if run_key == self._last_adapter_run_key and (now - self._last_adapter_run_time) < 1.2:
            self._log("Ignored duplicate adapter action trigger (debounced).")
            return
        self._last_adapter_run_key = run_key
        self._last_adapter_run_time = now

        requires_confirmation = self._requires_confirmation(adapter_name, adapter_action)
        if requires_confirmation and not self.adapter_confirm_var.get():
            messagebox.showwarning(
                "Autobot",
                "This adapter action is sensitive and requires confirmation. Tick the confirmation checkbox.",
            )
            return

        plan = WorkflowPlan(
            name="adapter_ui_call",
            description=f"Adapter action from UI: {adapter_name}.{adapter_action}",
            steps=[
                TaskStep(
                    action="adapter_call",
                    args={
                        "adapter": adapter_name,
                        "adapter_action": adapter_action,
                        "params": params,
                        "confirmed": self.adapter_confirm_var.get(),
                    },
                    description=f"Adapter call {adapter_name}.{adapter_action}",
                )
            ],
        )
        self._run_plan(plan)

    def _set_adapter_policy(self) -> None:
        profile = self.adapter_policy_var.get().strip()
        plan = WorkflowPlan(
            name="adapter_set_policy_ui",
            description=f"Set adapter policy via UI: {profile}",
            steps=[
                TaskStep(
                    action="adapter_set_policy",
                    args={"profile": profile},
                    description=f"Set adapter policy to {profile}",
                )
            ],
        )
        self._run_plan(plan)

    def _prepare_sensitive_action(self) -> None:
        adapter_name = self.adapter_var.get().strip()
        adapter_action = self.adapter_action_var.get().strip()
        try:
            params = json.loads(self.adapter_params_var.get().strip() or "{}")
        except json.JSONDecodeError as error:
            messagebox.showerror("Autobot", f"Invalid JSON params: {error}")
            return
        if not isinstance(params, dict):
            messagebox.showerror("Autobot", "Params JSON must be an object.")
            return
        plan = WorkflowPlan(
            name="adapter_prepare_sensitive_ui",
            description=f"Prepare sensitive action: {adapter_name}.{adapter_action}",
            steps=[
                TaskStep(
                    action="adapter_prepare_sensitive",
                    args={"adapter": adapter_name, "adapter_action": adapter_action, "params": params},
                    save_as="sensitive_token_payload",
                    description=f"Prepare token for {adapter_name}.{adapter_action}",
                )
            ],
        )
        self._run_plan(plan)

    def _confirm_sensitive_action(self) -> None:
        token = self.adapter_prepare_token_var.get().strip()
        if not token:
            messagebox.showerror("Autobot", "Enter a prepared token first.")
            return
        plan = WorkflowPlan(
            name="adapter_confirm_sensitive_ui",
            description="Confirm prepared sensitive action token.",
            steps=[TaskStep(action="adapter_confirm_sensitive", args={"token": token}, description="Confirm token")],
        )
        self._run_plan(plan)

    def _stop_task(self) -> None:
        if self.autonomous_runner:
            self.autonomous_runner.cancel()
        if self.engine:
            self.engine.cancel()
            # Force close browser immediately to avoid waiting on long hangs.
            self.engine.close()
        self._is_running = False
        self._set_running_ui_state(False)

    def _log(self, message: str) -> None:
        def writer() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.root.after(0, writer)
        self._capture_token_from_message(message)

    def _load_adapter_library(self) -> dict[str, dict]:
        engine = AutomationEngine(logger=self._log)
        try:
            data = engine.get_adapter_library()
            if isinstance(data, dict):
                return data
            return {}
        finally:
            engine.close()

    def _refresh_adapter_actions(self) -> None:
        adapter_name = self.adapter_var.get().strip()
        actions = sorted(list((self._adapter_library.get(adapter_name) or {}).keys()))
        self.adapter_action_picker["values"] = actions
        if actions:
            self.adapter_action_var.set(actions[0])
            self._update_adapter_docs()
        else:
            self.adapter_action_var.set("")
            self.adapter_docs.configure(text="")

    def _update_adapter_docs(self) -> None:
        adapter_name = self.adapter_var.get().strip()
        action_name = self.adapter_action_var.get().strip()
        spec = ((self._adapter_library.get(adapter_name) or {}).get(action_name) or {})
        if not spec:
            self.adapter_docs.configure(text="")
            return
        requires = bool(spec.get("requires_confirmation", False))
        description = str(spec.get("description", ""))
        suffix = " [CONFIRM REQUIRED]" if requires else ""
        self.adapter_docs.configure(text=description + suffix)

    def _requires_confirmation(self, adapter_name: str, action_name: str) -> bool:
        spec = ((self._adapter_library.get(adapter_name) or {}).get(action_name) or {})
        return bool(spec.get("requires_confirmation", False))

    def _capture_token_from_message(self, message: str) -> None:
        marker = "Prepared sensitive action token for:"
        if marker not in message:
            return
        payload = {}
        if self.engine:
            payload = self.engine.state.get("last_sensitive_prepare", {}) or {}
        token = str(payload.get("token", "")).strip()
        if token:
            self.adapter_prepare_token_var.set(token)

    def _apply_browser_mode(self, silent: bool = False) -> None:
        mode = self.browser_mode_var.get().strip().lower()
        if mode not in {"auto", "human_profile", "devtools"}:
            mode = "auto"
            self.browser_mode_var.set(mode)
        os.environ["AUTOBOT_BROWSER_MODE"] = mode
        if not silent:
            self._log(f"Browser mode applied: {mode}")

    def _set_running_ui_state(self, running: bool) -> None:
        normal_or_disabled = "disabled" if running else "normal"
        self.run_button.configure(state=normal_or_disabled)
        self.run_workflow_button.configure(state=normal_or_disabled)
        self.run_benchmarks_button.configure(state=normal_or_disabled)
        self.generate_plan_button.configure(state=normal_or_disabled)
        self.execute_plan_button.configure(state="disabled" if running or not self._ai_plan_steps else "normal")
        self.run_adapter_button.configure(state=normal_or_disabled)
        self.prepare_sensitive_button.configure(state=normal_or_disabled)
        self.confirm_token_button.configure(state=normal_or_disabled)
        self.set_policy_button.configure(state=normal_or_disabled)
        self.run_autonomous_button.configure(state=normal_or_disabled)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _apply_plan_draft(self, draft: PlanDraft) -> None:
        self._ai_plan_steps = draft.steps
        summary = draft.summary[:240] + ("..." if len(draft.summary) > 240 else "")
        self.ai_plan_label.configure(text=f"{draft.title}: {summary} | steps={len(draft.steps)}")
        self.execute_plan_button.configure(state="normal" if draft.steps and not self._is_running else "disabled")
        self._log(f"AI plan generated: {draft.title} ({len(draft.steps)} steps)")
        for idx, step in enumerate(draft.steps, start=1):
            self._log(f"  AI Step {idx}: {step.action} - {step.description}")

    def _allowed_actions_for_ai(self) -> list[str]:
        return [
            "log",
            "wait",
            "open_url",
            "search_google",
            "adapter_list_actions",
            "adapter_get_telemetry",
            "adapter_call",
            "adapter_set_policy",
            "adapter_prepare_sensitive",
            "adapter_confirm_sensitive",
            "open_vscode",
            "open_app",
            "open_path",
            "run_command",
            "clipboard_get",
            "clipboard_set",
            "desktop_type",
            "desktop_hotkey",
            "desktop_move",
            "desktop_click",
            "desktop_switch_window",
            "desktop_press",
            "browser_mode_status",
        ]

    def _build_plan_from_text_block(self, raw_text: str) -> WorkflowPlan:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if len(lines) <= 1:
            return build_plan_from_text(raw_text.strip())

        merged_steps: list[TaskStep] = []
        for line in lines:
            plan = build_plan_from_text(line)
            merged_steps.extend(plan.steps)

        return WorkflowPlan(
            name="batch_commands",
            description="Execute multiple quick commands sequentially.",
            steps=merged_steps,
        )

    def _log_known_state_outputs(self, state: dict) -> None:
        adapter_library = state.get("adapter_library")
        if isinstance(adapter_library, dict):
            self._log(f"Adapter library loaded: {', '.join(sorted(adapter_library.keys()))}")
        telemetry = state.get("adapter_telemetry")
        if isinstance(telemetry, dict):
            self._log(f"Adapter telemetry loaded for: {', '.join(sorted(telemetry.keys()))}")
        mode_status = state.get("browser_mode_status")
        if isinstance(mode_status, dict):
            active = mode_status.get("active_mode", "")
            configured = mode_status.get("configured_mode", "")
            self._log(f"Browser mode status: active={active}, configured={configured}")
        benchmark_results = state.get("benchmark_results")
        if isinstance(benchmark_results, list):
            success_count = sum(1 for item in benchmark_results if isinstance(item, dict) and item.get("success"))
            self._log(f"Benchmarks: {success_count}/{len(benchmark_results)} passed.")


def launch_ui() -> None:
    root = tk.Tk()
    AutobotUI(root)
    root.mainloop()


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value.strip())
    except ValueError:
        return default


def _looks_like_mass_message(goal: str) -> bool:
    lowered = goal.lower()
    keywords = [
        "message all contacts",
        "bulk message",
        "blast message",
        "spam",
        "send to all",
    ]
    return any(item in lowered for item in keywords)

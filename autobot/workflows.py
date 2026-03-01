from __future__ import annotations

from .engine import TaskStep, WorkflowPlan


def simple_search_workflow(query: str) -> WorkflowPlan:
    return WorkflowPlan(
        name="search",
        description=f"Search Google for: {query}",
        steps=[
            TaskStep(
                action="search_google",
                args={"query": query},
                description=f"Search Google for '{query}'",
            )
        ],
    )


def open_target_workflow(target: str) -> WorkflowPlan:
    normalized = target.strip().lower()
    
    # Handle specific local apps first
    if normalized in {"vscode", "vs code", "code"}:
        return WorkflowPlan(
            name="open_vscode",
            description="Open Visual Studio Code.",
            steps=[TaskStep(action="open_vscode", description="Open VS Code")],
        )
    
    # Handle known websites
    known_sites = {
        "overleaf": "https://www.overleaf.com",
        "casa": "https://www.casa.ai",
        "casa ai": "https://www.casa.ai",
        "grok": "https://grok.com",
        "deepseek": "https://chat.deepseek.com",
        "chatgpt": "https://chatgpt.com",
        "openai": "https://openai.com",
        "google": "https://www.google.com",
        "chrome": "https://www.google.com",
        "browser": "https://www.google.com",
        "whatsapp": "https://web.whatsapp.com",
        "instagram": "https://www.instagram.com",
        "google ai studio": "https://aistudio.google.com",
        "aistudio": "https://aistudio.google.com",
        "leetcode": "https://leetcode.com/problemset/",
        "kaggle": "https://www.kaggle.com",
    }
    
    if normalized in known_sites:
        url = known_sites[normalized]
        return WorkflowPlan(
            name="open_target",
            description=f"Open target: {normalized}",
            steps=[TaskStep(action="open_url", args={"url": url}, description=f"Open {url}")],
        )

    # If it looks like a URL (contains a dot), open it directly
    if "." in normalized and " " not in normalized:
        return WorkflowPlan(
            name="open_target",
            description=f"Open URL: {target}",
            steps=[TaskStep(action="open_url", args={"url": target}, description=f"Open {target}")],
        )

    # Fallback for unknown software/sites: Search Google for the official site
    return WorkflowPlan(
        name="open_search_target",
        description=f"Search for and open target: {target}",
        steps=[
            TaskStep(
                action="search_google", 
                args={"query": f"{target} official website"}, 
                description=f"Search Google for '{target}'"
            ),
            TaskStep(
                action="log",
                args={"message": f"I've searched for '{target}'. You can now click the official link or ask me to perform actions on the page."},
                description="Guide user after search"
            )
        ],
    )


def website_builder_workflow(topic: str) -> WorkflowPlan:
    topic = topic.strip() or "new product"
    return WorkflowPlan(
        name="website_builder",
        description="Open your coding stack and prep a website build flow.",
        steps=[
            TaskStep(action="open_vscode", description="Open VS Code", continue_on_error=True),
            TaskStep(action="open_url", args={"url": "https://www.casa.ai"}, description="Open CASA AI", continue_on_error=True),
            TaskStep(action="open_url", args={"url": "https://grok.com"}, description="Open Grok", continue_on_error=True),
            TaskStep(
                action="search_google",
                args={"query": f"best modern website layout ideas for {topic}"},
                description="Gather design references",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={
                    "message": (
                        "Workspace is ready. Next loop: ask CASA/Grok for code, run in browser, capture errors, iterate."
                    )
                },
                description="Log next action guidance",
            ),
        ],
    )


def research_paper_workflow(topic: str) -> WorkflowPlan:
    topic = topic.strip() or "AI systems"
    overleaf_project_title = "Autobot Research Paper"
    return WorkflowPlan(
        name="research_paper",
        description="End-to-end research flow: search, ask Grok for LaTeX, open Overleaf, and compile.",
        steps=[
            TaskStep(
                action="search_google",
                args={"query": f"latest peer-reviewed references on {topic}"},
                description="Search Google for latest peer-reviewed references",
                continue_on_error=True,
            ),
            TaskStep(
                action="wait",
                args={"seconds": 8},
                description="Give Google time to load results like a human would",
            ),
            TaskStep(
                action="clipboard_set",
                args={
                    "text": (
                        "You are an expert research assistant. Using the latest peer-reviewed references on "
                        f"{topic}, pick one reference that would support a strong paper and write a full LaTeX "
                        "article for it. Return ONLY LaTeX source code, suitable to paste directly into Overleaf. "
                        "Do not explain anything."
                    )
                },
                description="Prepare Grok prompt for LaTeX paper based on latest references",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "open_home", "params": {}},
                description="Open Grok in browser",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={
                    "adapter": "grok_web",
                    "adapter_action": "ask_latex_from_clipboard",
                    "params": {
                        "instruction": (
                            "Paste the latest peer-reviewed references from the browser context and write a LaTeX article "
                            "for the best reference as described in the clipboard prompt."
                        )
                    },
                },
                description="Ask Grok to generate LaTeX paper from clipboard prompt",
                continue_on_error=True,
            ),
            TaskStep(
                action="wait",
                args={"seconds": 90},
                description="Wait for Grok to finish responding before copying (slow, human-paced)",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "copy_visible_response", "params": {}},
                retries=2,
                retry_delay_seconds=8.0,
                description="Copy Grok LaTeX response to clipboard",
                continue_on_error=True,
            ),
            TaskStep(
                action="clipboard_get",
                save_as="latex_text",
                description="Capture LaTeX source from clipboard",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={
                    "message": (
                        "Captured LaTeX length: {latex_text} (first 200 chars shown below in logs if needed)."
                    )
                },
                description="Log that LaTeX was captured (for debugging)",
                condition="bool(state.get('latex_text'))",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "overleaf_web", "adapter_action": "open_dashboard", "params": {}},
                description="Open Overleaf dashboard (sign in with Google if prompted)",
                condition="state.get('latex_text') and '\\\\begin' in state.get('latex_text','')",
                continue_on_error=True,
            ),
            TaskStep(
                action="wait",
                args={"seconds": 10},
                description="Give Overleaf dashboard time to load",
                condition="state.get('latex_text') and '\\\\begin' in state.get('latex_text','')",
            ),
            TaskStep(
                action="adapter_call",
                args={
                    "adapter": "overleaf_web",
                    "adapter_action": "open_project",
                    "params": {"title": overleaf_project_title},
                },
                description=f"Open Overleaf project: {overleaf_project_title}",
                condition="state.get('latex_text') and '\\\\begin' in state.get('latex_text','')",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={
                    "adapter": "overleaf_web",
                    "adapter_action": "replace_editor_text",
                    "params": {"text": "{latex_text}"},
                },
                description="Paste LaTeX into active Overleaf project",
                condition="state.get('latex_text') and '\\\\begin' in state.get('latex_text','')",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "overleaf_web", "adapter_action": "compile_project", "params": {}},
                description="Compile LaTeX project in Overleaf",
                condition="state.get('latex_text') and '\\\\begin' in state.get('latex_text','')",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={
                    "message": (
                        "Research paper workflow finished. Grok LaTeX should be in Overleaf; review and click Recompile again if needed."
                    )
                },
                description="Log completion of research paper workflow",
            ),
        ],
    )


def console_fix_assist_workflow(local_url: str = "http://localhost:3000") -> WorkflowPlan:
    return WorkflowPlan(
        name="console_fix_assist",
        description="Open local app and gather console diagnostics.",
        steps=[
            TaskStep(action="open_url", args={"url": local_url}, description="Open local app", continue_on_error=True),
            TaskStep(action="wait", args={"seconds": 2}, description="Wait for app to settle"),
            TaskStep(
                action="browser_read_console_errors",
                save_as="console_errors",
                description="Capture browser console-like errors",
                continue_on_error=True,
            ),
            TaskStep(
                action="clipboard_set",
                args={"text": "{console_errors}"},
                description="Copy captured errors to clipboard",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={"message": "Console errors copied to clipboard. Paste into your coding assistant to request a fix."},
                description="Log next action guidance",
            ),
        ],
    )


def code_iteration_workflow(test_command: str = "pytest") -> WorkflowPlan:
    """Run tests, capture output, open Grok with failure context so you (or autonomy) can iterate until passing.
    Topic = shell command to run (e.g. pytest, npm test, python -m pytest)."""
    cmd = (test_command or "pytest").strip()
    return WorkflowPlan(
        name="code_iteration",
        description=f"Run '{cmd}', capture output, open Grok with context for fix iteration.",
        steps=[
            TaskStep(
                action="run_command",
                args={"command": cmd, "timeout_seconds": 120},
                save_as="last_test_output",
                description=f"Run: {cmd}",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 2}, description="Brief pause"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "open_home", "params": {}},
                description="Open Grok",
                continue_on_error=True,
            ),
            TaskStep(
                action="clipboard_set",
                args={
                    "text": (
                        "My tests failed. Fix the code. Return only the corrected code or minimal patch.\n\n"
                        "Test output:\n{last_test_output}"
                    )
                },
                description="Put failure context in clipboard for Grok",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={
                    "message": (
                        "Grok is open; clipboard has the failure output. Paste (Ctrl+V) and get fix. "
                        "Apply the fix, then run this workflow again—or use Autonomous mode with the same test command to iterate until tests pass."
                    )
                },
                description="Log how to iterate",
            ),
        ],
    )


def portfolio_workflow(topic: str = "") -> WorkflowPlan:
    """Create a personal portfolio via Google AI Studio: sign-in, prompt Gemini, save HTML, optional git/vercel.
    Topic = optional short description (e.g. 'data scientist') for the portfolio prompt."""
    extra = (topic or "").strip()
    prompt_instruction = f" Focus on: {extra}." if extra else ""
    return WorkflowPlan(
        name="portfolio",
        description="Open Google AI Studio, sign in, get portfolio code from Gemini, save and optionally deploy.",
        steps=[
            TaskStep(
                action="request_human_input",
                args={
                    "prompt": "Enter your Google account password for AI Studio sign-in (value is used only in this run).",
                    "key": "human_input_password",
                },
                description="Request password for Google sign-in",
            ),
            TaskStep(
                action="open_url",
                args={"url": "https://aistudio.google.com"},
                description="Open Google AI Studio",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 8}, description="Let page load"),
            TaskStep(
                action="notify_user",
                args={
                    "message": "Sign in with your student email and the password you entered. When you see AI Studio home, the workflow will continue."
                },
                description="Notify user to sign in",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 60}, description="Time for user to sign in"),
            TaskStep(
                action="clipboard_set",
                args={
                    "text": (
                        "Create a single-file HTML personal portfolio page. "
                        "Include: hero section, about, skills, contact. Use modern CSS, responsive layout."
                        + prompt_instruction
                        + " Return only the full HTML document, no markdown."
                    )
                },
                description="Set portfolio prompt in clipboard for Gemini",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={
                    "message": "Paste the prompt in Google AI Studio (Gemini). When the code is generated, copy the full HTML to clipboard. Workflow will save it after 90s."
                },
                description="Guide user to paste in Gemini and copy result",
            ),
            TaskStep(action="wait", args={"seconds": 90}, description="Time for user to get code from Gemini"),
            TaskStep(
                action="clipboard_get",
                save_as="portfolio_html",
                description="Capture portfolio HTML from clipboard",
                continue_on_error=True,
            ),
            TaskStep(
                action="write_file",
                args={
                    "path": "{run_dir}/portfolio/index.html",
                    "text": "{portfolio_html}",
                },
                description="Save portfolio HTML to run folder",
                continue_on_error=True,
            ),
            TaskStep(
                action="notify_user",
                args={"message": "Portfolio saved to run folder. Look at the screen. You can push to GitHub and deploy to Vercel next."},
                description="Alert user that portfolio is ready",
                continue_on_error=True,
            ),
            TaskStep(
                action="run_command",
                args={
                    "command": "cd \"{run_dir}/portfolio\" && git init 2>nul || git init",
                    "timeout_seconds": 10,
                },
                continue_on_error=True,
                description="Init git in portfolio folder",
            ),
            TaskStep(
                action="run_command",
                args={
                    "command": "cd \"{run_dir}/portfolio\" && git add .",
                    "timeout_seconds": 10,
                },
                continue_on_error=True,
                description="Git add portfolio files",
            ),
            TaskStep(
                action="run_command",
                args={
                    "command": "cd \"{run_dir}/portfolio\" && git commit -m \"Add portfolio from Autobot\"",
                    "timeout_seconds": 10,
                },
                continue_on_error=True,
                description="Git commit portfolio",
            ),
            TaskStep(
                action="log",
                args={
                    "message": "To deploy: add remote (git remote add origin <url>), push (git push -u origin main), then run 'vercel' in the portfolio folder or connect repo on vercel.com."
                },
                description="Log next steps for GitHub and Vercel",
            ),
        ],
    )


def leetcode_solve_workflow(slug: str = "") -> WorkflowPlan:
    """Solve one LeetCode problem: open problem, copy statement, get code from Grok, paste, submit, read result.
    Topic = problem slug (e.g. two-sum, add-two-numbers)."""
    problem_slug = (slug or "two-sum").strip()
    return WorkflowPlan(
        name="leetcode_solve",
        description=f"Solve LeetCode problem '{problem_slug}' using Grok for code.",
        steps=[
            TaskStep(
                action="state_set",
                args={"key": "leetcode_problem_slug", "value": problem_slug},
                description="Store problem slug for later steps",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "leetcode_web", "adapter_action": "open_home", "params": {}},
                description="Open LeetCode",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "leetcode_web", "adapter_action": "open_problem", "params": {"slug": problem_slug}},
                description=f"Open problem {problem_slug}",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "leetcode_web", "adapter_action": "copy_problem_statement", "params": {}},
                description="Copy problem statement",
                continue_on_error=True,
            ),
            TaskStep(action="clipboard_get", save_as="problem_statement", description="Save statement to state", continue_on_error=True),
            TaskStep(
                action="clipboard_set",
                args={
                    "text": (
                        "Solve this LeetCode problem. Return only the runnable code in the required language, no explanation.\n\n"
                        "{problem_statement}"
                    )
                },
                description="Prepare prompt for Grok",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "open_home", "params": {}},
                description="Open Grok for code generation",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 5}, description="Let Grok load"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "ask_latex_from_clipboard", "params": {"instruction": "Solve the pasted LeetCode problem. Return only the code."}},
                description="Ask Grok for solution code",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 90}, description="Wait for Grok response"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "copy_visible_response", "params": {}},
                description="Copy Grok solution",
                continue_on_error=True,
            ),
            TaskStep(action="clipboard_get", save_as="solution_code", description="Save solution to state", continue_on_error=True),
            TaskStep(
                action="adapter_call",
                args={"adapter": "leetcode_web", "adapter_action": "open_problem", "params": {"slug": "{leetcode_problem_slug}"}},
                description="Return to LeetCode problem",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 4}, description="Let problem page load"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "leetcode_web", "adapter_action": "paste_code", "params": {}},
                description="Paste solution into editor",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "leetcode_web", "adapter_action": "submit", "params": {}},
                description="Submit solution",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 15}, description="Wait for submission result"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "leetcode_web", "adapter_action": "get_submission_result", "params": {}},
                save_as="submission_result",
                description="Read pass/fail result",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={"message": "LeetCode submission done. Check state['submission_result'] or the page. If failed, re-run with same slug after fixing code."},
                description="Log result",
            ),
        ],
    )


def kaggle_submit_workflow(competition_slug: str = "", wait_seconds: float = 1200) -> WorkflowPlan:
    """Open Kaggle competition, run notebook, submit. Long wait for scoring (default 20 min).
    Topic = competition slug (e.g. titanic). Use wait_seconds to allow multi-task switch during wait."""
    slug = (competition_slug or "titanic").strip()
    wait_s = max(60, min(3600, wait_seconds))  # 1 min to 1 hour
    return WorkflowPlan(
        name="kaggle_submit",
        description=f"Open Kaggle competition '{slug}', run notebook, submit, wait {wait_s}s for score.",
        steps=[
            TaskStep(
                action="adapter_call",
                args={"adapter": "kaggle_web", "adapter_action": "open_competition", "params": {"slug": slug}},
                description=f"Open competition {slug}",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 5}, description="Let competition page load"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "kaggle_web", "adapter_action": "open_my_notebooks", "params": {}},
                description="Open My Notebooks",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 5}, description="Let notebooks load"),
            TaskStep(
                action="log",
                args={"message": "Open your latest notebook (or create one). Workflow will run and submit after 10s."},
                description="Guide user",
            ),
            TaskStep(action="wait", args={"seconds": 10}, description="Time to open notebook"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "kaggle_web", "adapter_action": "run_notebook", "params": {}},
                description="Run notebook",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 60}, description="Notebook execution (adjust if needed)"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "kaggle_web", "adapter_action": "submit_to_competition", "params": {}},
                description="Submit to competition",
                continue_on_error=True,
            ),
            TaskStep(
                action="wait",
                args={"seconds": wait_s},
                description=f"Wait for Kaggle to score submission (~{wait_s // 60} min)",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "kaggle_web", "adapter_action": "read_leaderboard_status", "params": {}},
                save_as="kaggle_leaderboard_status",
                description="Read leaderboard status",
                continue_on_error=True,
            ),
            TaskStep(
                action="log",
                args={"message": "Kaggle submission wait done. Check leaderboard. While waiting, you can run other tasks (LeetCode, portfolio) in a separate run."},
                description="Log next steps",
            ),
        ],
    )


def open_whatsapp_stay_workflow(phone: str = "") -> WorkflowPlan:
    """Open WhatsApp Web and leave it open (no close steps). Topic = optional phone number."""
    phone = (phone or "").strip()
    if phone:
        steps = [
            TaskStep(action="browser_set_mode", args={"mode": "human_profile"}, description="Use human profile mode", continue_on_error=True),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "open_chat", "params": {"phone": phone}},
                description=f"Open WhatsApp chat: {phone}",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 2}, description="Let page load and stay open"),
        ]
    else:
        steps = [
            TaskStep(action="browser_set_mode", args={"mode": "human_profile"}, description="Use human profile mode", continue_on_error=True),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "open_home", "params": {}},
                description="Open WhatsApp Web home",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 2}, description="Let page load and stay open"),
        ]
    return WorkflowPlan(
        name="open_whatsapp_stay",
        description="Open WhatsApp Web and leave it open (no tab close).",
        steps=steps,
    )


def builtin_workflows() -> dict[str, WorkflowPlan]:
    return {
        "open_whatsapp_stay": open_whatsapp_stay_workflow(""),
        "website_builder": website_builder_workflow("new product"),
        "research_paper": research_paper_workflow("AI systems"),
        "portfolio": portfolio_workflow(""),
        "leetcode_solve": leetcode_solve_workflow("two-sum"),
        "kaggle_submit": kaggle_submit_workflow("titanic", 1200),
        "console_fix_assist": console_fix_assist_workflow(),
        "code_iteration": code_iteration_workflow("pytest"),
        "tool_call_stress": tool_call_stress_workflow(
            whatsapp_phone="",
            docs_existing_url="",
            download_check_path="",
            outgoing_message="Autobot tool-calling test message",
        ),
    }


def tool_call_stress_workflow(
    whatsapp_phone: str,
    docs_existing_url: str,
    download_check_path: str,
    outgoing_message: str,
    close_tabs_at_end: bool = False,
) -> WorkflowPlan:
    steps: list[TaskStep] = [
            TaskStep(action="browser_set_mode", args={"mode": "human_profile"}, description="Force human profile mode", continue_on_error=True),
            TaskStep(action="adapter_set_policy", args={"profile": "trusted"}, description="Set trusted policy for test", continue_on_error=True),
            TaskStep(
                action="state_set",
                args={"key": "whatsapp_message_sent", "value": outgoing_message},
                description="Record message for artifacts",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "open_chat", "params": {"phone": whatsapp_phone}},
                condition="true" if whatsapp_phone else "false",
                description="Open WhatsApp chat by phone",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "type_message", "params": {"text": outgoing_message}},
                condition="true" if whatsapp_phone else "false",
                description="Type WhatsApp test message",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "send_typed_message", "params": {}, "confirmed": True},
                condition="true" if whatsapp_phone else "false",
                description="Send WhatsApp test message",
                continue_on_error=True,
            ),
            TaskStep(
                action="screenshot",
                args={"filename": "01_whatsapp_sent.png"},
                condition="true" if whatsapp_phone else "false",
                continue_on_error=True,
                description="Screenshot after WhatsApp send",
            ),
            TaskStep(action="clipboard_set", args={"text": outgoing_message}, description="Store sent message in clipboard", continue_on_error=True),
            TaskStep(
                action="adapter_call",
                args={"adapter": "google_docs_web", "adapter_action": "open_new_document", "params": {}},
                description="Open new Google Doc",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "google_docs_web", "adapter_action": "type_text", "params": {"text": outgoing_message}},
                continue_on_error=True,
                description="Type copied WhatsApp text in Google Doc",
            ),
            TaskStep(
                action="screenshot",
                args={"filename": "02_google_doc_typed.png"},
                continue_on_error=True,
                description="Screenshot after Google Doc typed",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "google_docs_web", "adapter_action": "open_document_url", "params": {"url": docs_existing_url}},
                condition="true" if docs_existing_url else "false",
                continue_on_error=True,
                description="Open an existing Google Doc",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "google_docs_web", "adapter_action": "copy_all_text", "params": {}},
                continue_on_error=True,
                description="Copy all text from selected Google Doc",
            ),
            TaskStep(action="clipboard_get", save_as="doc_text", continue_on_error=True, description="Capture doc text"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "open_home", "params": {}},
                description="Open Grok",
                continue_on_error=True,
            ),
            TaskStep(
                action="clipboard_set",
                args={
                    "text": (
                        "Convert the following text into accurate LaTeX for Overleaf. "
                        "Return only LaTeX source.\n\n{doc_text}"
                    )
                },
                description="Prepare Grok latex prompt in clipboard",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "ask_latex_from_clipboard", "params": {}},
                description="Ask Grok for latex conversion",
                continue_on_error=True,
            ),
            TaskStep(action="wait", args={"seconds": 8}, description="Wait for Grok response"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "copy_visible_response", "params": {}},
                continue_on_error=True,
                description="Copy latex response from Grok",
            ),
            TaskStep(action="clipboard_get", save_as="latex_text", continue_on_error=True, description="Capture latex text"),
            TaskStep(
                action="screenshot",
                args={"filename": "03_grok_response.png"},
                continue_on_error=True,
                description="Screenshot after Grok response",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "overleaf_web", "adapter_action": "open_dashboard", "params": {}},
                description="Open Overleaf",
                continue_on_error=True,
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "overleaf_web", "adapter_action": "replace_editor_text", "params": {"text": "{latex_text}"}},
                continue_on_error=True,
                description="Paste latex into Overleaf",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "overleaf_web", "adapter_action": "compile_project", "params": {}},
                continue_on_error=True,
                description="Compile Overleaf document",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "overleaf_web", "adapter_action": "download_pdf", "params": {}, "confirmed": True},
                continue_on_error=True,
                description="Download compiled PDF",
            ),
            TaskStep(
                action="wait_for_file",
                args={"path": download_check_path, "timeout_seconds": 45, "poll_interval_seconds": 2},
                save_as="pdf_downloaded",
                condition="true" if download_check_path else "false",
                continue_on_error=True,
                description="Verify download file appears",
            ),
            TaskStep(
                action="screenshot",
                args={"filename": "04_overleaf_download.png"},
                continue_on_error=True,
                description="Screenshot after Overleaf download",
            ),
        ]
    if close_tabs_at_end:
        steps.extend([
            TaskStep(
                action="desktop_hotkey",
                args={"keys": ["ctrl", "w"]},
                retries=2,
                continue_on_error=True,
                description="Close active tab (cleanup)",
            ),
            TaskStep(
                action="desktop_hotkey",
                args={"keys": ["ctrl", "w"]},
                retries=2,
                continue_on_error=True,
                description="Close active tab (cleanup)",
            ),
            TaskStep(
                action="desktop_hotkey",
                args={"keys": ["ctrl", "w"]},
                retries=2,
                continue_on_error=True,
                description="Close active tab (cleanup)",
            ),
            TaskStep(
                action="screenshot",
                args={"filename": "05_final.png"},
                continue_on_error=True,
                description="Final screenshot",
            ),
        ])
    steps.append(
        TaskStep(
            action="log",
            args={
                "message": (
                    "Tool-call stress test complete. Check run folder for history.json, screenshots/, and artifacts.json."
                )
            },
            description="Summarize stress test completion",
        ),
    )
    return WorkflowPlan(
        name="tool_call_stress",
        description="Stress-test chained tool calling from WhatsApp -> Docs -> Grok -> Overleaf -> download."
        + (" Cleanup: close tabs at end." if close_tabs_at_end else " Tabs left open."),
        steps=steps,
    )

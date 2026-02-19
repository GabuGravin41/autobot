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
    if normalized in {"vscode", "vs code", "code"}:
        return WorkflowPlan(
            name="open_vscode",
            description="Open Visual Studio Code.",
            steps=[TaskStep(action="open_vscode", description="Open VS Code")],
        )
    if normalized == "overleaf":
        target = "https://www.overleaf.com"
    if normalized in {"casa", "casa ai"}:
        target = "https://www.casa.ai"
    if normalized in {"grok"}:
        target = "https://grok.com"
    if normalized in {"deepseek"}:
        target = "https://chat.deepseek.com"

    return WorkflowPlan(
        name="open_target",
        description=f"Open target: {target}",
        steps=[TaskStep(action="open_url", args={"url": target}, description=f"Open {target}")],
    )


def website_builder_workflow(topic: str) -> WorkflowPlan:
    topic = topic.strip() or "new product"
    return WorkflowPlan(
        name="website_builder",
        description="Open your coding stack and prep a website build flow.",
        steps=[
            TaskStep(action="open_vscode", description="Open VS Code"),
            TaskStep(action="open_url", args={"url": "https://www.casa.ai"}, description="Open CASA AI"),
            TaskStep(action="open_url", args={"url": "https://grok.com"}, description="Open Grok"),
            TaskStep(
                action="search_google",
                args={"query": f"best modern website layout ideas for {topic}"},
                description="Gather design references",
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
    return WorkflowPlan(
        name="research_paper",
        description="Open research tools and prep paper draft flow.",
        steps=[
            TaskStep(action="open_url", args={"url": "https://grok.com"}, description="Open Grok"),
            TaskStep(action="open_url", args={"url": "https://chat.deepseek.com"}, description="Open DeepSeek"),
            TaskStep(action="open_url", args={"url": "https://www.overleaf.com"}, description="Open Overleaf"),
            TaskStep(
                action="search_google",
                args={"query": f"latest peer-reviewed references on {topic}"},
                description="Collect supporting references",
            ),
            TaskStep(
                action="log",
                args={
                    "message": (
                        "Research stack is open. Prompt an LLM for LaTeX draft and paste into Overleaf for compilation."
                    )
                },
                description="Log paper drafting guidance",
            ),
        ],
    )


def console_fix_assist_workflow(local_url: str = "http://localhost:3000") -> WorkflowPlan:
    return WorkflowPlan(
        name="console_fix_assist",
        description="Open local app and gather console diagnostics.",
        steps=[
            TaskStep(action="open_url", args={"url": local_url}, description="Open local app"),
            TaskStep(action="wait", args={"seconds": 2}, description="Wait for app to settle"),
            TaskStep(
                action="browser_read_console_errors",
                save_as="console_errors",
                description="Capture browser console-like errors",
            ),
            TaskStep(
                action="clipboard_set",
                args={"text": "{console_errors}"},
                description="Copy captured errors to clipboard",
            ),
            TaskStep(
                action="log",
                args={"message": "Console errors copied to clipboard. Paste into your coding assistant to request a fix."},
                description="Log next action guidance",
            ),
        ],
    )


def builtin_workflows() -> dict[str, WorkflowPlan]:
    return {
        "website_builder": website_builder_workflow("new product"),
        "research_paper": research_paper_workflow("AI systems"),
        "console_fix_assist": console_fix_assist_workflow(),
    }

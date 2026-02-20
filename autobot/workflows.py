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


def open_whatsapp_stay_workflow(phone: str = "") -> WorkflowPlan:
    """Open WhatsApp Web and leave it open (no close steps). Topic = optional phone number."""
    phone = (phone or "").strip()
    if phone:
        steps = [
            TaskStep(action="browser_set_mode", args={"mode": "human_profile"}, description="Use human profile mode"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "open_chat", "params": {"phone": phone}},
                description=f"Open WhatsApp chat: {phone}",
            ),
            TaskStep(action="wait", args={"seconds": 2}, description="Let page load and stay open"),
        ]
    else:
        steps = [
            TaskStep(action="browser_set_mode", args={"mode": "human_profile"}, description="Use human profile mode"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "open_home", "params": {}},
                description="Open WhatsApp Web home",
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
        "console_fix_assist": console_fix_assist_workflow(),
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
            TaskStep(action="browser_set_mode", args={"mode": "human_profile"}, description="Force human profile mode"),
            TaskStep(action="adapter_set_policy", args={"profile": "trusted"}, description="Set trusted policy for test"),
            TaskStep(
                action="state_set",
                args={"key": "whatsapp_message_sent", "value": outgoing_message},
                description="Record message for artifacts",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "open_chat", "params": {"phone": whatsapp_phone}},
                condition="true" if whatsapp_phone else "false",
                description="Open WhatsApp chat by phone",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "type_message", "params": {"text": outgoing_message}},
                condition="true" if whatsapp_phone else "false",
                description="Type WhatsApp test message",
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "whatsapp_web", "adapter_action": "send_typed_message", "params": {}, "confirmed": True},
                condition="true" if whatsapp_phone else "false",
                description="Send WhatsApp test message",
            ),
            TaskStep(
                action="screenshot",
                args={"filename": "01_whatsapp_sent.png"},
                condition="true" if whatsapp_phone else "false",
                continue_on_error=True,
                description="Screenshot after WhatsApp send",
            ),
            TaskStep(action="clipboard_set", args={"text": outgoing_message}, description="Store sent message in clipboard"),
            TaskStep(
                action="adapter_call",
                args={"adapter": "google_docs_web", "adapter_action": "open_new_document", "params": {}},
                description="Open new Google Doc",
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
            ),
            TaskStep(
                action="adapter_call",
                args={"adapter": "grok_web", "adapter_action": "ask_latex_from_clipboard", "params": {}},
                description="Ask Grok for latex conversion",
            ),
            TaskStep(action="wait", args={"seconds": 8}, continue_on_error=True, description="Wait for Grok response"),
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

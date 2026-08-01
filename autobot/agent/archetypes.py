"""
Task Archetype Library — Procedure templates for common task patterns.

When the agent's goal matches a known task archetype (login, file download,
form submission, etc.), the procedure is injected as first-step context.
This gives the agent accumulated human knowledge about how tasks of this type
are typically solved — enabling transfer from task-type to novel instances.

ARC-AGI connection: this is the "few-shot examples" mechanism. ARC-AGI gives
solvers training demonstrations of the task category. Archetypes give the agent
proven procedure examples for the task type. Both enable generalization from
pattern to novel instance on a domain the agent has never visited before.
"""
from __future__ import annotations

ARCHETYPES: dict[str, dict] = {
    "login": {
        "keywords": ["log in", "sign in", "login", "authenticate", "credentials", "my account"],
        "procedure": [
            "1. Take a screenshot to locate the login form.",
            "2. Find the username/email input field via the DOM snapshot (look for type=email or type=text near a password field).",
            "3. Check for SSO/OAuth buttons ('Continue with Google', 'Sign in with Apple') — prefer these over manual entry.",
            "4. Click the username field to focus it. Press Ctrl+A to select all, then Delete to clear. Type the username.",
            "5. Tab to or click the password field. Clear it the same way, then type the password.",
            "6. Click the Submit/Login button or press Enter.",
            "7. Wait 3 seconds and take a screenshot. Verify the URL changed or a welcome message appeared.",
            "8. If a CAPTCHA or 2FA prompt appears, pause and write REMEMBER:needs_human_input=true in your memory field.",
        ],
    },
    "file_download": {
        "keywords": ["download", "save file", "export", "get file", "fetch file", "save as"],
        "procedure": [
            "1. Navigate to the page containing the file.",
            "2. Find the download link or button via the DOM snapshot.",
            "3. If a direct URL is available, prefer navigate(url) over clicking (more reliable).",
            "4. After clicking the download trigger, check for a browser download bar or Save dialog.",
            "5. If a Save dialog appears: use computer.keyboard.type('/path/to/save') then press Enter.",
            "6. Verify the download completed: terminal.run('ls -lt ~/Downloads | head -5')",
            "7. Confirm the file exists and has non-zero size before calling done().",
        ],
    },
    "search_and_click": {
        "keywords": ["search for", "find", "look up", "look for", "google", "search", "find out"],
        "procedure": [
            "1. Navigate to the search engine or target site.",
            "2. Locate the search input field via the DOM snapshot.",
            "3. Click to focus, press Ctrl+A to clear, type the query, then press Enter.",
            "4. Wait 2 seconds for results to load, then take a screenshot.",
            "5. Scan the DOM snapshot for the most relevant result link.",
            "6. Click or navigate to the URL of the chosen result.",
            "7. Verify the target page loaded correctly before continuing.",
        ],
    },
    "form_submission": {
        "keywords": ["fill", "submit", "complete form", "fill out", "fill in", "enter data", "apply", "register", "sign up"],
        "procedure": [
            "1. Take a screenshot to identify the full form structure.",
            "2. Use the DOM snapshot to list all input fields with their indices and labels.",
            "3. Fill fields in visual order (top to bottom). For each: click to focus, Ctrl+A to select all, Delete to clear, then type.",
            "4. For dropdowns: click to open, wait for options to appear (1 step), then click the correct option.",
            "5. For checkboxes and radio buttons: click once and verify the checked state in the next screenshot.",
            "6. Before submitting, check for required-field validation indicators (red borders, asterisks, error messages).",
            "7. Click Submit and wait 3 seconds. Verify success: URL changed, confirmation message visible, no error banner.",
        ],
    },
    "code_execution": {
        "keywords": ["run code", "execute", "terminal", "script", "compile", "test", "solve coding", "code solution", "run script"],
        "procedure": [
            "1. Open a terminal: use computer.terminal.run() for shell commands, or Ctrl+Alt+T to open a terminal window.",
            "2. Navigate to the correct directory before running: terminal.run('cd /path/to/project')",
            "3. Run the command and capture stdout/stderr.",
            "4. Check the exit code and output for errors before proceeding — do not retry without reading the error.",
            "5. For web-based code editors (LeetCode, Codeforces, Codesignal): use the keyboard shortcut to run (usually Ctrl+Enter or Ctrl+Shift+Enter), not the mouse button.",
            "6. For submission: verify the output matches expected results before submitting.",
        ],
    },
    "data_extraction": {
        "keywords": ["extract", "scrape", "collect", "gather data", "copy all", "get list", "get all", "pull data"],
        "procedure": [
            "1. Navigate to the target page containing the data.",
            "2. Use the DOM snapshot to identify the data structure (table, list, repeated elements).",
            "3. For small datasets: select all text (Ctrl+A), copy (Ctrl+C), paste into a terminal to inspect.",
            "4. For tables: use terminal.run() with a Python/curl command for reliable bulk extraction.",
            "5. Handle pagination: check for 'Next' buttons or page number inputs in the DOM snapshot.",
            "6. Store extracted data in a temporary file as a backup: terminal.run('echo \"data\" > ~/Desktop/extracted.txt')",
            "7. Verify completeness (row count, expected fields) before calling done().",
        ],
    },
    "file_upload": {
        "keywords": ["upload", "attach file", "attach", "submit file", "send file", "upload file"],
        "procedure": [
            "1. Verify the file exists at the expected path: terminal.run('ls -la /path/to/file')",
            "2. Locate the upload button or file input element via the DOM snapshot.",
            "3. Click the upload input — an OS file dialog will open.",
            "4. In the file dialog: use computer.keyboard.type('/absolute/path/to/file') to type the full path directly.",
            "5. Press Enter to confirm the path.",
            "6. Wait 3 seconds and take a screenshot. Verify the file name appears on the page.",
            "7. If there is a separate 'Upload' or 'Confirm' button after file selection, click it.",
        ],
    },
    "navigation": {
        "keywords": ["navigate to", "go to", "open", "visit", "browse to", "open the website", "go to the website"],
        "procedure": [
            "1. Use the navigate() action with the full URL — never type in the address bar.",
            "2. Wait for the page to load (2-3 seconds or use wait() if slow).",
            "3. Take a screenshot to confirm you arrived at the correct page.",
            "4. If redirected unexpectedly (login wall, paywall, 404), handle the obstacle before continuing.",
        ],
    },
}


class TaskArchetypeLibrary:
    """
    Matches a task goal to its most relevant archetype and formats a procedure
    guide for injection into the agent's first-step context.

    Scoring: keyword overlap, minimum score of 1 to match.
    Tie-breaking: longer keyword phrases matched first (more specific wins).
    """

    def match(self, goal: str) -> dict | None:
        """
        Return the best-matching archetype dict, or None if no match.

        Returns a dict with keys:
          - name: str
          - procedure: list[str]
        """
        goal_lower = goal.lower()
        best_name: str | None = None
        best_score = 0
        best_specificity = 0  # tie-break: total chars of matched keywords

        for name, archetype in ARCHETYPES.items():
            score = 0
            specificity = 0
            for kw in archetype["keywords"]:
                if kw in goal_lower:
                    score += 1
                    specificity += len(kw)
            if score > best_score or (score == best_score and specificity > best_specificity):
                best_score = score
                best_specificity = specificity
                best_name = name

        if best_name is None or best_score == 0:
            return None

        archetype = ARCHETYPES[best_name]
        return {
            "name": best_name,
            "procedure": archetype["procedure"],
        }

    def format_for_prompt(self, archetype: dict) -> str:
        """Format an archetype as a first-step context injection (XML block)."""
        steps = "\n".join(archetype["procedure"])
        return (
            f"<archetype_guide name='{archetype['name']}'>\n"
            f"This task matches the '{archetype['name']}' archetype. "
            f"Here is a proven procedure for this type of task:\n\n"
            f"{steps}\n\n"
            f"Adapt this procedure to the specific context you observe on screen. "
            f"Do not follow it blindly — always verify against the actual page state.\n"
            f"</archetype_guide>"
        )


# Module-level singleton
archetype_library = TaskArchetypeLibrary()

# Autobot Build Roadmap (Execution Queue)

This is the prioritized build queue to continue polishing Autobot while you are away.

## Phase 1 - Reliability Hardening (today)

1. Add adapter telemetry: [implemented]
   - action duration
   - selector success/failure rates
   - failure snapshots (URL + HTML snippet + optional screenshot path)
2. Add resilient selector strategy: [implemented]
   - primary selectors + fallback selectors
   - per-site selector config files
3. Add session health checks: [implemented]
   - detect login-expired pages
   - auto-trigger `attempt_google_continue_login`
4. Add structured run history: [implemented]
   - write JSON logs to `runs/<timestamp>.json`
   - include step inputs, outputs, and errors

## Phase 2 - Safe Action Control

1. Add two-step confirmation for risky actions: [implemented]
   - `prepare_send` -> preview -> `confirm_send`
2. Add policy profiles: [implemented]
   - `strict`, `balanced`, `trusted`
3. Add denylist rules:
   - block bulk messaging patterns
   - block repeated sends within short intervals
4. Add emergency stop:
   - global hotkey to immediately cancel pyautogui activity

## Phase 3 - Deep App Capabilities

1. WhatsApp Web:
   - contact search quality improvements [partial: human-nav map + phone-route open]
   - media/file send workflow
   - unread message digest extraction
2. Instagram:
   - robust DM thread open [partial: human-nav map]
   - read recent thread context
   - post/comment navigation primitives
3. Overleaf:
   - project create
   - section-level replace in editor [partial: human-nav map]
   - compile status parsing and error extraction
4. VS Code desktop routines:
   - deterministic file edits from adapter actions
   - terminal pane state checks

## Phase 4 - Autonomous Intelligence

1. Planner memory:
   - preserve successful action sequences per site
2. Multi-provider LLM routing:
   - route long context jobs to Grok-compatible endpoint
   - short control loops to lower-cost provider
3. Goal decomposition:
   - break large goals into staged sub-goals
4. Verification loops:
   - objective checks after each loop
   - stop conditions based on measurable completion

## Phase 5 - Desktop Breadth

1. File Explorer adapter
2. Excel routines (open workbook, cell write, save)
3. Word routines (open doc, insert text, export PDF)
4. Cross-app transfer flows:
   - browser -> clipboard -> Office apps
   - Office -> browser upload

## Testing Strategy

1. Add smoke tests for planner and adapter registry. [implemented]
2. Add adapter integration tests with mocked browser pages.
3. Add dry-run mode for all adapter actions.
4. Add end-to-end script for:
   - open target
   - execute workflow
   - verify expected state markers

/**
 * apiService.ts — All calls to the Autobot FastAPI backend.
 * 
 * Replaces geminiService.ts (which called Gemini directly from the browser).
 * The backend now handles all AI calls via /api/chat.
 */

// In dev: Vite proxy forwards /api → http://127.0.0.1:8000 so BASE_URL = ''
// In production: React and FastAPI are on the same origin so BASE_URL = ''
// Override with VITE_API_URL only if explicitly set (e.g. for a remote backend)
import { API_BASE, WS_BASE } from '../config';

const BASE_URL = API_BASE;


/** Generic fetch helper with JSON body and 15s timeout */
async function apiFetch<T>(path: string, options?: RequestInit & { timeoutMs?: number }): Promise<T> {
    const timeoutMs = options?.timeoutMs ?? 15_000;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const res = await fetch(`${BASE_URL}${path}`, {
            headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
            signal: controller.signal,
            ...options,
        });
        if (!res.ok) {
            const err = await res.text();
            throw new Error(`API Error ${res.status}: ${err}`);
        }
        return res.json();
    } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') {
            throw new Error(`Request to ${path} timed out after ${timeoutMs}ms`);
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

// ── Status ─────────────────────────────────────────────────────────────────
export interface HumanInputPending {
    prompt: string;
    key: string;
}

export interface AuthNotification {
    url: string;
    type: string;
    message: string;
}

export interface ApprovalPending {
    key: string;
    message: string;
}

export interface BackendStatus {
    status: string;
    run_status: 'idle' | 'running' | 'done' | 'failed' | 'cancelled';
    active_run_id: string | null;
    screenshot_b64: string | null;
    browser: { active: boolean; mode: string; url: string };
    llm_enabled: boolean;
    llm_provider: string;
    llm_model: string;
    human_input_pending?: HumanInputPending | null;
    human_approval_pending?: ApprovalPending | null;
    anti_sleep_enabled?: boolean;
    auth_notification?: AuthNotification | null;
    narrative?: string | null;
    paused?: boolean;
}

// ── Onboarding ──────────────────────────────────────────────────────────────
export interface OnboardingData {
    name?: string;
    kaggle_username?: string;
    editor?: string;
    ai_tools?: string;
    language?: string;
}

export const submitOnboarding = (data: OnboardingData): Promise<{ status: string; saved: string[] }> =>
    apiFetch('/api/onboarding', { method: 'POST', body: JSON.stringify(data) });

export const getOnboardingStatus = (): Promise<{ complete: boolean }> =>
    apiFetch('/api/onboarding/status');

export const getStatus = (): Promise<BackendStatus> =>
    apiFetch<BackendStatus>('/api/status');

// ── Adapters ────────────────────────────────────────────────────────────────
export interface BackendAdapter {
    name: string;
    description: string;
    actions: string[];
    telemetry: Record<string, any>;
}

export const getAdapters = (): Promise<{ adapters: BackendAdapter[] }> =>
    apiFetch('/api/adapters');

// ── Runs ────────────────────────────────────────────────────────────────────
export interface BackendRun {
    id: string;
    name?: string;
    planName?: string;
    status: 'success' | 'failed' | 'running' | 'cancelled';
    timestamp: string;
    stepsCompleted: number;
    totalSteps: number;
    logs?: string[];
    artifacts?: Record<string, any>;
}

export const getRuns = (): Promise<{ runs: BackendRun[] }> =>
    apiFetch('/api/runs');

export const getRun = (runId: string): Promise<BackendRun> =>
    apiFetch(`/api/run/${runId}`);

export const deleteRun = (runId: string): Promise<{ status: string }> =>
    apiFetch(`/api/run/${runId}`, { method: 'DELETE' });

// ── Planning ────────────────────────────────────────────────────────────────
export interface PlanStep {
    action: string;
    args: Record<string, any>;
    description: string;
    save_as?: string;
    retries?: number;
    continue_on_error?: boolean;
    status?: 'pending' | 'running' | 'completed' | 'failed';
}

export interface ScheduledTask {
    id: string;
    name: string;
    description: string;
    steps: PlanStep[];
}

export interface BackendPlan {
    id: string;
    name: string;
    description: string;
    steps: PlanStep[];
}

/** Convert a natural-language task into a WorkflowPlan via the backend */
export const planFromText = (task: string): Promise<{ plan: BackendPlan }> =>
    apiFetch('/api/plan/text', { method: 'POST', body: JSON.stringify({ task }) });

/** Execute a plan. Now routed to the NEW Agent architecture. */
export const runPlan = (plan: BackendPlan): Promise<{ run_id: string; status: string }> => {
    // Include the full step list so the agent has complete context, not just a one-liner.
    const stepsText = plan.steps?.length
        ? '\n\nStep-by-step plan:\n' + plan.steps.map((s, i) => `${i + 1}. ${s.description}`).join('\n')
        : '';
    const goal = (plan.description || plan.name || 'Execute plan') + stepsText;
    return apiFetch('/api/agent/run', { method: 'POST', body: JSON.stringify({ goal }) });
};

/** Cancel a running plan */
export const cancelRun = (runId: string): Promise<{ status: string }> =>
    apiFetch(`/api/agent/cancel`, { method: 'POST' });

/** Pause the active agent run (idles after the current step) */
export const pauseRun = (): Promise<{ status: string }> =>
    apiFetch('/api/agent/pause', { method: 'POST' });

/** Resume a paused agent run */
export const resumeRun = (): Promise<{ status: string }> =>
    apiFetch('/api/agent/resume', { method: 'POST' });

// ── Autonomous Multi-Agent Runner ───────────────────────────────────────────
export interface AutonomousStatus {
    status: 'idle' | 'running' | 'done' | 'cancelled' | 'failed';
    goal: string;
    phase: string;
    loops: number;
    last_log: string;
}

/** Now routed to the NEW Agent architecture. */
export const runAutonomous = (goal: string, pageContext?: { url: string, title: string, text: string }): Promise<{ status: string; goal: string }> =>
    apiFetch('/api/agent/run', {
        method: 'POST',
        body: JSON.stringify({ goal, use_vision: true }),
    });

export const getAutonomousStatus = (): Promise<AutonomousStatus> =>
    apiFetch('/api/agent/status');

export const cancelAutonomous = (): Promise<{ status: string }> =>
    apiFetch('/api/agent/cancel', { method: 'POST' });

// ── Mission Mode (Multi-Objective Tasks) ─────────────────────────────────────
/** Start a multi-objective mission. Best for complex tasks like Kaggle competitions, research, multi-app coding. */
export const runMission = (goal: string): Promise<{ run_id: string; status: string; goal: string; mode: string }> =>
    apiFetch('/api/mission/run', {
        method: 'POST',
        body: JSON.stringify({ goal }),
    });

/**
 * Start an orchestrated multi-agent run.
 * Automatically routes simple tasks to a single AgentLoop, and complex tasks
 * to specialized sub-agents (WebNavigator, CodeExecutor, DataExtractor, etc.)
 */
export const runOrchestrated = (goal: string): Promise<{ run_id: string; status: string; goal: string; mode: string }> =>
    apiFetch('/api/orchestrate', {
        method: 'POST',
        body: JSON.stringify({ goal }),
    });

// ── Learning / RL Stats ──────────────────────────────────────────────────────
export interface LearningStats {
    rl_enabled: boolean;
    total_experiences: number;
    learned_contexts: number;
    total_policy_observations: number;
    current_run_steps: number;
    run_id: string;
    memory_entries?: number;
    memory_hits?: number;
    memory_high_value?: number;
    error?: string;
}

/** Get RL pipeline stats — how much the agent has learned from past runs. */
export const getLearningStats = (): Promise<LearningStats> =>
    apiFetch<LearningStats>('/api/learning/stats');

// ── Memory API ──────────────────────────────────────────────────────────────
export interface MemoryEntry {
    key: string;
    value: string;
}

export const getMemoryEntries = (): Promise<{ entries: MemoryEntry[]; total: number }> =>
    apiFetch('/api/memory/entries');

export const pruneMemory = (): Promise<{ removed: number; remaining: number; stats: Record<string, number> }> =>
    apiFetch('/api/memory/prune', { method: 'POST' });

export const deleteMemoryEntry = (key: string): Promise<{ deleted: string }> =>
    apiFetch(`/api/memory/entry/${encodeURIComponent(key)}`, { method: 'DELETE' });

/** Start the LeetCode multi-AI solving mission. */
export const runLeetCodeMission = (
    num_problems: number = 5,
    language: string = 'python3',
): Promise<{ run_id: string; status: string; goal: string; mode: string }> =>
    apiFetch('/api/mission/leetcode', {
        method: 'POST',
        body: JSON.stringify({ num_problems, language }),
        timeoutMs: 30_000,
    });

// ── AI Chat ─────────────────────────────────────────────────────────────────
export interface ChatResponse {
    reply: string;
    plan: BackendPlan | null;
}

/**
 * Send a chat message to the Autobot AI planner.
 * Supports multi-turn conversation by passing message history.
 */
export const sendChat = (
    message: string,
    state?: Record<string, any>,
    history?: Array<{ role: string; content: string }>,
): Promise<ChatResponse> =>
    apiFetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ message, state: state || {}, history: history || [] }),
        timeoutMs: 300_000,  // 5 min — slow local models (Ollama) can take 2-3min
    });

// Alias for compatibility with any code that imported generatePlan
export const generatePlan = async (prompt: string): Promise<BackendPlan> => {
    const res = await sendChat(prompt);
    if (res.plan) return res.plan;
    throw new Error(res.reply || 'No plan generated');
};

// ── Settings ────────────────────────────────────────────────────────────────
export interface BackendSettings {
    llm_provider: string;
    llm_model: string;
    browser_mode: string;
    has_openrouter_key: boolean;
    has_openai_key: boolean;
    has_google_key: boolean;
    has_xai_key: boolean;
    has_vertex_key: boolean;
    approval_mode: 'strict' | 'balanced' | 'trusted';
    using_default_key: boolean;
}

export const getSettings = (): Promise<BackendSettings> =>
    apiFetch('/api/settings');

export const updateSettings = (settings: Partial<{
    llm_provider: string;
    llm_model: string;
    browser_mode: string;
    openrouter_api_key: string;
    openai_api_key: string;
    google_api_key: string;
    vertex_api_key: string;
    xai_api_key: string;
    approval_mode: string;
}>): Promise<{ status: string; keys_changed: string[] }> =>
    apiFetch('/api/settings', { method: 'POST', body: JSON.stringify(settings) });

// ── Logs (polling fallback when WebSocket is unavailable) ────────────────────
export interface LogsResponse {
    logs: string[];
}

export const getLogs = (limit: number = 500): Promise<LogsResponse> =>
    apiFetch(`/api/logs?limit=${limit}`).then((r: any) => ({ logs: r.logs || [] }));

// ── WebSocket event stream — unified real-time sync for all clients ──────────
//
// Connects to /ws/events which pushes JSON events whenever anything changes.
// All clients (laptop, phone, tablet) see the same events at the same time.
//
// Event shapes:
//   { type: "snapshot",  run_status, active_run_id, screenshot_ts, logs, narrative, step }
//   { type: "status",    run_status, active_run_id }
//   { type: "log",       seq, line }
//   { type: "screenshot",ts }
//   { type: "narrative", text, step }
//   { type: "ping" }

export interface AutobotEvent {
    type: 'snapshot' | 'status' | 'log' | 'screenshot' | 'narrative' | 'ping';
    run_status?: string;
    active_run_id?: string | null;
    screenshot_ts?: number;
    ts?: number;
    logs?: string[];
    narrative?: string;
    text?: string;
    step?: number;
    seq?: number;
    line?: string;
}

export function connectEventStream(
    onEvent: (event: AutobotEvent) => void,
    onClose?: () => void,
): () => void {
    let closed = false;
    let reconnectAttempt = 0;
    const wsUrl = (WS_BASE || 'ws://127.0.0.1:8000').replace(/\/?$/, '') + '/ws/events';

    function tryConnect() {
        if (closed) return;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            reconnectAttempt = 0;
        };

        ws.onmessage = (e) => {
            try {
                const event = JSON.parse(e.data) as AutobotEvent;
                if (event.type !== 'ping') onEvent(event);
            } catch { /* ignore malformed */ }
        };

        ws.onerror = () => ws.close();

        ws.onclose = () => {
            onClose?.();
            if (closed) return;
            reconnectAttempt++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempt - 1), 10000);
            setTimeout(tryConnect, delay);
        };
    }

    tryConnect();
    return () => { closed = true; };
}

// ── WebSocket log streaming (with exponential backoff and optional polling fallback) ─
const WS_RECONNECT_BASE_MS = 1000;
const WS_RECONNECT_MAX_MS = 10000;  // max 10s between retries (was 30s)
const WS_MAX_RECONNECT_ATTEMPTS = 15;

export function connectLogStream(
    onMessage: (line: string) => void,
    onClose?: () => void,
    options?: { usePollingFallback?: boolean; onLogsSnapshot?: (logs: string[]) => void },
): () => void {
    let closed = false;
    let reconnectAttempt = 0;
    let pollInterval: ReturnType<typeof setInterval> | null = null;

    const wsUrl = (WS_BASE || 'ws://127.0.0.1:8000').replace(/\/?$/, '') + '/ws/logs';

    function scheduleReconnect() {
        if (closed) return;
        reconnectAttempt += 1;
        if (options?.usePollingFallback && reconnectAttempt >= 5) {
            if (!pollInterval && (options.onLogsSnapshot || onMessage)) {
                pollInterval = setInterval(async () => {
                    if (closed) return;
                    try {
                        const { logs } = await getLogs(300);
                        if (options.onLogsSnapshot) options.onLogsSnapshot(logs);
                        else logs.forEach((line) => onMessage(line));
                    } catch { /* ignore */ }
                }, 3000);
            }
            return;
        }
        const delay = Math.min(WS_RECONNECT_BASE_MS * Math.pow(2, reconnectAttempt - 1), WS_RECONNECT_MAX_MS);
        setTimeout(tryConnect, delay);
    }

    // Track highest sequence number seen to deduplicate messages from multiple WS connections
    let highestSeq = -1;
    // Track seen content to deduplicate historical replays (React StrictMode opens 2 connections)
    const seenContent = new Set<string>();

    function tryConnect() {
        if (closed) return;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (e) => {
            if (e.data === '__ping__') return;
            const raw: string = e.data;
            // Messages are formatted as "seq|content" or "hN|content" (historical)
            const pipeIdx = raw.indexOf('|');
            if (pipeIdx > 0) {
                const prefix = raw.substring(0, pipeIdx);
                const content = raw.substring(pipeIdx + 1);
                if (prefix.startsWith('h')) {
                    // Historical replay — deduplicate by content
                    if (seenContent.has(content)) return;
                    seenContent.add(content);
                    // Cap set size to prevent memory growth on very long runs
                    if (seenContent.size > 2000) seenContent.clear();
                    onMessage(content);
                } else {
                    const seq = parseInt(prefix, 10);
                    if (!isNaN(seq)) {
                        if (seq <= highestSeq) return; // Duplicate — skip
                        highestSeq = seq;
                        seenContent.add(content); // Also track live msgs for cross-dedup
                        onMessage(content);
                    } else {
                        onMessage(raw); // Unknown format — pass through
                    }
                }
            } else {
                onMessage(raw); // No prefix — pass through
            }
        };

        ws.onerror = () => {
            ws.close();
        };

        ws.onclose = () => {
            onClose?.();
            if (!closed) scheduleReconnect();
        };

        ws.onopen = () => {
            reconnectAttempt = 0;
        };
    }

    tryConnect();

    return () => {
        closed = true;
        if (pollInterval) clearInterval(pollInterval);
        // Note: we don't have a ref to the current WebSocket to close it; next reconnect will be no-op due to closed
    };
}

// ── Workflows ───────────────────────────────────────────────────────────────
export interface BackendWorkflow {
    id: string;
    name: string;
    description: string;
    topic_label: string;
    goal?: string;
    source?: 'builtin' | 'user';
    created_at?: string;
}

export const getWorkflows = (): Promise<{ workflows: BackendWorkflow[] }> =>
    apiFetch('/api/workflows');

export const runWorkflow = (workflow_id: string, topic: string = ''): Promise<{ run_id: string; status: string; plan_name: string }> =>
    apiFetch('/api/workflows/run', { method: 'POST', body: JSON.stringify({ workflow_id, topic }) });

export const saveWorkflow = (payload: {
    name: string;
    description: string;
    goal: string;
    topic_label?: string;
}): Promise<{ status: string; workflow: BackendWorkflow }> =>
    apiFetch('/api/workflows/save', { method: 'POST', body: JSON.stringify(payload) });

export const deleteWorkflow = (workflow_id: string): Promise<{ status: string }> =>
    apiFetch(`/api/workflows/${workflow_id}`, { method: 'DELETE' });

// ── Human input (when a run requests password/token) ────────────────────────
export const getPendingHumanInput = (): Promise<{ pending: boolean; prompt?: string; key?: string }> =>
    apiFetch('/api/human_input');

export const submitHumanInput = (key: string, value: string): Promise<{ status: string; key: string }> =>
    apiFetch('/api/human_input', { method: 'POST', body: JSON.stringify({ key, response: value }) });

// ── Anti-Sleep ──────────────────────────────────────────────────────────────
export const toggleAntiSleep = (enabled: boolean): Promise<{ status: string; enabled: boolean }> =>
    apiFetch('/api/utils/anti-sleep', { method: 'POST', body: JSON.stringify({ enabled }) });

// ── Scheduler ───────────────────────────────────────────────────────────────

/** Task as returned by the scheduler API */
export interface QueuedTask {
    id: string;
    goal: string;
    status: 'queued' | 'scheduled' | 'starting' | 'running' | 'paused' | 'done' | 'failed' | 'cancelled';
    priority: number;
    run_at: string | null;        // ISO string or null
    created_at: string;
    started_at: string | null;
    finished_at: string | null;
    current_step: number;
    max_steps: number | null;
    eval_signal: string;
    metrics: Record<string, number>;
    stop_progress: string;
    elapsed_seconds: number;
    result: string | null;
    error: string | null;
}

export const getTasks = (): Promise<{ tasks: QueuedTask[] }> =>
    apiFetch<{ tasks: QueuedTask[] }>('/api/tasks');

export const getTaskDetail = (taskId: string): Promise<QueuedTask> =>
    apiFetch<QueuedTask>(`/api/tasks/${taskId}`);

export const getTaskLogs = (taskId: string, since = 0): Promise<{ lines: string[]; total: number }> =>
    apiFetch(`/api/tasks/${taskId}/logs?since=${since}`);

export const addTask = (
    goal: string,
    priority = 1,
    run_at?: number,
): Promise<{ status: string; task_id: string }> =>
    apiFetch('/api/tasks', {
        method: 'POST',
        body: JSON.stringify({ goal, priority, run_at: run_at ?? null }),
    });

export const cancelTask = (taskId: string): Promise<{ status: string; task_id: string }> =>
    apiFetch(`/api/tasks/${taskId}`, { method: 'DELETE' });

export const pauseTask = (taskId: string): Promise<{ status: string; task_id: string }> =>
    apiFetch(`/api/tasks/${taskId}/pause`, { method: 'POST' });

export const resumeTask = (taskId: string): Promise<{ status: string; task_id: string }> =>
    apiFetch(`/api/tasks/${taskId}/resume`, { method: 'POST' });

export const setTaskPriority = (taskId: string, priority: number): Promise<{ status: string }> =>
    apiFetch(`/api/tasks/${taskId}/priority?priority=${priority}`, { method: 'PATCH' });

export interface WaitingTask {
    task_id: string;
    goal: string;
    waiting_seconds: number;
}

export interface ScreenLockStatus {
    locked: boolean;
    holder_id: string | null;
    holder_goal: string;
    held_for_seconds: number;
    last_released_by: string | null;
    waiting_tasks: WaitingTask[];
}

export interface ScheduleStatus {
    max_concurrent: number;
    slots_used: number;
    slots_free: number;
    running: number;
    queued: number;
    paused: number;
    screen_lock: ScreenLockStatus;
}

export const getScreenLockStatus = (): Promise<ScreenLockStatus> =>
    apiFetch<ScreenLockStatus>('/api/screen-lock');

export const getScheduleStatus = (): Promise<ScheduleStatus> =>
    apiFetch<ScheduleStatus>('/api/schedule/status');

// ── Tunnel ──────────────────────────────────────────────────────────────────
export const startTunnel = (): Promise<{ status: string; url: string }> =>
    apiFetch('/api/tunnel/start', { method: 'POST' });

export const stopTunnel = (): Promise<{ status: string }> =>
    apiFetch('/api/tunnel/stop', { method: 'POST' });

export const getTunnelStatus = (): Promise<{ active: boolean; url: string | null }> =>
    apiFetch('/api/tunnel/status');

// ── Browser Utils ──────────────────────────────────────────────────────────
export const getBrowserScreenshotUrl = (): string =>
    `${BASE_URL}/api/browser/screenshot?t=${Date.now()}`;

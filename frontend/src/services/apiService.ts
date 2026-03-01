/**
 * apiService.ts — All calls to the Autobot FastAPI backend.
 * 
 * Replaces geminiService.ts (which called Gemini directly from the browser).
 * The backend now handles all AI calls via /api/chat.
 */

// In dev: Vite proxy forwards /api → http://127.0.0.1:8000 so BASE_URL = ''
// In production: React and FastAPI are on the same origin so BASE_URL = ''
// Override with VITE_API_URL only if explicitly set (e.g. for a remote backend)
const BASE_URL: string =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_URL)
    || '';


/** Generic fetch helper with JSON body */
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
        headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
        ...options,
    });
    if (!res.ok) {
        const err = await res.text();
        throw new Error(`API Error ${res.status}: ${err}`);
    }
    return res.json();
}

// ── Status ─────────────────────────────────────────────────────────────────
export interface HumanInputPending {
    prompt: string;
    key: string;
}

export interface BackendStatus {
    status: string;
    run_status: 'idle' | 'running' | 'done' | 'failed' | 'cancelled';
    active_run_id: string | null;
    browser: { active: boolean; mode: string; url: string };
    llm_enabled: boolean;
    llm_provider: string;
    llm_model: string;
    human_input_pending?: HumanInputPending | null;
}

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

export interface BackendPlan {
    id: string;
    name: string;
    description: string;
    steps: PlanStep[];
}

/** Convert a natural-language task into a WorkflowPlan via the backend */
export const planFromText = (task: string): Promise<{ plan: BackendPlan }> =>
    apiFetch('/api/plan/text', { method: 'POST', body: JSON.stringify({ task }) });

/** Execute a plan. Returns immediately with run_id; poll /api/run/:id for status */
export const runPlan = (plan: BackendPlan): Promise<{ run_id: string; status: string }> =>
    apiFetch('/api/plan/run', { method: 'POST', body: JSON.stringify({ plan }) });

/** Cancel a running plan */
export const cancelRun = (runId: string): Promise<{ status: string }> =>
    apiFetch(`/api/run/${runId}/cancel`, { method: 'POST' });

// ── AI Chat ─────────────────────────────────────────────────────────────────
export interface ChatResponse {
    reply: string;
    plan: BackendPlan | null;
}

/**
 * Send a chat message to the Autobot AI planner.
 * The backend routes this through LLMBrain (DeepSeek/OpenRouter) or 
 * falls back to text-based planning.
 */
export const sendChat = (message: string, state?: Record<string, any>): Promise<ChatResponse> =>
    apiFetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ message, state: state || {} }),
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
}

export const getSettings = (): Promise<BackendSettings> =>
    apiFetch('/api/settings');

export const updateSettings = (settings: Partial<{
    llm_provider: string;
    llm_model: string;
    browser_mode: string;
    openrouter_api_key: string;
    openai_api_key: string;
}>): Promise<{ status: string; keys_changed: string[] }> =>
    apiFetch('/api/settings', { method: 'POST', body: JSON.stringify(settings) });

// ── Logs (polling fallback when WebSocket is unavailable) ────────────────────
export interface LogsResponse {
    logs: string[];
}

export const getLogs = (limit: number = 500): Promise<LogsResponse> =>
    apiFetch(`/api/logs?limit=${limit}`).then((r: any) => ({ logs: r.logs || [] }));

// ── WebSocket log streaming (with exponential backoff and optional polling fallback) ─
const WS_RECONNECT_BASE_MS = 2000;
const WS_RECONNECT_MAX_MS = 30000;
const WS_MAX_RECONNECT_ATTEMPTS = 15;

export function connectLogStream(
    onMessage: (line: string) => void,
    onClose?: () => void,
    options?: { usePollingFallback?: boolean; onLogsSnapshot?: (logs: string[]) => void },
): () => void {
    let closed = false;
    let reconnectAttempt = 0;
    let pollInterval: ReturnType<typeof setInterval> | null = null;

    const wsUrl = (BASE_URL.replace(/^http/, 'ws') || 'ws://127.0.0.1:8000').replace(/\/?$/, '') + '/ws/logs';

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

    function tryConnect() {
        if (closed) return;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (e) => {
            if (e.data !== '__ping__') onMessage(e.data);
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
}

export const getWorkflows = (): Promise<{ workflows: BackendWorkflow[] }> =>
    apiFetch('/api/workflows');

export const runWorkflow = (workflow_id: string, topic: string = ''): Promise<{ run_id: string; status: string; plan_name: string }> =>
    apiFetch('/api/workflows/run', { method: 'POST', body: JSON.stringify({ workflow_id, topic }) });

// ── Human input (when a run requests password/token) ────────────────────────
export const getPendingHumanInput = (): Promise<{ pending: boolean; prompt?: string; key?: string }> =>
    apiFetch('/api/human_input');

export const submitHumanInput = (key: string, value: string): Promise<{ status: string; key: string }> =>
    apiFetch('/api/human_input', { method: 'POST', body: JSON.stringify({ key, value }) });

// ── Browser Utils ──────────────────────────────────────────────────────────
export const getBrowserScreenshotUrl = (): string =>
    `${BASE_URL}/api/browser/screenshot?t=${Date.now()}`;

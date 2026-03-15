/**
 * TaskQueuePanel — Multi-task queue UI for the Dashboard.
 *
 * Shows:
 *  - ScreenLock status bar (which task currently controls the computer)
 *  - List of queued / running / done tasks with live metrics
 *  - Inline "Add Task" input (chat-safe: doesn't interrupt active task)
 *  - Cancel button per task
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
    CheckCircle2, Clock, X, Plus, ChevronDown, ChevronUp,
    Activity, Cpu, BarChart2, Loader2, Ban, Terminal,
} from 'lucide-react';
import { QueuedTask, ScreenLockStatus, getTaskLogs } from '../services/apiService';

// ── Helpers ──────────────────────────────────────────────────────────────────

function statusColor(status: QueuedTask['status']): string {
    switch (status) {
        case 'running':   return 'text-[var(--brand-primary)]';
        case 'starting':  return 'text-amber-400';
        case 'done':      return 'text-emerald-400';
        case 'failed':    return 'text-red-400';
        case 'cancelled': return 'text-[var(--base-text-muted)]';
        case 'scheduled': return 'text-amber-300';
        default:          return 'text-[var(--base-text-muted)]';
    }
}

function statusBg(status: QueuedTask['status']): string {
    switch (status) {
        case 'running':   return 'bg-[var(--brand-primary)]/20';
        case 'starting':  return 'bg-amber-400/10';
        case 'done':      return 'bg-emerald-500/10';
        case 'failed':    return 'bg-red-500/10';
        case 'cancelled': return 'bg-[var(--base-border)]';
        case 'scheduled': return 'bg-amber-300/10';
        default:          return 'bg-[var(--base-border)]';
    }
}

function StatusIcon({ status }: { status: QueuedTask['status'] }) {
    const cls = `shrink-0 ${statusColor(status)}`;
    switch (status) {
        case 'running':   return <Loader2 size={15} className={`${cls} animate-spin`} />;
        case 'starting':  return <Activity size={15} className={cls} />;
        case 'done':      return <CheckCircle2 size={15} className={cls} />;
        case 'failed':    return <AlertCircle size={15} className={cls} />;
        case 'cancelled': return <Ban size={15} className={cls} />;
        case 'scheduled': return <Clock size={15} className={cls} />;
        default:          return <Clock size={15} className={cls} />;
    }
}

function EvalBadge({ signal }: { signal: string }) {
    const map: Record<string, { label: string; cls: string }> = {
        continue:  { label: 'Going',    cls: 'bg-emerald-500/10 text-emerald-400' },
        replan:    { label: 'Replan',   cls: 'bg-amber-500/10 text-amber-400' },
        complete:  { label: 'Done',     cls: 'bg-[var(--brand-primary)]/10 text-[var(--brand-primary)]' },
        pause:     { label: 'Paused',   cls: 'bg-amber-400/10 text-amber-300' },
        escalate:  { label: 'Alert',    cls: 'bg-red-500/10 text-red-400' },
    };
    const { label, cls } = map[signal] ?? { label: signal, cls: 'bg-[var(--base-border)] text-[var(--base-text-muted)]' };
    return (
        <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-widest ${cls}`}>
            {label}
        </span>
    );
}

function StepProgress({ task }: { task: QueuedTask }) {
    if (!task.max_steps || task.status === 'queued' || task.status === 'scheduled') return null;
    const pct = Math.min(100, Math.round((task.current_step / task.max_steps) * 100));
    return (
        <div className="mt-2 space-y-1">
            <div className="flex justify-between text-[9px] font-mono text-[var(--base-text-muted)]">
                <span>Step {task.current_step} / {task.max_steps === null ? '∞' : task.max_steps}</span>
                <span>{task.stop_progress || `${pct}%`}</span>
            </div>
            <div className="h-1 w-full bg-[var(--base-border)] rounded-full overflow-hidden">
                <motion.div
                    animate={{ width: `${task.max_steps ? pct : 0}%` }}
                    className="h-full bg-[var(--brand-primary)] rounded-full"
                />
            </div>
        </div>
    );
}

function MetricsPills({ metrics }: { metrics: Record<string, number> }) {
    const entries = Object.entries(metrics);
    if (!entries.length) return null;
    return (
        <div className="flex flex-wrap gap-1.5 mt-2">
            {entries.map(([k, v]) => (
                <span key={k} className="flex items-center gap-1 px-2 py-0.5 rounded bg-[var(--brand-primary)]/10 text-[var(--brand-primary)] text-[9px] font-mono font-bold">
                    <BarChart2 size={9} />
                    {k}: {v}
                </span>
            ))}
        </div>
    );
}

// ── Per-task log viewer ───────────────────────────────────────────────────────

function TaskLogViewer({ taskId, isRunning }: { taskId: string; isRunning: boolean }) {
    const [lines, setLines] = useState<string[]>([]);
    const [total, setTotal] = useState(0);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let cancelled = false;
        let since = 0;

        const poll = async () => {
            if (cancelled) return;
            try {
                const resp = await getTaskLogs(taskId, since);
                if (resp.lines.length > 0) {
                    setLines(prev => [...prev, ...resp.lines]);
                    since = resp.total;
                    setTotal(resp.total);
                }
            } catch { /* ignore */ }
        };

        poll(); // immediate first load
        const interval = isRunning ? setInterval(poll, 2000) : null;
        return () => {
            cancelled = true;
            if (interval) clearInterval(interval);
        };
    }, [taskId, isRunning]);

    // Auto-scroll to bottom when new lines arrive
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [lines]);

    return (
        <div className="mt-3 rounded-xl overflow-hidden border border-[var(--base-border)] bg-black/30">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--base-border)]">
                <Terminal size={11} className="text-[var(--base-text-muted)]" />
                <span className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)]">
                    Logs
                </span>
                <span className="ml-auto text-[9px] font-mono text-[var(--base-text-muted)]">
                    {total} lines
                </span>
            </div>
            <div className="h-40 overflow-y-auto p-3 space-y-0.5 custom-scrollbar font-mono text-[10px]">
                {lines.length === 0 ? (
                    <span className="text-[var(--base-text-muted)] italic">No logs yet...</span>
                ) : (
                    lines.map((line, i) => (
                        <div key={i} className={`leading-relaxed ${
                            line.includes('❌') || line.includes('Error') || line.includes('failed')
                                ? 'text-red-400'
                                : line.includes('✅') || line.includes('complete') || line.includes('done')
                                    ? 'text-emerald-400'
                                    : line.includes('🧠') || line.includes('💭')
                                        ? 'text-[var(--brand-primary)]'
                                        : 'text-[var(--base-text-muted)]'
                        }`}>
                            {line}
                        </div>
                    ))
                )}
                <div ref={bottomRef} />
            </div>
        </div>
    );
}

// ── Screen lock bar ───────────────────────────────────────────────────────────

function ScreenLockBar({ lockStatus }: { lockStatus: ScreenLockStatus | null }) {
    if (!lockStatus) return null;
    return (
        <div className={`flex items-center gap-3 px-4 py-2.5 rounded-xl text-[10px] font-bold uppercase tracking-widest mb-4 ${
            lockStatus.locked
                ? 'bg-[var(--brand-primary)]/10 border border-[var(--brand-primary)]/20 text-[var(--brand-primary)]'
                : 'bg-[var(--base-border)] text-[var(--base-text-muted)]'
        }`}>
            <Cpu size={13} className={lockStatus.locked ? 'animate-pulse' : ''} />
            {lockStatus.locked ? (
                <span>
                    Screen controlled by task <span className="font-mono">{lockStatus.holder_id}</span>
                    {lockStatus.holder_goal && (
                        <span className="ml-2 normal-case font-normal text-[var(--base-text-muted)]">
                            — {lockStatus.holder_goal}
                        </span>
                    )}
                    <span className="ml-2 font-mono">{lockStatus.held_for_seconds}s</span>
                </span>
            ) : (
                <span>Screen available — no task is active</span>
            )}
        </div>
    );
}

// ── Main component ────────────────────────────────────────────────────────────

interface TaskQueuePanelProps {
    tasks: QueuedTask[];
    lockStatus: ScreenLockStatus | null;
    onAddTask: (goal: string) => void;
    onCancelTask: (id: string) => void;
}

export default function TaskQueuePanel({
    tasks, lockStatus, onAddTask, onCancelTask,
}: TaskQueuePanelProps) {
    const [input, setInput] = useState('');
    const [isAdding, setIsAdding] = useState(false);
    const [expandedId, setExpandedId] = useState<string | null>(null);

    const toggleExpand = (id: string) =>
        setExpandedId(prev => (prev === id ? null : id));

    const handleAdd = () => {
        const goal = input.trim();
        if (!goal) return;
        onAddTask(goal);
        setInput('');
        setIsAdding(false);
    };

    const active = tasks.filter(t => ['starting', 'running'].includes(t.status));
    const queued = tasks.filter(t => ['queued', 'scheduled'].includes(t.status));
    const finished = tasks.filter(t => ['done', 'failed', 'cancelled'].includes(t.status));

    return (
        <div className="space-y-5">
            {/* Header */}
            <div className="flex items-center justify-between px-1">
                <div>
                    <h3 className="text-xl font-bold tracking-tight">Task Queue</h3>
                    <p className="text-[10px] text-[var(--base-text-muted)] uppercase tracking-widest mt-0.5">
                        {active.length} running · {queued.length} waiting
                    </p>
                </div>
                <button
                    onClick={() => setIsAdding(v => !v)}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[var(--brand-primary)]/20 text-[var(--brand-primary)] text-[10px] font-bold uppercase tracking-widest hover:bg-[var(--brand-primary)]/30 transition-colors"
                >
                    <Plus size={13} /> Add Task
                </button>
            </div>

            {/* Screen lock */}
            <ScreenLockBar lockStatus={lockStatus} />

            {/* Inline add form */}
            <AnimatePresence>
                {isAdding && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="glass-panel p-4 rounded-2xl border border-[var(--brand-primary)]/20 flex gap-3">
                            <input
                                autoFocus
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                onKeyDown={e => {
                                    if (e.key === 'Enter') handleAdd();
                                    if (e.key === 'Escape') { setIsAdding(false); setInput(''); }
                                }}
                                placeholder="Describe what you want Autobot to do..."
                                className="flex-1 bg-transparent text-sm text-[var(--base-text)] placeholder-[var(--base-text-muted)] outline-none"
                            />
                            <button
                                onClick={handleAdd}
                                disabled={!input.trim()}
                                className="px-4 py-1.5 rounded-lg bg-[var(--brand-primary)] text-white text-[10px] font-bold uppercase tracking-widest disabled:opacity-40 hover:brightness-110 transition-all"
                            >
                                Queue
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Task list */}
            {tasks.length === 0 ? (
                <div className="text-center py-10 text-[var(--base-text-muted)] text-sm italic">
                    Queue is empty — add a task above or start a workflow.
                </div>
            ) : (
                <div className="space-y-3">
                    {/* Active first */}
                    {[...active, ...queued, ...finished].map(task => (
                        <motion.div
                            key={task.id}
                            layout
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -8 }}
                            className={`glass-panel p-4 rounded-2xl border group transition-all ${
                                task.status === 'running'
                                    ? 'border-[var(--brand-primary)]/30 glow-border'
                                    : 'border-[var(--base-border)]'
                            }`}
                        >
                            <div className="flex items-start gap-3">
                                {/* Status icon */}
                                <div className={`mt-0.5 w-8 h-8 rounded-xl flex items-center justify-center shrink-0 ${statusBg(task.status)}`}>
                                    <StatusIcon status={task.status} />
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <span className="text-sm font-semibold leading-tight truncate">{task.goal}</span>
                                        <span className={`text-[9px] font-mono uppercase tracking-widest ${statusColor(task.status)}`}>
                                            {task.status}
                                        </span>
                                        {task.status === 'running' && (
                                            <EvalBadge signal={task.eval_signal} />
                                        )}
                                    </div>

                                    {/* Step progress */}
                                    <StepProgress task={task} />

                                    {/* Metrics */}
                                    {task.status === 'running' && (
                                        <MetricsPills metrics={task.metrics} />
                                    )}

                                    {/* Footer */}
                                    <div className="flex items-center gap-3 mt-2 text-[9px] text-[var(--base-text-muted)] font-mono">
                                        <span>#{task.id}</span>
                                        {task.elapsed_seconds > 0 && (
                                            <span>{task.elapsed_seconds}s elapsed</span>
                                        )}
                                        {task.run_at && task.status === 'scheduled' && (
                                            <span className="text-amber-300">
                                                Scheduled: {new Date(task.run_at).toLocaleString()}
                                            </span>
                                        )}
                                        {task.error && (
                                            <span className="text-red-400 truncate max-w-[200px]">{task.error}</span>
                                        )}
                                    </div>
                                </div>

                                {/* Expand + Cancel buttons */}
                                <div className="flex items-center gap-1 shrink-0">
                                    <button
                                        onClick={() => toggleExpand(task.id)}
                                        className="p-1.5 rounded-lg hover:bg-[var(--brand-primary)]/10 text-[var(--base-text-muted)] hover:text-[var(--brand-primary)] transition-all"
                                        title={expandedId === task.id ? 'Hide logs' : 'Show logs'}
                                    >
                                        {expandedId === task.id
                                            ? <ChevronUp size={14} />
                                            : <ChevronDown size={14} />}
                                    </button>
                                    {['queued', 'scheduled', 'starting', 'running'].includes(task.status) && (
                                        <button
                                            onClick={() => onCancelTask(task.id)}
                                            className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-red-500/10 text-red-400 transition-all"
                                            title="Cancel task"
                                        >
                                            <X size={14} />
                                        </button>
                                    )}
                                </div>
                            </div>

                            {/* Expandable log viewer */}
                            <AnimatePresence>
                                {expandedId === task.id && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className="overflow-hidden"
                                    >
                                        <TaskLogViewer
                                            taskId={task.id}
                                            isRunning={task.status === 'running'}
                                        />
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>
                    ))}
                </div>
            )}
        </div>
    );
}

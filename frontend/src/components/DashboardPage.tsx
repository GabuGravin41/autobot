import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import {
    Terminal, Play, CheckCircle2, AlertCircle, Activity, Zap, Monitor,
    Download, PlusCircle, Search, ChevronRight, FileText, Mail,
    Maximize2, Minimize2, X, Move,
} from 'lucide-react';
import { RunHistory } from '../types';
import { BackendStatus, BackendAdapter, getBrowserScreenshotUrl } from '../services/apiService';

interface DashboardPageProps {
    backendOnline: boolean;
    backendStatus: BackendStatus | null;
    activeRun: RunHistory | null;
    liveRuns: RunHistory[];
    liveAdapters: BackendAdapter[];
    liveLogLines: string[];
    screenshotUrl: string;
    onRefreshScreenshot: () => void;
    onAbortRun: () => void;
    onSelectRun: (run: RunHistory) => void;
    onSelectArtifact: (artifact: any) => void;
    scheduledTasks: any[];
    onCancelTask: (id: string) => void;
}

export default function DashboardPage({
    backendOnline, backendStatus, activeRun, liveRuns, liveAdapters,
    liveLogLines, screenshotUrl, onRefreshScreenshot, onAbortRun,
    onSelectRun, onSelectArtifact, scheduledTasks, onCancelTask,
}: DashboardPageProps) {
    const navigate = useNavigate();
    const [isScreenPopout, setIsScreenPopout] = useState(false);
    const [popoutPos, setPopoutPos] = useState({ x: 100, y: 100 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
    const [popoutSize, setPopoutSize] = useState<'medium' | 'large'>('medium');

    const handleDragStart = (e: React.MouseEvent) => {
        setIsDragging(true);
        setDragOffset({ x: e.clientX - popoutPos.x, y: e.clientY - popoutPos.y });
    };

    const handleDrag = (e: React.MouseEvent) => {
        if (!isDragging) return;
        setPopoutPos({ x: e.clientX - dragOffset.x, y: e.clientY - dragOffset.y });
    };

    const handleDragEnd = () => setIsDragging(false);

    const runArtifacts = activeRun?.artifacts ? Object.entries(activeRun.artifacts).map(([k, v]) => ({
        id: k,
        title: k,
        type: 'file',
        timestamp: 'New',
        content: v
    })) : [];

    return (
        <motion.div
            key="dashboard"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-8"
        >
            {/* Auth notification banner */}
            {backendStatus?.auth_notification && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-panel p-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 flex items-center gap-4"
                >
                    <div className="w-10 h-10 rounded-xl bg-amber-500/20 flex items-center justify-center shrink-0">
                        <span className="text-xl">🔐</span>
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="text-xs font-bold uppercase tracking-widest text-amber-400 mb-0.5">Login Detected</div>
                        <p className="text-sm text-[var(--base-text-muted)] truncate">{backendStatus.auth_notification.message}</p>
                        <p className="text-[10px] text-amber-400/60 font-mono mt-1">{backendStatus.auth_notification.url}</p>
                    </div>
                    <div className="flex gap-2 shrink-0">
                        <button
                            onClick={onRefreshScreenshot}
                            className="px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-400 text-[10px] font-bold uppercase tracking-widest hover:bg-amber-500/30 transition-colors"
                        >
                            View Screen
                        </button>
                    </div>
                </motion.div>
            )}

            {/* Active run card */}
            {activeRun && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="glass-panel p-8 rounded-3xl glow-border relative overflow-hidden group mb-8"
                >
                    <div className="absolute inset-0 hologram-grid opacity-10 pointer-events-none" />
                    <div className="relative z-10">
                        <div className="flex justify-between items-start mb-8">
                            <div>
                                <div className="flex items-center gap-3 mb-2">
                                    <div className="px-2 py-0.5 rounded bg-[var(--brand-primary)]/20 text-[var(--brand-primary)] text-[10px] font-bold uppercase tracking-wider flex items-center gap-2">
                                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--brand-primary)] animate-pulse" />
                                        Active Operation
                                    </div>
                                    <h3 className="text-2xl font-bold tracking-tight">{activeRun.planName}</h3>
                                </div>
                                <p className="text-[var(--base-text-muted)] text-sm">Initialized at {activeRun.timestamp} • Session ID: {activeRun.id}</p>
                            </div>
                            <div className="text-right">
                                {activeRun.id === 'autonomous' && (
                                    <div className="mb-2">
                                        <div className="text-[10px] text-[var(--brand-primary)] uppercase tracking-widest font-bold">Autonomous Phase</div>
                                        <div className="text-sm font-mono text-[var(--base-text)] italic">{activeRun.artifacts?.current_phase || 'Decomposing...'}</div>
                                    </div>
                                )}
                                <div className="text-3xl font-mono font-bold text-[var(--brand-primary)]">
                                    {activeRun.totalSteps > 0 ? Math.round((activeRun.stepsCompleted / activeRun.totalSteps) * 100) : (activeRun.id === 'autonomous' ? 'AU' : '0%')}
                                </div>
                                <div className="text-[10px] text-[var(--base-text-muted)] uppercase tracking-widest font-bold">Execution Progress</div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            <div className="lg:col-span-2 space-y-6">
                                {/* Progress bar */}
                                <div className="space-y-3">
                                    <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)]">
                                        <span>Step {activeRun.stepsCompleted} of {activeRun.totalSteps}</span>
                                        <span className="text-[var(--brand-primary)]">{activeRun.status === 'running' ? 'Processing...' : activeRun.status}</span>
                                    </div>
                                    <div className="h-3 w-full bg-[var(--base-border)] rounded-full overflow-hidden p-0.5 border border-[var(--base-border)]">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${activeRun.totalSteps > 0 ? (activeRun.stepsCompleted / activeRun.totalSteps) * 100 : 0}%` }}
                                            className="h-full bg-gradient-to-r from-brand-500 to-accent-cyan rounded-full shadow-[0_0_15px_rgba(var(--brand-500-rgb),0.5)]"
                                        />
                                    </div>
                                </div>

                                {/* Log terminal */}
                                <div className="glass-panel bg-[var(--base-border)] rounded-2xl p-6 h-[280px] flex flex-col">
                                    <div className="flex items-center justify-between mb-4 pb-4 border-b border-[var(--base-border)]">
                                        <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] flex items-center gap-2">
                                            <Terminal size={14} /> Execution Logs
                                        </div>
                                        <div className="flex gap-1">
                                            <div className="w-2 h-2 rounded-full bg-red-500/20" />
                                            <div className="w-2 h-2 rounded-full bg-amber-500/20" />
                                            <div className="w-2 h-2 rounded-full bg-emerald-500/20" />
                                        </div>
                                    </div>
                                    <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-2 font-mono text-[11px]">
                                        {activeRun.logs.map((log, i) => (
                                            <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} key={i} className="flex gap-3">
                                                <span className="text-[var(--base-text-muted)]">[{i.toString().padStart(2, '0')}]</span>
                                                <span className={i === activeRun.logs.length - 1 ? 'text-[var(--brand-primary)]' : 'text-[var(--base-text-muted)]'}>{log}</span>
                                            </motion.div>
                                        ))}
                                        <div className="animate-pulse text-[var(--brand-primary)]/50">_</div>
                                    </div>
                                </div>
                            </div>

                            {/* Right column: screen preview + controls */}
                            <div className="space-y-6">
                                {backendOnline && (
                                    <div className="space-y-4">
                                        <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] flex items-center gap-2 justify-between">
                                            <span className="flex items-center gap-2"><Monitor size={12} /> Screen preview</span>
                                            <button
                                                onClick={onRefreshScreenshot}
                                                className="px-2 py-1 rounded bg-[var(--base-border)] hover:bg-[var(--base-border)] text-[10px] font-bold uppercase tracking-wider transition-colors"
                                            >
                                                {screenshotUrl ? 'Refresh' : 'See current screen'}
                                            </button>
                                        </div>
                                        {screenshotUrl ? (
                                            <div
                                                className="relative aspect-video rounded-2xl overflow-hidden border border-[var(--base-border)] cursor-pointer group/screen"
                                                onClick={() => setIsScreenPopout(true)}
                                                title="Click to pop out"
                                            >
                                                <img src={screenshotUrl} alt="Current screen" className="w-full h-full object-cover"
                                                    onError={(e) => (e.currentTarget.style.display = 'none')} />
                                                {backendStatus?.browser?.active && (
                                                    <div className="absolute top-3 right-3 flex items-center gap-2 px-2 py-1 rounded bg-[var(--base-border)] backdrop-blur-md border border-[var(--base-border)]">
                                                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                                                        <span className="text-[8px] font-bold text-[var(--base-text)] tracking-widest uppercase">Live</span>
                                                    </div>
                                                )}
                                                <div className="absolute inset-0 bg-black/0 group-hover/screen:bg-black/30 transition-colors flex items-center justify-center opacity-0 group-hover/screen:opacity-100">
                                                    <Maximize2 size={24} className="text-white drop-shadow-lg" />
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="aspect-video rounded-2xl border border-dashed border-[var(--base-border)] flex items-center justify-center text-[var(--base-text-muted)] text-sm">
                                                Click "See current screen" to preview
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Latest log snippet */}
                                <div className="p-4 rounded-2xl bg-[var(--brand-primary)]/20 border border-brand-500/10">
                                    <div className="text-[9px] font-bold uppercase tracking-widest text-[var(--brand-primary)] mb-2">Latest Log</div>
                                    <div className="text-xs text-[var(--base-text-muted)] leading-relaxed font-mono truncate">
                                        {activeRun.logs.length > 0 ? activeRun.logs[activeRun.logs.length - 1] : 'Waiting for engine output...'}
                                    </div>
                                </div>

                                {/* Abort button */}
                                {activeRun.status === 'running' && (
                                    <button
                                        onClick={onAbortRun}
                                        className="w-full py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-[10px] font-bold uppercase tracking-widest hover:bg-red-500/20 transition-colors"
                                    >
                                        ■ Abort Run
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </motion.div>
            )}

            {/* Header */}
            <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-4xl font-bold tracking-tight mb-2">Command Center</h2>
                    <p className="text-[var(--base-text-muted)]">Real-time overview of your autonomous operations.</p>
                </div>
                <div className="flex items-center gap-3">
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest ${backendOnline
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : 'bg-red-500/10 text-red-400 border border-red-500/20'
                        }`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
                        {backendOnline ? 'Engine Online' : 'Engine Offline'}
                    </div>
                    <button
                        onClick={() => {
                            const logText = activeRun?.logs?.join('\n') || liveLogLines.join('\n');
                            const blob = new Blob([logText], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url; a.download = 'autobot-logs.txt'; a.click();
                            URL.revokeObjectURL(url);
                        }}
                        className="btn-secondary py-2 px-4 text-[10px] uppercase tracking-widest flex items-center gap-2"
                    >
                        <Download size={14} /> Export Logs
                    </button>
                    <button
                        onClick={() => navigate('/planner')}
                        className="btn-primary py-2 px-6 text-[10px] uppercase tracking-widest flex items-center gap-2"
                    >
                        <PlusCircle size={14} /> New Workflow
                    </button>
                </div>
            </header>

            {/* Stats row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
                {[
                    { label: 'Active Runs', value: liveRuns.filter(r => r.status === 'running').length + (activeRun?.status === 'running' ? 1 : 0), color: 'text-[var(--brand-primary)]' },
                    { label: 'Total Runs', value: liveRuns.length, color: 'text-[var(--base-text-muted)]' },
                    { label: 'Adapters', value: liveAdapters.length, color: 'text-amber-400' },
                    { label: 'Backend Uptime', value: backendOnline ? '100%' : '0%', color: 'text-[var(--brand-primary)]' },
                ].map(({ label, value, color }) => (
                    <div key={label} className="glass-panel p-6 rounded-3xl border-[var(--base-border)]">
                        <div className={`text-[10px] font-bold uppercase tracking-[0.2em] mb-1 ${color}`}>{label}</div>
                        <div className="text-4xl font-bold tracking-tighter">{value}</div>
                    </div>
                ))}
            </div>

            {/* Runs + artifacts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-6">
                    <div className="flex items-center justify-between px-2">
                        <h3 className="text-xl font-bold tracking-tight">Active Operations</h3>
                        <button onClick={() => navigate('/history')} className="text-[10px] font-bold uppercase tracking-widest text-[var(--brand-primary)] hover:underline">
                            View All
                        </button>
                    </div>
                    <div className="max-h-[400px] overflow-y-auto custom-scrollbar pr-1">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {liveRuns.map(run => (
                                <div
                                    key={run.id}
                                    onClick={() => onSelectRun(run as any)}
                                    className="glass-panel p-4 rounded-2xl border-[var(--base-border)] hover:border-brand-500/30 transition-all group cursor-pointer"
                                >
                                    <div className="flex items-center gap-3 mb-3">
                                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${run.status === 'running' ? 'bg-[var(--brand-primary)]/20 text-[var(--brand-primary)] animate-pulse'
                                            : run.status === 'success' ? 'bg-emerald-500/10 text-emerald-400'
                                                : 'bg-red-500/10 text-red-400'
                                            }`}>
                                            {run.status === 'running' ? <Play size={16} />
                                                : run.status === 'success' ? <CheckCircle2 size={16} />
                                                    : <AlertCircle size={16} />}
                                        </div>
                                        <div className="min-w-0">
                                            <div className="font-bold text-sm truncate group-hover:text-[var(--brand-primary)] transition-colors">{run.planName}</div>
                                            <div className="text-[9px] text-[var(--base-text-muted)] uppercase tracking-widest truncate mt-0.5">
                                                {run.timestamp}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="flex-1 h-1.5 bg-[var(--base-border)] rounded-full overflow-hidden">
                                            <div className="h-full bg-[var(--brand-primary)] transition-all duration-500" style={{ width: `${run.progress ?? 0}%` }} />
                                        </div>
                                        <span className="text-[10px] font-mono text-[var(--base-text-muted)]">{run.progress ?? 0}%</span>
                                        <ChevronRight size={14} className="text-[var(--base-text-muted)] group-hover:text-[var(--brand-primary)] transition-colors shrink-0" />
                                    </div>
                                </div>
                            ))}
                            {liveRuns.length === 0 && !activeRun && (
                                <div className="col-span-full glass-panel p-10 rounded-3xl border-dashed border-[var(--base-border)] text-center text-[var(--base-text-muted)] text-sm">
                                    No runs yet — start a workflow from the AI Planner or Workflows tab.
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Task Queue section */}
                    <div className="mt-12 space-y-6">
                        <div className="flex items-center justify-between px-2">
                            <h3 className="text-xl font-bold tracking-tight text-[var(--brand-primary)]">Operation Queue</h3>
                            <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)]">
                                {scheduledTasks.length} Pending Actions
                            </div>
                        </div>
                        <div className="grid grid-cols-1 gap-4">
                            {scheduledTasks.map(task => (
                                <div key={task.id} className="glass-panel p-4 rounded-2xl border-[var(--base-border)] flex items-center justify-between group">
                                    <div className="flex items-center gap-4">
                                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold ${
                                            task.status === 'running' ? 'bg-brand-500/20 text-brand-500 animate-spin' :
                                            task.status === 'done' ? 'bg-emerald-500/20 text-emerald-500' :
                                            'bg-[var(--base-border)] text-[var(--base-text-muted)]'
                                        }`}>
                                            {task.status === 'running' ? '↻' : task.id}
                                        </div>
                                        <div>
                                            <div className="text-sm font-bold">{task.goal}</div>
                                            <div className="text-[9px] uppercase tracking-widest opacity-60">Status: {task.status} • Created: {new Date(task.created_at).toLocaleTimeString()}</div>
                                        </div>
                                    </div>
                                    {task.status === 'queued' && (
                                        <button 
                                            onClick={() => onCancelTask(task.id)}
                                            className="opacity-0 group-hover:opacity-100 p-2 rounded-lg hover:bg-red-500/10 text-red-400 transition-all"
                                        >
                                            <AlertCircle size={14} />
                                        </button>
                                    )}
                                </div>
                            ))}
                            {scheduledTasks.length === 0 && (
                                <div className="text-center py-8 text-[var(--base-text-muted)] text-sm italic">
                                    Queue is empty.
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Artifacts sidebar */}
                <div className="space-y-6">
                    {backendOnline && screenshotUrl && (
                        <div className="glass-panel p-4 rounded-3xl border-[var(--base-border)]">
                            <div className="flex items-center justify-between gap-2 mb-3">
                                <h3 className="text-sm font-bold tracking-tight flex items-center gap-2"><Monitor size={14} /> Screen</h3>
                                <div className="flex items-center gap-2">
                                    <button onClick={() => setIsScreenPopout(true)} className="px-2 py-1 rounded-lg bg-[var(--base-border)] hover:bg-[var(--brand-primary)]/20 text-[10px] font-bold uppercase tracking-wider transition-colors" title="Pop out">
                                        <Maximize2 size={12} />
                                    </button>
                                    <button onClick={onRefreshScreenshot} className="px-2 py-1 rounded-lg bg-[var(--brand-primary)]/20 hover:bg-[var(--brand-primary)]/20 text-[10px] font-bold uppercase tracking-wider transition-colors">
                                        Refresh
                                    </button>
                                </div>
                            </div>
                            <div
                                className="relative aspect-video rounded-xl overflow-hidden border border-[var(--base-border)] cursor-pointer group/sidebar-screen"
                                onClick={() => setIsScreenPopout(true)}
                                title="Click to pop out"
                            >
                                <img src={screenshotUrl} alt="Current screen" className="w-full h-full object-cover"
                                    onError={(e) => (e.currentTarget.style.display = 'none')} />
                                <div className="absolute inset-0 bg-black/0 group-hover/sidebar-screen:bg-black/30 transition-colors flex items-center justify-center opacity-0 group-hover/sidebar-screen:opacity-100">
                                    <Maximize2 size={20} className="text-white drop-shadow-lg" />
                                </div>
                            </div>
                        </div>
                    )}

                    <h3 className="text-xl font-bold tracking-tight px-2">Recent Artifacts</h3>
                    <div className="glass-panel p-6 rounded-3xl border-[var(--base-border)] space-y-6">
                        {runArtifacts.length === 0 && (
                            <p className="text-[var(--base-text-muted)] text-sm text-center py-4">No artifacts generated in the current run.</p>
                        )}
                        {runArtifacts.map((artifact, idx) => (
                            <div key={artifact.id} className={`flex items-start gap-4 ${idx !== runArtifacts.length - 1 ? 'pb-6 border-b border-[var(--base-border)]' : ''}`}>
                                <div className="p-2 rounded-lg bg-[var(--base-border)] text-[var(--base-text-muted)]">
                                    <FileText size={16} />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="text-sm font-bold truncate">{artifact.title}</div>
                                    <div className="text-[10px] text-[var(--base-text-muted)] uppercase tracking-widest mt-0.5">{artifact.timestamp}</div>
                                    <div className="mt-3 flex items-center gap-2">
                                        <button
                                            onClick={() => onSelectArtifact(artifact)}
                                            className="px-2 py-1 rounded bg-[var(--brand-primary)]/20 text-[var(--brand-primary)] text-[9px] font-bold uppercase tracking-widest hover:bg-[var(--brand-primary)]/20 transition-colors"
                                        >View</button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
            {/* Floating pop-out screen viewer */}
            <AnimatePresence>
                {isScreenPopout && screenshotUrl && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="fixed z-[200] shadow-2xl shadow-black/50 rounded-2xl overflow-hidden border border-[var(--base-border)] bg-[var(--base-bg)]"
                        style={{
                            left: popoutPos.x,
                            top: popoutPos.y,
                            width: popoutSize === 'large' ? '80vw' : '50vw',
                            maxWidth: popoutSize === 'large' ? '1200px' : '800px',
                        }}
                        onMouseMove={handleDrag}
                        onMouseUp={handleDragEnd}
                        onMouseLeave={handleDragEnd}
                    >
                        {/* Title bar — draggable */}
                        <div
                            className="flex items-center justify-between px-4 py-2 bg-[var(--base-border)] cursor-move select-none"
                            onMouseDown={handleDragStart}
                        >
                            <div className="flex items-center gap-2">
                                <Move size={14} className="text-[var(--base-text-muted)]" />
                                <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)]">
                                    Live Screen Preview
                                </span>
                                {backendStatus?.browser?.active && (
                                    <div className="flex items-center gap-1.5 ml-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                                        <span className="text-[8px] font-bold text-red-400 tracking-widest uppercase">Live</span>
                                    </div>
                                )}
                            </div>
                            <div className="flex items-center gap-1">
                                <button
                                    onClick={onRefreshScreenshot}
                                    className="px-2 py-1 rounded bg-[var(--base-border)] hover:bg-[var(--brand-primary)]/20 text-[9px] font-bold uppercase tracking-wider transition-colors"
                                >
                                    Refresh
                                </button>
                                <button
                                    onClick={() => setPopoutSize(s => s === 'medium' ? 'large' : 'medium')}
                                    className="p-1.5 rounded hover:bg-[var(--brand-primary)]/20 transition-colors"
                                    title={popoutSize === 'medium' ? 'Enlarge' : 'Shrink'}
                                >
                                    {popoutSize === 'medium' ? <Maximize2 size={14} /> : <Minimize2 size={14} />}
                                </button>
                                <button
                                    onClick={() => setIsScreenPopout(false)}
                                    className="p-1.5 rounded hover:bg-red-500/20 text-red-400 transition-colors"
                                    title="Close"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        </div>
                        {/* Screenshot */}
                        <div className="relative">
                            <img
                                src={screenshotUrl}
                                alt="Live screen"
                                className="w-full h-auto"
                                onError={(e) => (e.currentTarget.style.display = 'none')}
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}

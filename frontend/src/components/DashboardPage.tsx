import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import {
    Terminal, Play, CheckCircle2, AlertCircle, Activity, Zap, Monitor,
    Download, PlusCircle, Search, ChevronRight, FileText, Mail,
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
}

export default function DashboardPage({
    backendOnline, backendStatus, activeRun, liveRuns, liveAdapters,
    liveLogLines, screenshotUrl, onRefreshScreenshot, onAbortRun,
    onSelectRun, onSelectArtifact,
}: DashboardPageProps) {
    const navigate = useNavigate();

    const artifacts = liveAdapters.map(a => ({
        id: a.name,
        title: a.description,
        type: 'adapter',
        timestamp: 'online',
    }));

    return (
        <motion.div
            key="dashboard"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-8"
        >
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
                                    <div className="px-2 py-0.5 rounded bg-brand-500/20 text-brand-400 text-[10px] font-bold uppercase tracking-wider flex items-center gap-2">
                                        <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
                                        Active Operation
                                    </div>
                                    <h3 className="text-2xl font-bold tracking-tight">{activeRun.planName}</h3>
                                </div>
                                <p className="text-white/40 text-sm">Initialized at {activeRun.timestamp} • Session ID: {activeRun.id}</p>
                            </div>
                            <div className="text-right">
                                {activeRun.id === 'autonomous' && (
                                    <div className="mb-2">
                                        <div className="text-[10px] text-brand-400 uppercase tracking-widest font-bold">Autonomous Phase</div>
                                        <div className="text-sm font-mono text-white italic">{activeRun.artifacts?.current_phase || 'Decomposing...'}</div>
                                    </div>
                                )}
                                <div className="text-3xl font-mono font-bold text-brand-400">
                                    {activeRun.totalSteps > 0 ? Math.round((activeRun.stepsCompleted / activeRun.totalSteps) * 100) : (activeRun.id === 'autonomous' ? 'AU' : '0%')}
                                </div>
                                <div className="text-[10px] text-white/40 uppercase tracking-widest font-bold">Execution Progress</div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            <div className="lg:col-span-2 space-y-6">
                                {/* Progress bar */}
                                <div className="space-y-3">
                                    <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-white/40">
                                        <span>Step {activeRun.stepsCompleted} of {activeRun.totalSteps}</span>
                                        <span className="text-brand-400">{activeRun.status === 'running' ? 'Processing...' : activeRun.status}</span>
                                    </div>
                                    <div className="h-3 w-full bg-white/5 rounded-full overflow-hidden p-0.5 border border-white/10">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${activeRun.totalSteps > 0 ? (activeRun.stepsCompleted / activeRun.totalSteps) * 100 : 0}%` }}
                                            className="h-full bg-gradient-to-r from-brand-500 to-accent-cyan rounded-full shadow-[0_0_15px_rgba(var(--brand-500-rgb),0.5)]"
                                        />
                                    </div>
                                </div>

                                {/* Log terminal */}
                                <div className="glass-panel bg-black/40 rounded-2xl p-6 h-[280px] flex flex-col">
                                    <div className="flex items-center justify-between mb-4 pb-4 border-b border-white/5">
                                        <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 flex items-center gap-2">
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
                                                <span className="text-white/20">[{i.toString().padStart(2, '0')}]</span>
                                                <span className={i === activeRun.logs.length - 1 ? 'text-brand-400' : 'text-white/60'}>{log}</span>
                                            </motion.div>
                                        ))}
                                        <div className="animate-pulse text-brand-400/50">_</div>
                                    </div>
                                </div>
                            </div>

                            {/* Right column: screen preview + controls */}
                            <div className="space-y-6">
                                {backendOnline && (
                                    <div className="space-y-4">
                                        <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 flex items-center gap-2 justify-between">
                                            <span className="flex items-center gap-2"><Monitor size={12} /> Screen preview</span>
                                            <button
                                                onClick={onRefreshScreenshot}
                                                className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-[10px] font-bold uppercase tracking-wider transition-colors"
                                            >
                                                {screenshotUrl ? 'Refresh' : 'See current screen'}
                                            </button>
                                        </div>
                                        {screenshotUrl ? (
                                            <div className="relative aspect-video rounded-2xl overflow-hidden border border-white/10">
                                                <img src={screenshotUrl} alt="Current screen" className="w-full h-full object-cover"
                                                    onError={(e) => (e.currentTarget.style.display = 'none')} />
                                                {backendStatus?.browser?.active && (
                                                    <div className="absolute top-3 right-3 flex items-center gap-2 px-2 py-1 rounded bg-black/60 backdrop-blur-md border border-white/10">
                                                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                                                        <span className="text-[8px] font-bold text-white tracking-widest uppercase">Live</span>
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="aspect-video rounded-2xl border border-dashed border-white/20 flex items-center justify-center text-white/40 text-sm">
                                                Click "See current screen" to preview
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Latest log snippet */}
                                <div className="p-4 rounded-2xl bg-brand-500/5 border border-brand-500/10">
                                    <div className="text-[9px] font-bold uppercase tracking-widest text-brand-400 mb-2">Latest Log</div>
                                    <div className="text-xs text-white/60 leading-relaxed font-mono truncate">
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
                    <p className="text-white/40">Real-time overview of your autonomous operations.</p>
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
                    { label: 'Active Runs', value: liveRuns.filter(r => r.status === 'running').length + (activeRun?.status === 'running' ? 1 : 0), color: 'text-brand-400' },
                    { label: 'Total Runs', value: liveRuns.length, color: 'text-white/40' },
                    { label: 'Adapters', value: liveAdapters.length, color: 'text-amber-400' },
                    { label: 'Backend Uptime', value: backendOnline ? '100%' : '0%', color: 'text-brand-400' },
                ].map(({ label, value, color }) => (
                    <div key={label} className="glass-panel p-6 rounded-3xl border-white/5">
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
                        <button onClick={() => navigate('/history')} className="text-[10px] font-bold uppercase tracking-widest text-brand-400 hover:underline">
                            View All
                        </button>
                    </div>
                    <div className="space-y-4">
                        {liveRuns.map(run => (
                            <div
                                key={run.id}
                                onClick={() => onSelectRun(run as any)}
                                className="glass-panel p-6 rounded-3xl border-white/5 hover:border-brand-500/30 transition-all group cursor-pointer"
                            >
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                                    <div className="flex items-center gap-4">
                                        <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${run.status === 'running' ? 'bg-brand-500/10 text-brand-400 animate-pulse'
                                            : run.status === 'success' ? 'bg-emerald-500/10 text-emerald-400'
                                                : 'bg-red-500/10 text-red-400'
                                            }`}>
                                            {run.status === 'running' ? <Play size={20} />
                                                : run.status === 'success' ? <CheckCircle2 size={20} />
                                                    : <AlertCircle size={20} />}
                                        </div>
                                        <div>
                                            <div className="font-bold group-hover:text-brand-400 transition-colors">{run.planName}</div>
                                            <div className="text-[10px] text-white/40 uppercase tracking-widest flex items-center gap-2 mt-1">
                                                <span>{run.id}</span>
                                                <span className="w-1 h-1 rounded-full bg-white/20" />
                                                <span>{run.timestamp}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-6">
                                        <div className="text-right hidden xs:block">
                                            <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 mb-1">Progress</div>
                                            <div className="flex items-center gap-3">
                                                <div className="w-24 h-1.5 bg-white/5 rounded-full overflow-hidden">
                                                    <div className="h-full bg-brand-500 transition-all duration-500" style={{ width: `${run.progress ?? 0}%` }} />
                                                </div>
                                                <span className="text-xs font-mono">{run.progress ?? 0}%</span>
                                            </div>
                                        </div>
                                        <ChevronRight size={16} className="text-white/20 group-hover:text-brand-400 transition-colors" />
                                    </div>
                                </div>
                            </div>
                        ))}
                        {liveRuns.length === 0 && !activeRun && (
                            <div className="glass-panel p-10 rounded-3xl border-dashed border-white/10 text-center text-white/30 text-sm">
                                No runs yet — start a workflow from the AI Planner or Workflows tab.
                            </div>
                        )}
                    </div>
                </div>

                {/* Artifacts sidebar */}
                <div className="space-y-6">
                    {backendOnline && screenshotUrl && (
                        <div className="glass-panel p-4 rounded-3xl border-white/5">
                            <div className="flex items-center justify-between gap-2 mb-3">
                                <h3 className="text-sm font-bold tracking-tight flex items-center gap-2"><Monitor size={14} /> Screen</h3>
                                <button onClick={onRefreshScreenshot} className="px-2 py-1 rounded-lg bg-brand-500/20 hover:bg-brand-500/30 text-[10px] font-bold uppercase tracking-wider transition-colors">
                                    Refresh
                                </button>
                            </div>
                            <div className="relative aspect-video rounded-xl overflow-hidden border border-white/10">
                                <img src={screenshotUrl} alt="Current screen" className="w-full h-full object-cover"
                                    onError={(e) => (e.currentTarget.style.display = 'none')} />
                            </div>
                        </div>
                    )}

                    <h3 className="text-xl font-bold tracking-tight px-2">Recent Artifacts</h3>
                    <div className="glass-panel p-6 rounded-3xl border-white/5 space-y-6">
                        {artifacts.length === 0 && (
                            <p className="text-white/30 text-sm text-center py-4">Adapters will appear here once the backend is online.</p>
                        )}
                        {artifacts.map((artifact, idx) => (
                            <div key={artifact.id} className={`flex items-start gap-4 ${idx !== artifacts.length - 1 ? 'pb-6 border-b border-white/5' : ''}`}>
                                <div className="p-2 rounded-lg bg-white/5 text-white/40">
                                    {artifact.type === 'email' ? <Mail size={16} /> : <FileText size={16} />}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="text-sm font-bold truncate">{artifact.title}</div>
                                    <div className="text-[10px] text-white/40 uppercase tracking-widest mt-0.5">{artifact.timestamp}</div>
                                    <div className="mt-3 flex items-center gap-2">
                                        <button
                                            onClick={() => onSelectArtifact(artifact)}
                                            className="px-2 py-1 rounded bg-brand-500/10 text-brand-400 text-[9px] font-bold uppercase tracking-widest hover:bg-brand-500/20 transition-colors"
                                        >View</button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

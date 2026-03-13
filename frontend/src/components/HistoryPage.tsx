import React from 'react';
import { motion } from 'motion/react';
import { CheckCircle2, XCircle, Eye, Trash2 } from 'lucide-react';
import { RunHistory } from '../types';

interface HistoryPageProps {
    runs: RunHistory[];
    onViewDetails: (run: RunHistory) => void;
    onDeleteRun: (runId: string) => void;
}

export default function HistoryPage({ runs, onViewDetails, onDeleteRun }: HistoryPageProps) {
    return (
        <motion.div
            key="history"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-8"
        >
            <header>
                <h2 className="text-4xl font-bold tracking-tight mb-2">Operation History</h2>
                <p className="text-[var(--base-text-muted)]">Audit trail of all system activities.</p>
            </header>

            <div className="glass-panel rounded-3xl overflow-hidden flex flex-col" style={{ maxHeight: 'calc(100vh - 200px)' }}>
                <div className="overflow-y-auto custom-scrollbar">
                    <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 z-10">
                            <tr className="bg-[var(--base-border)] text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] border-b border-[var(--base-border)]">
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4">Operation</th>
                                <th className="px-6 py-4">Timestamp</th>
                                <th className="px-6 py-4">Progress</th>
                                <th className="px-6 py-4 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {runs.map(run => (
                                <tr key={run.id} className="border-b border-[var(--base-border)] hover:bg-white/[0.02] transition-colors group">
                                    <td className="px-6 py-4">
                                        <div className={`flex items-center gap-2 ${run.status === 'success' ? 'text-emerald-400' : run.status === 'running' ? 'text-[var(--brand-primary)]' : 'text-red-400'}`}>
                                            {run.status === 'success' ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                                            <span className="text-[10px] font-bold uppercase tracking-widest">{run.status}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="text-sm font-bold">{run.planName}</div>
                                        <div className="text-[10px] text-[var(--base-text-muted)] font-mono truncate max-w-[200px]">{run.id}</div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="text-xs text-[var(--base-text-muted)]">{run.timestamp}</div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-3">
                                            <div className="w-20 h-1 bg-[var(--base-border)] rounded-full overflow-hidden">
                                                <div className="h-full bg-[var(--brand-primary)]" style={{ width: `${run.progress ?? 0}%` }} />
                                            </div>
                                            <span className="text-[10px] text-[var(--base-text-muted)]">{run.progress ?? 0}%</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <div className="flex items-center justify-end gap-2">
                                            <button
                                                onClick={() => onViewDetails(run)}
                                                className="p-2 rounded-lg bg-[var(--base-border)] hover:bg-[var(--base-border)] text-[var(--base-text-muted)] hover:text-[var(--brand-primary)] transition-colors"
                                                title="View Details"
                                            >
                                                <Eye size={16} />
                                            </button>
                                            <button
                                                onClick={() => onDeleteRun(run.id)}
                                                className="p-2 rounded-lg bg-[var(--base-border)] hover:bg-[var(--base-border)] text-[var(--base-text-muted)] hover:text-red-400 transition-colors"
                                                title="Delete Record"
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                            {runs.length === 0 && (
                                <tr>
                                    <td colSpan={5} className="px-6 py-16 text-center text-[var(--base-text-muted)] text-sm">No runs recorded yet.</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </motion.div>
    );
}

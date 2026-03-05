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
                <p className="text-white/40">Audit trail of all system activities.</p>
            </header>

            <div className="glass-panel rounded-3xl overflow-hidden">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-white/5 text-[10px] font-bold uppercase tracking-widest text-white/40 border-b border-white/10">
                            <th className="px-6 py-4">Status</th>
                            <th className="px-6 py-4">Operation</th>
                            <th className="px-6 py-4">Timestamp</th>
                            <th className="px-6 py-4">Progress</th>
                            <th className="px-6 py-4 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {runs.map(run => (
                            <tr key={run.id} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors group">
                                <td className="px-6 py-4">
                                    <div className={`flex items-center gap-2 ${run.status === 'success' ? 'text-emerald-400' : run.status === 'running' ? 'text-brand-400' : 'text-red-400'}`}>
                                        {run.status === 'success' ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                                        <span className="text-[10px] font-bold uppercase tracking-widest">{run.status}</span>
                                    </div>
                                </td>
                                <td className="px-6 py-4">
                                    <div className="text-sm font-bold">{run.planName}</div>
                                    <div className="text-[10px] text-white/20 font-mono truncate max-w-[200px]">{run.id}</div>
                                </td>
                                <td className="px-6 py-4">
                                    <div className="text-xs text-white/60">{run.timestamp}</div>
                                </td>
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-20 h-1 bg-white/10 rounded-full overflow-hidden">
                                            <div className="h-full bg-brand-500" style={{ width: `${run.progress ?? 0}%` }} />
                                        </div>
                                        <span className="text-[10px] text-white/40">{run.progress ?? 0}%</span>
                                    </div>
                                </td>
                                <td className="px-6 py-4 text-right">
                                    <div className="flex items-center justify-end gap-2">
                                        <button
                                            onClick={() => onViewDetails(run)}
                                            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-brand-400 transition-colors"
                                            title="View Details"
                                        >
                                            <Eye size={16} />
                                        </button>
                                        <button
                                            onClick={() => onDeleteRun(run.id)}
                                            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-red-400 transition-colors"
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
                                <td colSpan={5} className="px-6 py-16 text-center text-white/30 text-sm">No runs recorded yet.</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </motion.div>
    );
}

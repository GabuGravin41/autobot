import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { CheckCircle2, XCircle, Eye, Trash2, BookmarkPlus, X, CheckCircle } from 'lucide-react';
import { RunHistory } from '../types';
import { saveWorkflow } from '../services/apiService';

interface HistoryPageProps {
    runs: RunHistory[];
    onViewDetails: (run: RunHistory) => void;
    onDeleteRun: (runId: string) => void;
}

function SaveFromRunModal({ run, onClose }: { run: RunHistory; onClose: () => void }) {
    const [name, setName] = useState(run.planName || '');
    const [description, setDescription] = useState(run.planName || '');
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState('');
    const goal = run.goal || run.planName || '';

    const handleSave = async () => {
        if (!name.trim() || !goal.trim()) { setError('Cannot save — run has no goal recorded.'); return; }
        setSaving(true); setError('');
        try {
            await saveWorkflow({ name: name.trim(), description: description.trim() || name.trim(), goal });
            setSaved(true);
            setTimeout(onClose, 1200);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Save failed.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[300] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={onClose}>
            <motion.div
                initial={{ opacity: 0, scale: 0.92 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.92 }}
                className="glass-panel rounded-3xl p-8 w-full max-w-md space-y-5 shadow-2xl"
                onClick={e => e.stopPropagation()}
            >
                <div className="flex items-center justify-between">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <BookmarkPlus size={16} className="text-[var(--brand-primary)]" /> Save as Workflow
                    </h3>
                    <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-[var(--base-border)] text-[var(--base-text-muted)]"><X size={14} /></button>
                </div>
                <div className="space-y-3">
                    <div>
                        <label className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1 block">Name *</label>
                        <input value={name} onChange={e => setName(e.target.value)} className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm focus:outline-none focus:border-[var(--brand-primary)]/50" />
                    </div>
                    <div>
                        <label className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1 block">Description</label>
                        <input value={description} onChange={e => setDescription(e.target.value)} className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm focus:outline-none focus:border-[var(--brand-primary)]/50" />
                    </div>
                    <div>
                        <label className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1 block">Goal (read-only)</label>
                        <div className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2 px-4 text-xs font-mono text-[var(--base-text-muted)] max-h-24 overflow-y-auto">
                            {goal || <span className="italic opacity-50">No goal recorded for this run.</span>}
                        </div>
                    </div>
                </div>
                {error && <p className="text-xs text-red-400">{error}</p>}
                <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 py-2.5 rounded-xl border border-[var(--base-border)] text-xs font-bold uppercase tracking-widest text-[var(--base-text-muted)] hover:bg-[var(--base-border)] transition-all">Cancel</button>
                    <button onClick={handleSave} disabled={saving || !name.trim() || !goal.trim()} className="flex-1 py-2.5 rounded-xl bg-[var(--brand-primary)] text-white text-xs font-bold uppercase tracking-widest hover:brightness-110 transition-all flex items-center justify-center gap-2 disabled:opacity-50">
                        {saved ? <><CheckCircle size={12} /> Saved!</> : saving ? '...' : <><BookmarkPlus size={12} /> Save</>}
                    </button>
                </div>
            </motion.div>
        </div>
    );
}

export default function HistoryPage({ runs, onViewDetails, onDeleteRun }: HistoryPageProps) {
    const [savingRun, setSavingRun] = useState<RunHistory | null>(null);

    return (
        <>
            <motion.div
                key="history"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-8"
            >
                <header>
                    <h2 className="text-4xl font-bold tracking-tight mb-2">Operation History</h2>
                    <p className="text-[var(--base-text-muted)]">Audit trail of all runs. Save any completed run as a reusable workflow.</p>
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
                                                    onClick={() => setSavingRun(run)}
                                                    className="p-2 rounded-lg bg-[var(--base-border)] hover:bg-[var(--base-border)] text-[var(--base-text-muted)] hover:text-amber-400 transition-colors"
                                                    title="Save as Workflow"
                                                >
                                                    <BookmarkPlus size={16} />
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

            <AnimatePresence>
                {savingRun && (
                    <SaveFromRunModal run={savingRun} onClose={() => setSavingRun(null)} />
                )}
            </AnimatePresence>
        </>
    );
}

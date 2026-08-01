import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { Zap, Play, Plus, Trash2, BookmarkPlus, X, CheckCircle2 } from 'lucide-react';
import { BackendWorkflow, runWorkflow, saveWorkflow, deleteWorkflow } from '../services/apiService';
import { RunHistory } from '../types';

// ── Save-as-Workflow modal ────────────────────────────────────────────────────

interface SaveWorkflowModalProps {
    initialGoal?: string;
    onSave: (wf: BackendWorkflow) => void;
    onClose: () => void;
}

function SaveWorkflowModal({ initialGoal = '', onSave, onClose }: SaveWorkflowModalProps) {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [goal, setGoal] = useState(initialGoal);
    const [topicLabel, setTopicLabel] = useState('');
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');

    const handleSave = async () => {
        if (!name.trim() || !goal.trim()) { setError('Name and goal are required.'); return; }
        setSaving(true);
        setError('');
        try {
            const res = await saveWorkflow({ name: name.trim(), description: description.trim() || name.trim(), goal: goal.trim(), topic_label: topicLabel.trim() });
            onSave(res.workflow);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Failed to save workflow.');
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
                className="glass-panel rounded-3xl p-8 w-full max-w-lg space-y-5 shadow-2xl"
                onClick={e => e.stopPropagation()}
            >
                <div className="flex items-center justify-between">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <BookmarkPlus size={18} className="text-[var(--brand-primary)]" /> Save as Workflow
                    </h3>
                    <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-[var(--base-border)] text-[var(--base-text-muted)]"><X size={16} /></button>
                </div>
                <p className="text-xs text-[var(--base-text-muted)]">Turn any goal or completed run into a reusable one-click workflow.</p>

                <div className="space-y-3">
                    <div>
                        <label className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1 block">Workflow Name *</label>
                        <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Daily Email Digest" className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm focus:outline-none focus:border-[var(--brand-primary)]/50" />
                    </div>
                    <div>
                        <label className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1 block">Short Description</label>
                        <input value={description} onChange={e => setDescription(e.target.value)} placeholder="What does this workflow do?" className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm focus:outline-none focus:border-[var(--brand-primary)]/50" />
                    </div>
                    <div>
                        <label className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1 block">Goal / Instruction *</label>
                        <textarea value={goal} onChange={e => setGoal(e.target.value)} rows={4} placeholder="Paste the full goal or instruction here…" className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm font-mono resize-none focus:outline-none focus:border-[var(--brand-primary)]/50" />
                    </div>
                    <div>
                        <label className="text-[9px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1 block">Topic Label <span className="opacity-50">(optional — shown as input field on the card)</span></label>
                        <input value={topicLabel} onChange={e => setTopicLabel(e.target.value)} placeholder='e.g. "Enter a ticker symbol" — leave blank if no variable input needed' className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm focus:outline-none focus:border-[var(--brand-primary)]/50" />
                    </div>
                </div>

                {error && <p className="text-xs text-red-400">{error}</p>}

                <div className="flex gap-3 pt-1">
                    <button onClick={onClose} className="flex-1 py-2.5 rounded-xl border border-[var(--base-border)] text-xs font-bold uppercase tracking-widest text-[var(--base-text-muted)] hover:bg-[var(--base-border)] transition-all">Cancel</button>
                    <button onClick={handleSave} disabled={saving || !name.trim() || !goal.trim()} className="flex-1 py-2.5 rounded-xl bg-[var(--brand-primary)] text-white text-xs font-bold uppercase tracking-widest hover:brightness-110 transition-all flex items-center justify-center gap-2 disabled:opacity-50">
                        {saving ? '...' : <><CheckCircle2 size={12} /> Save Workflow</>}
                    </button>
                </div>
            </motion.div>
        </div>
    );
}

// ── Workflow card ─────────────────────────────────────────────────────────────

interface WorkflowCardProps {
    key?: string | number;
    workflow: BackendWorkflow;
    onRun: (topic: string) => void;
    onDelete?: () => void;
    disabled: boolean;
}

const WorkflowCard = ({ workflow, onRun, onDelete, disabled }: WorkflowCardProps) => {
    const [topic, setTopic] = useState('');
    const isUser = workflow.source === 'user';

    return (
        <div className="glass-panel p-8 rounded-3xl border-[var(--base-border)] hover:border-brand-500/30 transition-all group flex flex-col h-full relative">
            {isUser && onDelete && (
                <button
                    onClick={onDelete}
                    title="Delete workflow"
                    className="absolute top-4 right-4 p-1.5 rounded-lg text-[var(--base-text-muted)] hover:text-red-400 hover:bg-red-500/10 transition-all opacity-0 group-hover:opacity-100"
                >
                    <Trash2 size={13} />
                </button>
            )}
            <div className="flex items-start justify-between mb-6">
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors ${
                    isUser
                        ? 'bg-amber-500/20 group-hover:bg-amber-500/30'
                        : 'bg-[var(--brand-primary)]/20 group-hover:bg-[var(--brand-primary)]/20'
                }`}>
                    {isUser
                        ? <BookmarkPlus size={26} className="text-amber-400" />
                        : <Zap size={28} className="text-[var(--brand-primary)]" />
                    }
                </div>
                {isUser && (
                    <span className="text-[8px] font-bold uppercase tracking-widest text-amber-400 bg-amber-500/10 px-2 py-1 rounded-full">saved</span>
                )}
            </div>
            <h3 className="text-xl font-bold mb-3 group-hover:text-[var(--brand-primary)] transition-colors">{workflow.name}</h3>
            <p className="text-sm text-[var(--base-text-muted)] mb-6 leading-relaxed flex-1">{workflow.description}</p>
            {workflow.topic_label ? (
                <div className="mb-4">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-2">{workflow.topic_label}</label>
                    <input
                        type="text"
                        value={topic}
                        onChange={e => setTopic(e.target.value)}
                        placeholder={workflow.topic_label}
                        className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm focus:outline-none focus:border-brand-500/50"
                    />
                </div>
            ) : null}
            <button
                onClick={() => onRun(topic)}
                disabled={disabled}
                className="w-full py-4 rounded-xl bg-[var(--brand-primary)] text-white font-bold text-xs uppercase tracking-widest hover:brightness-110 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <Play size={16} fill="currentColor" /> Run
            </button>
        </div>
    );
};

// ── Page ─────────────────────────────────────────────────────────────────────

interface WorkflowsPageProps {
    workflows: BackendWorkflow[];
    isRunning: boolean;
    onRunStart: (run: RunHistory) => void;
    onRunIdUpdate: (runId: string) => void;
    onRunError: (msg: string) => void;
    onWorkflowsChange: (workflows: BackendWorkflow[]) => void;
}

export default function WorkflowsPage({ workflows, isRunning, onRunStart, onRunIdUpdate, onRunError, onWorkflowsChange }: WorkflowsPageProps) {
    const navigate = useNavigate();
    const [showSaveModal, setShowSaveModal] = useState(false);

    const handleRun = async (wf: BackendWorkflow, topic: string) => {
        navigate('/dashboard');
        onRunStart({
            id: 'pending',
            planName: wf.name,
            timestamp: new Date().toLocaleString(),
            status: 'running',
            stepsCompleted: 0,
            totalSteps: 0,
            artifacts: {},
            screenshots: [],
            logs: [`▶ Starting workflow: ${wf.name}...`],
        });
        try {
            const res = await runWorkflow(wf.id, topic);
            onRunIdUpdate(res.run_id);
        } catch (e) {
            onRunError(e instanceof Error ? e.message : 'Unknown error');
        }
    };

    const handleDelete = async (wf: BackendWorkflow) => {
        if (!confirm(`Delete "${wf.name}"?`)) return;
        try {
            await deleteWorkflow(wf.id);
            onWorkflowsChange(workflows.filter(w => w.id !== wf.id));
        } catch (e) {
            alert(e instanceof Error ? e.message : 'Delete failed');
        }
    };

    const handleSaved = (newWf: BackendWorkflow) => {
        onWorkflowsChange([...workflows, newWf]);
        setShowSaveModal(false);
    };

    const userWorkflows = workflows.filter(w => w.source === 'user');
    const builtinWorkflows = workflows.filter(w => w.source !== 'user');

    return (
        <>
            <motion.div
                key="workflows"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-8"
            >
                <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <h2 className="text-4xl font-bold tracking-tight mb-2">Workflow Library</h2>
                        <p className="text-[var(--base-text-muted)]">Launch pre-configured automation sequences. Save any run as a reusable workflow.</p>
                    </div>
                    <button
                        onClick={() => setShowSaveModal(true)}
                        className="btn-primary py-2.5 px-5 text-xs uppercase tracking-widest flex items-center gap-2 shrink-0"
                    >
                        <BookmarkPlus size={14} /> Save New Workflow
                    </button>
                </header>

                {/* User-saved workflows */}
                {userWorkflows.length > 0 && (
                    <section className="space-y-4">
                        <h3 className="text-sm font-bold uppercase tracking-widest text-[var(--base-text-muted)] flex items-center gap-2">
                            <BookmarkPlus size={13} /> My Saved Workflows
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {userWorkflows.map(wf => (
                                <WorkflowCard
                                    key={wf.id}
                                    workflow={wf}
                                    onRun={topic => handleRun(wf, topic)}
                                    onDelete={() => handleDelete(wf)}
                                    disabled={isRunning}
                                />
                            ))}
                        </div>
                    </section>
                )}

                {/* Built-in workflows */}
                <section className="space-y-4">
                    {userWorkflows.length > 0 && (
                        <h3 className="text-sm font-bold uppercase tracking-widest text-[var(--base-text-muted)] flex items-center gap-2">
                            <Zap size={13} /> Built-in Templates
                        </h3>
                    )}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {builtinWorkflows.map(wf => (
                            <WorkflowCard
                                key={wf.id}
                                workflow={wf}
                                onRun={topic => handleRun(wf, topic)}
                                disabled={isRunning}
                            />
                        ))}
                        <button
                            onClick={() => setShowSaveModal(true)}
                            className="glass-panel p-8 rounded-3xl border-dashed border-[var(--base-border)] hover:border-brand-500/30 hover:bg-[var(--brand-primary)]/20 transition-all flex flex-col items-center justify-center gap-4 group min-h-[280px]"
                        >
                            <div className="w-16 h-16 rounded-full bg-[var(--base-border)] flex items-center justify-center group-hover:bg-[var(--brand-primary)]/20 transition-colors">
                                <Plus size={32} className="text-[var(--base-text-muted)] group-hover:text-[var(--brand-primary)]" />
                            </div>
                            <div className="text-center">
                                <div className="font-bold text-lg">Save a Workflow</div>
                                <div className="text-xs text-[var(--base-text-muted)] mt-1">Turn any goal into a one-click workflow</div>
                            </div>
                        </button>
                    </div>
                </section>
            </motion.div>

            <AnimatePresence>
                {showSaveModal && (
                    <SaveWorkflowModal
                        onSave={handleSaved}
                        onClose={() => setShowSaveModal(false)}
                    />
                )}
            </AnimatePresence>
        </>
    );
}

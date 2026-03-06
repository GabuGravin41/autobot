import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, CheckCircle2, XCircle, Mail, FileText } from 'lucide-react';
import { RunHistory } from '../types';
import { BackendStatus, submitHumanInput } from '../services/apiService';
import { useState } from 'react';

interface ModalsProps {
    backendStatus: BackendStatus | null;
    selectedRun: RunHistory | null;
    onCloseRun: () => void;
    selectedArtifact: any | null;
    onCloseArtifact: () => void;
}

export default function Modals({
    backendStatus, selectedRun, onCloseRun, selectedArtifact, onCloseArtifact,
}: ModalsProps) {
    const [humanInputValue, setHumanInputValue] = useState('');
    const [humanInputSubmitting, setHumanInputSubmitting] = useState(false);

    return (
        <AnimatePresence>
            {/* Human input modal */}
            {backendStatus?.human_input_pending && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="absolute inset-0 bg-[var(--base-border)] backdrop-blur-sm" />
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        className="w-full max-w-md glass-panel p-8 rounded-3xl relative z-10"
                    >
                        <h3 className="text-xl font-bold mb-2">Input required</h3>
                        <p className="text-sm text-[var(--base-text-muted)] mb-4">{backendStatus.human_input_pending.prompt}</p>
                        <input
                            type="password"
                            value={humanInputValue}
                            onChange={e => setHumanInputValue(e.target.value)}
                            placeholder="Enter value..."
                            className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-3 px-4 text-sm focus:outline-none focus:border-brand-500/50 mb-4"
                            autoFocus
                        />
                        <button
                            onClick={async () => {
                                setHumanInputSubmitting(true);
                                try {
                                    await submitHumanInput(backendStatus!.human_input_pending!.key, humanInputValue);
                                    setHumanInputValue('');
                                } catch (e) {
                                    alert('Submit failed: ' + (e instanceof Error ? e.message : 'Unknown'));
                                } finally {
                                    setHumanInputSubmitting(false);
                                }
                            }}
                            disabled={humanInputSubmitting || !humanInputValue.trim()}
                            className="btn-primary w-full py-3 text-xs uppercase tracking-widest"
                        >
                            {humanInputSubmitting ? 'Submitting...' : 'Submit'}
                        </button>
                    </motion.div>
                </div>
            )}

            {/* Run detail modal */}
            {selectedRun && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        onClick={onCloseRun} className="absolute inset-0 bg-[var(--base-border)] backdrop-blur-sm" />
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        className="w-full max-w-2xl glass-panel p-8 rounded-3xl relative z-10 overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 p-6">
                            <button onClick={onCloseRun} className="p-2 rounded-xl bg-[var(--base-border)] hover:bg-[var(--base-border)] transition-colors">
                                <X size={20} />
                            </button>
                        </div>
                        <div className="flex items-center gap-4 mb-8">
                            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${selectedRun.status === 'success' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                {selectedRun.status === 'success' ? <CheckCircle2 size={28} /> : <XCircle size={28} />}
                            </div>
                            <div>
                                <h3 className="text-2xl font-bold">{selectedRun.planName}</h3>
                                <p className="text-[var(--base-text-muted)] text-sm">{selectedRun.timestamp} • {selectedRun.id}</p>
                            </div>
                        </div>
                        <div className="space-y-6">
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-4 rounded-2xl bg-[var(--base-border)]">
                                    <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1">Steps</div>
                                    <div className="text-xl font-bold">{selectedRun.stepsCompleted} / {selectedRun.totalSteps}</div>
                                </div>
                                <div className="p-4 rounded-2xl bg-[var(--base-border)]">
                                    <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-1">Status</div>
                                    <div className={`text-xl font-bold uppercase tracking-widest ${selectedRun.status === 'success' ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {selectedRun.status}
                                    </div>
                                </div>
                            </div>
                            <div className="space-y-3">
                                <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)]">Execution Logs</div>
                                <div className="glass-panel bg-[var(--base-border)] rounded-xl p-4 h-48 overflow-y-auto font-mono text-xs space-y-2 custom-scrollbar">
                                    {selectedRun.logs.map((log, i) => (
                                        <div key={i} className="flex gap-3">
                                            <span className="text-[var(--base-text-muted)]">[{i}]</span>
                                            <span className="text-[var(--base-text-muted)]">{log}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            )}

            {/* Artifact detail modal */}
            {selectedArtifact && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        onClick={onCloseArtifact} className="absolute inset-0 bg-[var(--base-border)] backdrop-blur-sm" />
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        className="w-full max-w-lg glass-panel p-8 rounded-3xl relative z-10"
                    >
                        <div className="absolute top-0 right-0 p-6">
                            <button onClick={onCloseArtifact} className="p-2 rounded-xl bg-[var(--base-border)] hover:bg-[var(--base-border)] transition-colors">
                                <X size={20} />
                            </button>
                        </div>
                        <div className="flex items-center gap-4 mb-8">
                            <div className="w-14 h-14 rounded-2xl bg-[var(--brand-primary)]/20 text-[var(--brand-primary)] flex items-center justify-center">
                                {selectedArtifact.type === 'email' ? <Mail size={28} /> : <FileText size={28} />}
                            </div>
                            <div>
                                <h3 className="text-2xl font-bold">{selectedArtifact.title}</h3>
                                <p className="text-[var(--base-text-muted)] text-sm">Generated {selectedArtifact.timestamp}</p>
                            </div>
                        </div>
                        <div className="p-6 rounded-2xl bg-[var(--base-border)] border border-[var(--base-border)]">
                            <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--brand-primary)] mb-4">Content Preview</div>
                            <div className="text-sm text-[var(--base-text-muted)] leading-relaxed">
                                {selectedArtifact.content || 'No preview available.'}
                            </div>
                        </div>
                        <div className="flex gap-3 mt-6">
                            <button className="btn-primary flex-1 py-3 text-xs uppercase tracking-widest">Download</button>
                            <button className="btn-secondary flex-1 py-3 text-xs uppercase tracking-widest">Share</button>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
}

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { Zap, Play, Plus } from 'lucide-react';
import { BackendWorkflow, runWorkflow } from '../services/apiService';
import { RunHistory } from '../types';

interface WorkflowCardProps {
    key?: string | number; // Required for React 19
    workflow: BackendWorkflow;
    onRun: (topic: string) => void;
    disabled: boolean;
}

const WorkflowCard = ({ workflow, onRun, disabled }: WorkflowCardProps) => {
    const [topic, setTopic] = useState('');
    return (
        <div className="glass-panel p-8 rounded-3xl border-white/5 hover:border-brand-500/30 transition-all group flex flex-col h-full">
            <div className="flex items-start justify-between mb-6">
                <div className="w-14 h-14 rounded-2xl bg-brand-500/10 flex items-center justify-center group-hover:bg-brand-500/20 transition-colors">
                    <Zap size={28} className="text-brand-400" />
                </div>
            </div>
            <h3 className="text-xl font-bold mb-3 group-hover:text-brand-400 transition-colors">{workflow.name}</h3>
            <p className="text-sm text-white/40 mb-6 leading-relaxed flex-1">{workflow.description}</p>
            {workflow.topic_label ? (
                <div className="mb-4">
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-white/40 mb-2">{workflow.topic_label}</label>
                    <input
                        type="text"
                        value={topic}
                        onChange={e => setTopic(e.target.value)}
                        placeholder={workflow.topic_label}
                        className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-sm focus:outline-none focus:border-brand-500/50"
                    />
                </div>
            ) : null}
            <button
                onClick={() => onRun(topic)}
                disabled={disabled}
                className="w-full py-4 rounded-xl bg-brand-500 text-black font-bold text-xs uppercase tracking-widest hover:bg-brand-400 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <Play size={16} fill="currentColor" /> Initialize Run
            </button>
        </div>
    );
};

interface WorkflowsPageProps {
    workflows: BackendWorkflow[];
    isRunning: boolean;
    onRunStart: (run: RunHistory) => void;
    onRunIdUpdate: (runId: string) => void;
    onRunError: (msg: string) => void;
}

export default function WorkflowsPage({ workflows, isRunning, onRunStart, onRunIdUpdate, onRunError }: WorkflowsPageProps) {
    const navigate = useNavigate();

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

    return (
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
                    <p className="text-white/40">Launch pre-configured automation sequences from the backend.</p>
                </div>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {workflows.map(wf => (
                    <WorkflowCard
                        key={wf.id}
                        workflow={wf}
                        onRun={topic => handleRun(wf, topic)}
                        disabled={isRunning}
                    />
                ))}
                <button
                    onClick={() => navigate('/planner')}
                    className="glass-panel p-8 rounded-3xl border-dashed border-white/10 hover:border-brand-500/30 hover:bg-brand-500/5 transition-all flex flex-col items-center justify-center gap-4 group min-h-[320px]"
                >
                    <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-brand-500/10 transition-colors">
                        <Plus size={32} className="text-white/20 group-hover:text-brand-400" />
                    </div>
                    <div className="text-center">
                        <div className="font-bold text-lg">Custom Workflow</div>
                        <div className="text-xs text-white/40 mt-1">Use AI Planner to build a custom sequence</div>
                    </div>
                </button>
            </div>
        </motion.div>
    );
}

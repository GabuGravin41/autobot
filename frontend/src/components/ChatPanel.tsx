import React, { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Send, Loader2, Eye, EyeOff, AlertCircle, Play, Globe } from 'lucide-react';
import { WorkflowPlan } from '../types';

interface ChatMessage {
    role: 'user' | 'bot';
    content: string;
    plan?: WorkflowPlan;
    artifact?: any;
}

interface ChatPanelProps {
    messages: ChatMessage[];
    input: string;
    setInput: (v: string) => void;
    isGenerating: boolean;
    onSend: () => void;
    onExecutePlan: (plan: WorkflowPlan) => void;
    showReasoning: Record<number, boolean>;
    setShowReasoning: React.Dispatch<React.SetStateAction<Record<number, boolean>>>;
}

export default function ChatPanel({
    messages, input, setInput, isGenerating, onSend, onExecutePlan,
    showReasoning, setShowReasoning,
}: ChatPanelProps) {
    const chatEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <motion.div
            key="planner"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            className="h-[calc(100vh-8rem)] flex flex-col"
        >
            <header className="mb-6">
                <h2 className="text-3xl font-bold mb-2">AI Planner</h2>
                <p className="text-white/40">Describe your task and I'll build a workflow for you.</p>
            </header>

            <div className="flex-1 glass-panel rounded-3xl flex flex-col overflow-hidden">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.map((msg, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div className={`max-w-[80%] p-4 rounded-2xl ${msg.role === 'user'
                                ? 'bg-brand-500 text-black font-medium'
                                : 'bg-white/5 border border-white/10'
                                }`}>
                                <p className="text-sm leading-relaxed">{msg.content}</p>

                                {/* Email artifact */}
                                {msg.artifact?.type === 'email' && (
                                    <div className="mt-4 p-4 rounded-xl bg-white/5 border border-white/10 space-y-3">
                                        <div className="flex justify-between items-start">
                                            <div className="flex items-center gap-2">
                                                <div className="p-1.5 rounded-lg bg-red-500/10 text-red-400"><AlertCircle size={14} /></div>
                                                <span className="text-[10px] font-bold uppercase tracking-widest text-red-400">Urgent Email</span>
                                            </div>
                                        </div>
                                        <div>
                                            <h4 className="text-sm font-bold text-white/90">{msg.artifact.subject}</h4>
                                            <p className="text-[10px] text-white/40">From: {msg.artifact.from}</p>
                                        </div>
                                        <p className="text-xs text-white/60 leading-relaxed italic">"{msg.artifact.summary}"</p>
                                    </div>
                                )}

                                {/* Reasoning toggle */}
                                {msg.role === 'bot' && (
                                    <div className="mt-3 border-t border-white/5 pt-3">
                                        <button
                                            onClick={() => setShowReasoning(prev => ({ ...prev, [i]: !prev[i] }))}
                                            className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-white/30 hover:text-brand-400 transition-colors"
                                        >
                                            {showReasoning[i] ? <EyeOff size={12} /> : <Eye size={12} />}
                                            {showReasoning[i] ? 'Hide Reasoning' : 'View Reasoning'}
                                        </button>
                                        <AnimatePresence>
                                            {showReasoning[i] && (
                                                <motion.div
                                                    initial={{ height: 0, opacity: 0 }}
                                                    animate={{ height: 'auto', opacity: 1 }}
                                                    exit={{ height: 0, opacity: 0 }}
                                                    className="overflow-hidden"
                                                >
                                                    <div className="mt-2 p-3 rounded-lg bg-black/40 border border-white/5 text-[11px] font-mono text-white/40 leading-relaxed">
                                                        <span className="text-accent-amber">{"<think>"}</span>
                                                        {" Analyzing user intent... Mapping to available adapters... Constructing multi-step workflow plan."}
                                                        <span className="text-accent-amber">{"</think>"}</span>
                                                    </div>
                                                </motion.div>
                                            )}
                                        </AnimatePresence>
                                    </div>
                                )}

                                {/* Plan card */}
                                {msg.plan && (
                                    <div className="mt-4 p-4 rounded-xl bg-black/40 border border-white/10 space-y-4">
                                        <div className="flex items-center justify-between">
                                            <h4 className="text-xs font-bold uppercase tracking-widest text-brand-400">Proposed Plan</h4>
                                            <span className="text-[10px] text-white/40">{msg.plan.steps?.length || 0} steps</span>
                                        </div>
                                        <div className="space-y-2">
                                            {msg.plan.steps?.map((step, si) => (
                                                <div key={si} className="flex gap-3 text-xs">
                                                    <span className="text-white/20 font-mono">{si + 1}.</span>
                                                    <div className="flex-1">
                                                        <span className="text-white/70">{step?.description || 'No description'}</span>
                                                        {step?.target_node && (
                                                            <span className="ml-2 inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[8px] font-bold uppercase tracking-widest border border-emerald-500/20">
                                                                <Globe size={8} /> {step.target_node}
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <button
                                            onClick={() => onExecutePlan(msg.plan!)}
                                            className="w-full py-2 rounded-lg bg-brand-500 text-black text-xs font-bold hover:bg-brand-400 transition-colors flex items-center justify-center gap-2"
                                        >
                                            <Play size={14} /> Execute Plan
                                        </button>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ))}

                    {isGenerating && (
                        <div className="flex justify-start">
                            <div className="bg-white/5 border border-white/10 p-4 rounded-2xl flex items-center gap-3">
                                <Loader2 size={16} className="animate-spin text-brand-400" />
                                <span className="text-sm text-white/60 italic">Thinking...</span>
                            </div>
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </div>

                {/* Input bar */}
                <div className="p-6 bg-white/[0.02] border-t border-white/5">
                    <div className="relative">
                        <input
                            type="text"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && onSend()}
                            placeholder={isGenerating ? "Autobot is thinking..." : "e.g. Find the most urgent email and send me a summary on WhatsApp"}
                            className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-6 pr-16 text-sm focus:outline-none focus:border-brand-500/50 transition-colors"
                        />
                        <button
                            onClick={onSend}
                            disabled={isGenerating || !input.trim()}
                            className={`absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 rounded-xl flex items-center justify-center text-black transition-all ${isGenerating ? 'bg-white/10 text-white/40' : 'bg-brand-500 hover:bg-brand-400'}`}
                        >
                            {isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
                        </button>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

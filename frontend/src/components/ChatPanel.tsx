import React, { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Send, Loader2, AlertCircle, Play, Globe, Bot, User, Trash2 } from 'lucide-react';
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
    onClearChat?: () => void;
}

export default function ChatPanel({
    messages, input, setInput, isGenerating, onSend, onExecutePlan,
    showReasoning, setShowReasoning, onClearChat,
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
            <header className="mb-6 flex items-start justify-between">
                <div>
                    <h2 className="text-3xl font-bold mb-2">AI Planner</h2>
                    <p className="text-[var(--base-text-muted)]">Chat with Autobot to plan your task. I'll ask questions if I need clarity, then propose a plan you can review and execute.</p>
                </div>
                {onClearChat && messages.length > 1 && (
                    <button
                        onClick={onClearChat}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] hover:text-red-400 hover:bg-red-500/10 transition-colors"
                        title="Start new conversation"
                    >
                        <Trash2 size={12} /> New Chat
                    </button>
                )}
            </header>

            <div className="flex-1 glass-panel rounded-3xl flex flex-col overflow-hidden">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.map((msg, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            {/* Bot avatar */}
                            {msg.role === 'bot' && (
                                <div className="w-8 h-8 rounded-xl bg-[var(--brand-primary)]/20 flex items-center justify-center shrink-0 mt-1">
                                    <Bot size={16} className="text-[var(--brand-primary)]" />
                                </div>
                            )}

                            <div className={`max-w-[75%] p-4 rounded-2xl ${msg.role === 'user'
                                ? 'bg-[var(--brand-primary)] text-white font-medium'
                                : 'bg-[var(--base-border)] border border-[var(--base-border)]'
                                }`}>
                                {/* Render message with line breaks */}
                                <div className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</div>

                                {/* Email artifact */}
                                {msg.artifact?.type === 'email' && (
                                    <div className="mt-4 p-4 rounded-xl bg-[var(--base-border)] border border-[var(--base-border)] space-y-3">
                                        <div className="flex justify-between items-start">
                                            <div className="flex items-center gap-2">
                                                <div className="p-1.5 rounded-lg bg-red-500/10 text-red-400"><AlertCircle size={14} /></div>
                                                <span className="text-[10px] font-bold uppercase tracking-widest text-red-400">Urgent Email</span>
                                            </div>
                                        </div>
                                        <div>
                                            <h4 className="text-sm font-bold text-[var(--base-text-muted)]">{msg.artifact.subject}</h4>
                                            <p className="text-[10px] text-[var(--base-text-muted)]">From: {msg.artifact.from}</p>
                                        </div>
                                        <p className="text-xs text-[var(--base-text-muted)] leading-relaxed italic">"{msg.artifact.summary}"</p>
                                    </div>
                                )}

                                {/* Plan card */}
                                {msg.plan && (
                                    <div className="mt-4 p-4 rounded-xl bg-[var(--base-border)] border border-[var(--base-border)] space-y-4">
                                        <div className="flex items-center justify-between">
                                            <h4 className="text-xs font-bold uppercase tracking-widest text-[var(--brand-primary)]">Proposed Plan</h4>
                                            <span className="text-[10px] text-[var(--base-text-muted)]">{msg.plan.steps?.length || 0} steps</span>
                                        </div>
                                        <div className="text-[11px] text-[var(--base-text-muted)] mb-2 font-medium">{msg.plan.name}</div>
                                        <div className="space-y-2">
                                            {msg.plan.steps?.map((step, si) => (
                                                <div key={si} className="flex gap-3 text-xs">
                                                    <span className="w-5 h-5 rounded-full bg-[var(--brand-primary)]/20 text-[var(--brand-primary)] flex items-center justify-center text-[9px] font-bold shrink-0">{si + 1}</span>
                                                    <div className="flex-1 pt-0.5">
                                                        <span className="text-[var(--base-text-muted)]">{step?.description || 'No description'}</span>
                                                        {step?.target_node && (
                                                            <span className="ml-2 inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[8px] font-bold uppercase tracking-widest border border-emerald-500/20">
                                                                <Globe size={8} /> {step.target_node}
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="flex gap-2 pt-2">
                                            <button
                                                onClick={() => onExecutePlan(msg.plan!)}
                                                className="flex-1 py-2.5 rounded-lg bg-[var(--brand-primary)] text-white text-xs font-bold hover:brightness-110 transition-all flex items-center justify-center gap-2"
                                            >
                                                <Play size={14} /> Execute Plan
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* User avatar */}
                            {msg.role === 'user' && (
                                <div className="w-8 h-8 rounded-xl bg-[var(--brand-primary)]/40 flex items-center justify-center shrink-0 mt-1">
                                    <User size={16} className="text-white" />
                                </div>
                            )}
                        </motion.div>
                    ))}

                    {isGenerating && (
                        <div className="flex gap-3 justify-start">
                            <div className="w-8 h-8 rounded-xl bg-[var(--brand-primary)]/20 flex items-center justify-center shrink-0">
                                <Bot size={16} className="text-[var(--brand-primary)]" />
                            </div>
                            <div className="bg-[var(--base-border)] border border-[var(--base-border)] p-4 rounded-2xl flex items-center gap-3">
                                <Loader2 size={16} className="animate-spin text-[var(--brand-primary)]" />
                                <span className="text-sm text-[var(--base-text-muted)] italic">Planning...</span>
                            </div>
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </div>

                {/* Input bar */}
                <div className="p-4 md:p-6 bg-white/[0.02] border-t border-[var(--base-border)]">
                    <div className="relative">
                        <textarea
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    onSend();
                                }
                            }}
                            placeholder={isGenerating ? "Autobot is thinking..." : "Describe what you want Autobot to do... (Enter to send, Shift+Enter for new line)"}
                            className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-2xl py-3 pl-5 pr-14 text-sm focus:outline-none focus:border-brand-500/50 transition-colors resize-none min-h-[48px] max-h-[120px]"
                            rows={1}
                            style={{ height: 'auto' }}
                            onInput={(e) => {
                                const t = e.currentTarget;
                                t.style.height = 'auto';
                                t.style.height = Math.min(t.scrollHeight, 120) + 'px';
                            }}
                        />
                        <button
                            onClick={onSend}
                            disabled={isGenerating || !input.trim()}
                            className={`absolute right-3 bottom-2 w-10 h-10 rounded-xl flex items-center justify-center text-white transition-all ${isGenerating ? 'bg-[var(--base-border)] text-[var(--base-text-muted)]' : 'bg-[var(--brand-primary)] hover:brightness-110'}`}
                        >
                            {isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
                        </button>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

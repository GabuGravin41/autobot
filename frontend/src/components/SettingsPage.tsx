import React, { useState } from 'react';
import { motion } from 'motion/react';
import {
    Shield, Zap, Palette, Sparkles, PlusCircle, CheckCircle2, Key,
} from 'lucide-react';
import { BrowserMode, AdapterPolicy, LLMModel } from '../types';
import { updateSettings } from '../services/apiService';

type Theme = 'beam' | 'blue-violet' | 'emerald' | 'blue' | 'amber';

interface SettingsPageProps {
    browserMode: BrowserMode;
    setBrowserMode: (m: BrowserMode) => void;
    policy: AdapterPolicy;
    setPolicy: (p: AdapterPolicy) => void;
    theme: Theme;
    setTheme: (t: Theme) => void;
    models: LLMModel[];
    setModels: React.Dispatch<React.SetStateAction<LLMModel[]>>;
    selectedModelId: string;
    setSelectedModelId: (id: string) => void;
}

export default function SettingsPage({
    browserMode, setBrowserMode, policy, setPolicy, theme, setTheme,
    models, setModels, selectedModelId, setSelectedModelId,
}: SettingsPageProps) {
    const [apiKey, setApiKey] = useState('');
    const [isApiKeySaved, setIsApiKeySaved] = useState(false);

    const handleSaveApiKey = async () => {
        if (!apiKey.trim()) return;
        try {
            const provider = selectedModelId.includes('gemini') ? 'gemini'
                : selectedModelId.includes('deepseek') ? 'openrouter' : 'openai';
            const updates: any = { llm_provider: provider, llm_model: selectedModelId };
            if (provider === 'openrouter') updates.openrouter_api_key = apiKey;
            if (provider === 'openai') updates.openai_api_key = apiKey;
            await updateSettings(updates);
            setIsApiKeySaved(true);
            setApiKey('');
            alert('Settings saved and LLM Brain updated.');
        } catch (e) {
            alert('Failed to save settings: ' + (e instanceof Error ? e.message : 'Unknown error'));
        }
    };

    const handleAddModel = () => {
        const name = prompt('Enter model name:');
        if (name) {
            setModels(prev => [...prev, {
                id: name.toLowerCase().replace(/\s+/g, '-'),
                name,
                provider: 'Custom',
                isCustom: true,
            }]);
        }
    };

    return (
        <motion.div
            key="settings"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-10"
        >
            <header>
                <h2 className="text-4xl font-bold tracking-tight mb-2">System Settings</h2>
                <p className="text-white/40">Configure your Autobot instance and security preferences.</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Browser Mode */}
                <section className="glass-panel p-8 rounded-3xl space-y-6">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <Shield size={20} className="text-brand-400" /> Security &amp; Policy
                    </h3>
                    <div>
                        <label className="block text-xs font-bold uppercase tracking-widest text-white/40 mb-3">Browser Mode</label>
                        <button
                            onClick={async () => {
                                setBrowserMode(BrowserMode.HUMAN_PROFILE);
                                await updateSettings({ browser_mode: 'human_profile' });
                            }}
                            className={`w-full p-4 rounded-xl border text-left transition-all ${browserMode === BrowserMode.HUMAN_PROFILE
                                ? 'bg-brand-500/10 border-brand-500/50 text-brand-400'
                                : 'bg-white/5 border-white/5 text-white/60 hover:bg-white/10'}`}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-bold">Human Profile</span>
                                {browserMode === BrowserMode.HUMAN_PROFILE && <CheckCircle2 size={16} />}
                            </div>
                            <p className="text-[10px] leading-relaxed opacity-60">Uses your existing Chrome profile with real cookies and history.</p>
                        </button>
                    </div>
                </section>

                {/* Adapter Policy */}
                <section className="glass-panel p-8 rounded-3xl space-y-6">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <Shield size={20} className="text-brand-400" /> Adapter Policy
                    </h3>
                    <div>
                        <label className="block text-xs font-bold uppercase tracking-widest text-white/40 mb-3">Policy Level</label>
                        <div className="grid grid-cols-3 gap-2">
                            {[AdapterPolicy.STRICT, AdapterPolicy.BALANCED, AdapterPolicy.TRUSTED].map(p => (
                                <button
                                    key={p}
                                    onClick={() => setPolicy(p)}
                                    className={`py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider border transition-all ${policy === p
                                        ? 'bg-brand-500/10 border-brand-500 text-brand-400'
                                        : 'bg-white/5 border-white/5 text-white/40 hover:bg-white/10'}`}
                                >{p}</button>
                            ))}
                        </div>
                        <div className="mt-4 p-4 rounded-xl bg-brand-500/5 border border-brand-500/10">
                            <div className="flex items-center gap-2 mb-2"><Zap size={14} className="text-brand-400" /><span className="text-xs font-bold text-brand-400">Policy Impact</span></div>
                            <p className="text-[10px] text-white/60 leading-relaxed">
                                {policy === AdapterPolicy.STRICT && "All sensitive actions require manual confirmation."}
                                {policy === AdapterPolicy.BALANCED && "Sensitive actions are allowed if they match the current plan context."}
                                {policy === AdapterPolicy.TRUSTED && "No restrictions. Autobot will execute all actions autonomously."}
                            </p>
                        </div>
                    </div>
                </section>

                {/* Theming */}
                <section className="glass-panel p-8 rounded-3xl space-y-6 md:col-span-2">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <Palette size={20} className="text-brand-400" /> Interface &amp; Theming
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-4">
                            <label className="text-xs font-bold text-white/40 uppercase tracking-widest">Accent Color</label>
                            <div className="grid grid-cols-3 gap-3">
                                {(['beam', 'blue-violet', 'emerald', 'blue', 'amber'] as const).map(t => (
                                    <button
                                        key={t}
                                        onClick={() => setTheme(t)}
                                        className={`p-4 rounded border transition-all flex flex-col items-center gap-2 ${theme === t
                                            ? 'bg-brand-500/10 border-brand-500 shadow-[0_0_20px_rgba(var(--brand-500-rgb),0.2)]'
                                            : 'bg-white/5 border-white/10 hover:bg-white/10'}`}
                                    >
                                        <div className={`w-6 h-6 rounded ${t === 'beam' ? 'bg-gradient-to-br from-cyan-400 to-violet-500'
                                            : t === 'blue-violet' ? 'bg-[#8b5cf6]'
                                                : t === 'emerald' ? 'bg-[#22c55e]'
                                                    : t === 'blue' ? 'bg-[#3b82f6]' : 'bg-[#f59e0b]'}`} />
                                        <span className="text-[10px] font-bold uppercase tracking-widest capitalize">{t.replace('-', ' ')}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>

                {/* AI Configuration */}
                <section className="glass-panel p-8 rounded-3xl space-y-6 md:col-span-2">
                    <div className="flex items-center justify-between">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <Sparkles size={20} className="text-brand-400" /> AI Configuration
                        </h3>
                        <button onClick={handleAddModel} className="btn-secondary py-2 px-4 text-[10px] uppercase tracking-widest flex items-center gap-2">
                            <PlusCircle size={14} /> Add Model
                        </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-4">
                            <label className="text-xs font-bold text-white/40 uppercase tracking-widest">Selected Model</label>
                            <div className="space-y-2">
                                {models.map(model => (
                                    <button
                                        key={model.id}
                                        onClick={() => setSelectedModelId(model.id)}
                                        className={`w-full p-4 rounded-2xl border text-left transition-all flex items-center justify-between ${selectedModelId === model.id
                                            ? 'bg-brand-500/10 border-brand-500 text-brand-400'
                                            : 'bg-white/5 border-white/10 hover:bg-white/10'}`}
                                    >
                                        <div>
                                            <div className="text-sm font-bold">{model.name}</div>
                                            <div className="text-[10px] opacity-60 uppercase tracking-widest">{model.provider}</div>
                                        </div>
                                        {selectedModelId === model.id && <CheckCircle2 size={16} />}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="space-y-4">
                            <label className="text-xs font-bold text-white/40 uppercase tracking-widest">API Authentication</label>
                            <div className="glass-card p-6 rounded-2xl space-y-4">
                                <div className="flex items-center gap-3 text-brand-400 mb-2">
                                    <Key size={18} /><span className="text-sm font-bold">Secure Token Vault</span>
                                </div>
                                <p className="text-[10px] text-white/40 leading-relaxed">
                                    Tokens are encrypted and stored in the secure backend vault.
                                </p>
                                <div className="relative">
                                    <input
                                        type="password"
                                        placeholder={isApiKeySaved ? "••••••••••••••••" : "Enter API Token..."}
                                        value={apiKey}
                                        onChange={e => setApiKey(e.target.value)}
                                        disabled={isApiKeySaved}
                                        className="input-field pr-12"
                                    />
                                    {isApiKeySaved ? (
                                        <div className="absolute right-3 top-1/2 -translate-y-1/2 text-emerald-500"><CheckCircle2 size={18} /></div>
                                    ) : (
                                        <button
                                            onClick={handleSaveApiKey}
                                            disabled={!apiKey.trim()}
                                            className="absolute right-2 top-1/2 -translate-y-1/2 btn-primary py-1.5 px-3 text-[10px] uppercase tracking-widest"
                                        >Save</button>
                                    )}
                                </div>
                                {isApiKeySaved && (
                                    <button onClick={() => setIsApiKeySaved(false)} className="text-[10px] font-bold uppercase tracking-widest text-white/20 hover:text-red-400 transition-colors">
                                        Reset Token
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </motion.div>
    );
}

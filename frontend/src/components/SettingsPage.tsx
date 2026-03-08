import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import {
    Shield, Zap, Palette, Sparkles, PlusCircle, CheckCircle2, Key, MousePointer2
} from 'lucide-react';
import { BrowserMode, AdapterPolicy, LLMModel } from '../types';
import { updateSettings, toggleAntiSleep, getStatus } from '../services/apiService';

type Theme = 'light' | 'dark';

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
    const [antiSleepEnabled, setAntiSleepEnabled] = useState(false);

    useEffect(() => {
        getStatus().then(status => {
            if (status.anti_sleep_enabled !== undefined) {
                setAntiSleepEnabled(status.anti_sleep_enabled);
            }
        });
    }, []);

    const handleToggleAntiSleep = async () => {
        try {
            const newState = !antiSleepEnabled;
            await toggleAntiSleep(newState);
            setAntiSleepEnabled(newState);
        } catch (e) {
            alert('Failed to toggle anti-sleep: ' + (e instanceof Error ? e.message : 'Unknown error'));
        }
    };

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
                <p className="text-[var(--base-text-muted)]">Configure your Autobot instance and security preferences.</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Browser Mode */}
                <section className="glass-panel p-8 rounded-3xl space-y-6">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <Shield size={20} className="text-[var(--brand-primary)]" /> Security &amp; Policy
                    </h3>
                    <div>
                        <label className="block text-xs font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-3">Browser Mode</label>
                        <button
                            onClick={async () => {
                                setBrowserMode(BrowserMode.HUMAN_PROFILE);
                                await updateSettings({ browser_mode: 'human_profile' });
                            }}
                            className={`w-full p-4 rounded-xl border text-left transition-all ${browserMode === BrowserMode.HUMAN_PROFILE
                                ? 'bg-[var(--brand-primary)]/20 border-brand-500/50 text-[var(--brand-primary)]'
                                : 'bg-[var(--base-border)] border-[var(--base-border)] text-[var(--base-text-muted)] hover:bg-[var(--base-border)]'}`}
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
                        <Shield size={20} className="text-[var(--brand-primary)]" /> Adapter Policy
                    </h3>
                    <div>
                        <label className="block text-xs font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-3">Policy Level</label>
                        <div className="grid grid-cols-3 gap-2">
                            {[AdapterPolicy.STRICT, AdapterPolicy.BALANCED, AdapterPolicy.TRUSTED].map(p => (
                                <button
                                    key={p}
                                    onClick={() => setPolicy(p)}
                                    className={`py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider border transition-all ${policy === p
                                        ? 'bg-[var(--brand-primary)]/20 border-brand-500 text-[var(--brand-primary)]'
                                        : 'bg-[var(--base-border)] border-[var(--base-border)] text-[var(--base-text-muted)] hover:bg-[var(--base-border)]'}`}
                                >{p}</button>
                            ))}
                        </div>
                        <div className="mt-4 p-4 rounded-xl bg-[var(--brand-primary)]/20 border border-brand-500/10">
                            <div className="flex items-center gap-2 mb-2"><Zap size={14} className="text-[var(--brand-primary)]" /><span className="text-xs font-bold text-[var(--brand-primary)]">Policy Impact</span></div>
                            <p className="text-[10px] text-[var(--base-text-muted)] leading-relaxed">
                                {policy === AdapterPolicy.STRICT && "All sensitive actions require manual confirmation."}
                                {policy === AdapterPolicy.BALANCED && "Sensitive actions are allowed if they match the current plan context."}
                                {policy === AdapterPolicy.TRUSTED && "No restrictions. Autobot will execute all actions autonomously."}
                            </p>
                        </div>
                    </div>
                </section>

                {/* Anti-Sleep Mode */}
                <section className="glass-panel p-8 rounded-3xl space-y-6">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <MousePointer2 size={20} className="text-[var(--brand-primary)]" /> System Utilities
                    </h3>
                    <div className="flex items-center justify-between p-4 rounded-2xl bg-[var(--base-border)] border border-[var(--base-border)]">
                        <div>
                            <span className="block text-sm font-bold capitalize">Anti-Sleep Mode</span>
                            <p className="text-[10px] text-[var(--base-text-muted)] leading-relaxed">
                                Prevents computer sleep by nudging the mouse every 60s.
                            </p>
                        </div>
                        <button
                            onClick={handleToggleAntiSleep}
                            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${antiSleepEnabled ? 'bg-[var(--brand-primary)]' : 'bg-gray-600'}`}
                        >
                            <span
                                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${antiSleepEnabled ? 'translate-x-6' : 'translate-x-1'}`}
                            />
                        </button>
                    </div>
                </section>

                {/* Theming */}
                <section className="glass-panel p-8 rounded-3xl space-y-6 md:col-span-2">
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        <Palette size={20} className="text-[var(--brand-primary)]" /> Appearance
                    </h3>
                    <div className="space-y-4">
                        <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest">Theme Mode</label>
                        <div className="flex items-center gap-3">
                            {(['light', 'dark'] as Theme[]).map(t => (
                                <button
                                    key={t}
                                    onClick={() => setTheme(t)}
                                    className={`px-4 py-2 rounded-xl border transition-all flex items-center gap-2 group ${theme === t
                                        ? 'bg-[var(--brand-primary)]/10 border-[var(--brand-primary)] text-[var(--brand-primary)]'
                                        : 'bg-[var(--base-border)] border-[var(--base-border)] hover:bg-[var(--base-border)] text-[var(--base-text-muted)]'}`}
                                >
                                    <div className={`w-3 h-3 rounded-full border border-black/20 ${t === 'light' ? 'bg-white' : 'bg-[#0a0a0c]'}`} />
                                    <span className="text-xs font-bold uppercase tracking-widest capitalize transition-colors">{t}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </section>

                {/* AI Configuration */}
                <section className="glass-panel p-8 rounded-3xl space-y-6 md:col-span-2">
                    <div className="flex items-center justify-between">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <Sparkles size={20} className="text-[var(--brand-primary)]" /> AI Configuration
                        </h3>
                        <button onClick={handleAddModel} className="btn-secondary py-2 px-4 text-[10px] uppercase tracking-widest flex items-center gap-2">
                            <PlusCircle size={14} /> Add Model
                        </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-4">
                            <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest">Selected Model</label>
                            <div className="space-y-2">
                                {models.map(model => (
                                    <button
                                        key={model.id}
                                        onClick={() => setSelectedModelId(model.id)}
                                        className={`w-full p-4 rounded-2xl border text-left transition-all flex items-center justify-between ${selectedModelId === model.id
                                            ? 'bg-[var(--brand-primary)]/20 border-brand-500 text-[var(--brand-primary)]'
                                            : 'bg-[var(--base-border)] border-[var(--base-border)] hover:bg-[var(--base-border)]'}`}
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
                            <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest">API Authentication</label>
                            <div className="glass-card p-6 rounded-2xl space-y-4">
                                <div className="flex items-center gap-3 text-[var(--brand-primary)] mb-2">
                                    <Key size={18} /><span className="text-sm font-bold">Secure Token Vault</span>
                                </div>
                                <p className="text-[10px] text-[var(--base-text-muted)] leading-relaxed">
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
                                    <button onClick={() => setIsApiKeySaved(false)} className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] hover:text-red-400 transition-colors">
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

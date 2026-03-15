import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import {
    Shield, Zap, Palette, Sparkles, PlusCircle, CheckCircle2, Key, MousePointer2,
    Trash2, Save, Eye, EyeOff, Server, Globe, AlertCircle,
} from 'lucide-react';
import { BrowserMode, AdapterPolicy, LLMModel } from '../types';
import { updateSettings, toggleAntiSleep, getStatus, getSettings, startTunnel, stopTunnel, getTunnelStatus } from '../services/apiService';

type Theme = 'light' | 'dark';

interface ProviderConfig {
    id: string;
    name: string;
    description: string;
    envKey: string;
    baseUrl: string;
    models: { id: string; name: string }[];
}

const PROVIDERS: ProviderConfig[] = [
    {
        id: 'google', name: 'Google Gemini', description: 'Free tier available. Best for vision + reasoning.',
        envKey: 'GOOGLE_API_KEY', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
        models: [
            { id: 'gemini-2.5-flash-preview-05-20', name: 'Gemini 2.5 Flash Preview' },
            { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash' },
            { id: 'gemini-2.0-flash-lite', name: 'Gemini 2.0 Flash Lite (Free)' },
            { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
        ],
    },
    {
        id: 'openrouter', name: 'OpenRouter', description: 'Access 200+ models from one API key.',
        envKey: 'OPENROUTER_API_KEY', baseUrl: 'https://openrouter.ai/api/v1',
        models: [
            { id: 'google/gemini-2.5-flash-preview', name: 'Gemini 2.5 Flash (via OR)' },
            { id: 'deepseek/deepseek-chat-v3-0324', name: 'DeepSeek V3' },
            { id: 'anthropic/claude-3.5-haiku', name: 'Claude 3.5 Haiku' },
            { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini' },
        ],
    },
    {
        id: 'openai', name: 'OpenAI', description: 'GPT-4o and newer models.',
        envKey: 'OPENAI_API_KEY', baseUrl: 'https://api.openai.com/v1',
        models: [
            { id: 'gpt-4o', name: 'GPT-4o' },
            { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
        ],
    },
    {
        id: 'xai', name: 'xAI (Grok)', description: 'Grok models via x.ai API.',
        envKey: 'XAI_API_KEY', baseUrl: 'https://api.x.ai/v1',
        models: [
            { id: 'grok-2-vision-1212', name: 'Grok 2 Vision' },
            { id: 'grok-beta', name: 'Grok Beta' },
        ],
    },
];

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
    const [antiSleepEnabled, setAntiSleepEnabled] = useState(false);
    const [selectedProvider, setSelectedProvider] = useState('google');
    const [modelInput, setModelInput] = useState('');
    const [apiKeyInput, setApiKeyInput] = useState('');
    const [showApiKey, setShowApiKey] = useState(false);
    const [saving, setSaving] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);
    const [currentConfig, setCurrentConfig] = useState({ provider: '', model: '', hasKey: false });
    const [customModels, setCustomModels] = useState<{ id: string; name: string }[]>([]);
    const [tunnelActive, setTunnelActive] = useState(false);
    const [tunnelUrl, setTunnelUrl] = useState<string | null>(null);
    const [tunnelLoading, setTunnelLoading] = useState(false);

    // Load current config from backend
    useEffect(() => {
        getTunnelStatus().then(t => {
            setTunnelActive(t.active);
            setTunnelUrl(t.url);
        }).catch(() => {});
        Promise.all([getStatus(), getSettings()]).then(([status, settings]) => {
            if (status.anti_sleep_enabled !== undefined) setAntiSleepEnabled(status.anti_sleep_enabled);
            setCurrentConfig({
                provider: settings.llm_provider || 'google',
                model: settings.llm_model || '',
                hasKey: settings.has_google_key || settings.has_openrouter_key || settings.has_openai_key,
            });
            // Set initial provider from saved config
            if (settings.llm_provider) setSelectedProvider(settings.llm_provider);
            if (settings.llm_model) setModelInput(settings.llm_model);
            if (settings.approval_mode) setPolicy(settings.approval_mode as any);
        }).catch(() => {});
    }, []);

    const provider = PROVIDERS.find(p => p.id === selectedProvider) || PROVIDERS[0];
    const allModels = [...provider.models, ...customModels];

    const handleToggleAntiSleep = async () => {
        try {
            const newState = !antiSleepEnabled;
            await toggleAntiSleep(newState);
            setAntiSleepEnabled(newState);
        } catch (e) {
            alert('Failed to toggle anti-sleep: ' + (e instanceof Error ? e.message : 'Unknown error'));
        }
    };

    const handleSave = async () => {
        if (!modelInput.trim()) {
            alert('Please select or enter a model name.');
            return;
        }
        setSaving(true);
        setSaveSuccess(false);
        try {
            const updates: Record<string, string> = {
                llm_provider: selectedProvider,
                llm_model: modelInput.trim(),
            };
            if (apiKeyInput.trim()) {
                if (selectedProvider === 'google') updates.google_api_key = apiKeyInput.trim();
                else if (selectedProvider === 'openrouter') updates.openrouter_api_key = apiKeyInput.trim();
                else if (selectedProvider === 'openai') updates.openai_api_key = apiKeyInput.trim();
            }
            await updateSettings(updates);
            setSaveSuccess(true);
            setApiKeyInput('');
            setCurrentConfig({ provider: selectedProvider, model: modelInput, hasKey: true });
            setTimeout(() => setSaveSuccess(false), 3000);
        } catch (e) {
            alert('Failed to save: ' + (e instanceof Error ? e.message : 'Unknown'));
        } finally {
            setSaving(false);
        }
    };

    const handleAddCustomModel = () => {
        const name = prompt('Enter custom model ID (e.g., meta-llama/llama-3-70b):');
        if (name?.trim()) {
            setCustomModels(prev => [...prev, { id: name.trim(), name: name.trim() }]);
            setModelInput(name.trim());
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
                <p className="text-[var(--base-text-muted)]">Configure your Autobot instance — LLM provider, API keys, and preferences.</p>
            </header>

            {/* Current config banner */}
            {currentConfig.model && (
                <div className="glass-panel p-4 rounded-2xl border border-[var(--brand-primary)]/20 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-[var(--brand-primary)]/20 flex items-center justify-center">
                        <Server size={18} className="text-[var(--brand-primary)]" />
                    </div>
                    <div className="flex-1">
                        <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)]">Active Configuration</div>
                        <div className="text-sm font-bold">
                            {currentConfig.provider.toUpperCase()} / {currentConfig.model}
                            {currentConfig.hasKey && <span className="ml-2 text-emerald-400 text-[10px]">Key configured</span>}
                        </div>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* LLM Provider & Model — Full width */}
                <section className="glass-panel p-8 rounded-3xl space-y-6 md:col-span-2">
                    <div className="flex items-center justify-between">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <Sparkles size={20} className="text-[var(--brand-primary)]" /> AI Model Configuration
                        </h3>
                        {saveSuccess && (
                            <div className="flex items-center gap-2 text-emerald-400 text-xs font-bold">
                                <CheckCircle2 size={14} /> Saved
                            </div>
                        )}
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                        {/* Column 1: Provider selection */}
                        <div className="space-y-3">
                            <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest">Provider</label>
                            <div className="space-y-2">
                                {PROVIDERS.map(p => (
                                    <button
                                        key={p.id}
                                        onClick={() => {
                                            setSelectedProvider(p.id);
                                            setModelInput(p.models[0]?.id || '');
                                        }}
                                        className={`w-full p-3 rounded-xl border text-left transition-all ${selectedProvider === p.id
                                            ? 'bg-[var(--brand-primary)]/10 border-[var(--brand-primary)]/50 text-[var(--brand-primary)]'
                                            : 'bg-[var(--base-border)] border-[var(--base-border)] text-[var(--base-text-muted)] hover:border-[var(--base-text-muted)]/30'
                                        }`}
                                    >
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm font-bold">{p.name}</span>
                                            {selectedProvider === p.id && <CheckCircle2 size={14} />}
                                        </div>
                                        <p className="text-[9px] opacity-60 mt-0.5">{p.description}</p>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Column 2: Model selection */}
                        <div className="space-y-3">
                            <div className="flex items-center justify-between">
                                <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest">Model</label>
                                <button
                                    onClick={handleAddCustomModel}
                                    className="text-[9px] font-bold text-[var(--brand-primary)] uppercase tracking-widest hover:underline flex items-center gap-1"
                                >
                                    <PlusCircle size={10} /> Custom
                                </button>
                            </div>
                            <div className="space-y-2 max-h-[300px] overflow-y-auto custom-scrollbar pr-1">
                                {allModels.map(m => (
                                    <button
                                        key={m.id}
                                        onClick={() => setModelInput(m.id)}
                                        className={`w-full p-3 rounded-xl border text-left transition-all ${modelInput === m.id
                                            ? 'bg-[var(--brand-primary)]/10 border-[var(--brand-primary)]/50 text-[var(--brand-primary)]'
                                            : 'bg-[var(--base-border)] border-[var(--base-border)] text-[var(--base-text-muted)] hover:border-[var(--base-text-muted)]/30'
                                        }`}
                                    >
                                        <div className="text-sm font-bold">{m.name}</div>
                                        <div className="text-[9px] opacity-50 font-mono">{m.id}</div>
                                    </button>
                                ))}
                            </div>
                            {/* Custom model ID input */}
                            <div>
                                <label className="text-[9px] font-bold text-[var(--base-text-muted)] uppercase tracking-widest mb-1 block">Or enter model ID directly</label>
                                <input
                                    type="text"
                                    value={modelInput}
                                    onChange={e => setModelInput(e.target.value)}
                                    placeholder="e.g. gemini-2.5-flash-preview-05-20"
                                    className="w-full bg-[var(--base-border)] border border-[var(--base-border)] rounded-xl py-2.5 px-4 text-sm font-mono focus:outline-none focus:border-brand-500/50"
                                />
                            </div>
                        </div>

                        {/* Column 3: API Key + Save */}
                        <div className="space-y-4">
                            <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest">API Key</label>
                            <div className="p-5 rounded-2xl bg-[var(--base-border)] border border-[var(--base-border)] space-y-4">
                                <div className="flex items-center gap-2 text-[var(--brand-primary)]">
                                    <Key size={16} />
                                    <span className="text-xs font-bold">{provider.name} API Key</span>
                                </div>
                                <p className="text-[9px] text-[var(--base-text-muted)] leading-relaxed">
                                    Enter your API key for {provider.name}. It will be saved to the backend .env file.
                                    {provider.id === 'google' && ' Get a free key at ai.google.dev.'}
                                    {provider.id === 'openrouter' && ' Get a key at openrouter.ai.'}
                                </p>
                                <div className="relative">
                                    <input
                                        type={showApiKey ? 'text' : 'password'}
                                        placeholder="Enter API key (leave blank to keep current)..."
                                        value={apiKeyInput}
                                        onChange={e => setApiKeyInput(e.target.value)}
                                        className="w-full bg-[var(--base-bg)] border border-[var(--base-border)] rounded-xl py-2.5 pl-4 pr-10 text-sm font-mono focus:outline-none focus:border-brand-500/50"
                                    />
                                    <button
                                        onClick={() => setShowApiKey(!showApiKey)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--base-text-muted)] hover:text-[var(--base-text)]"
                                    >
                                        {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
                                    </button>
                                </div>
                                <div className="flex items-center gap-2 p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                                    <AlertCircle size={12} className="text-amber-400 shrink-0" />
                                    <span className="text-[9px] text-amber-400">Key is stored in .env on disk — not encrypted. Keep your machine secure.</span>
                                </div>
                            </div>

                            <button
                                onClick={handleSave}
                                disabled={saving || !modelInput.trim()}
                                className="w-full py-3 rounded-xl bg-[var(--brand-primary)] text-white text-xs font-bold uppercase tracking-widest hover:brightness-110 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                            >
                                {saving ? (
                                    <><span className="animate-spin">...</span> Saving</>
                                ) : saveSuccess ? (
                                    <><CheckCircle2 size={14} /> Saved!</>
                                ) : (
                                    <><Save size={14} /> Save Configuration</>
                                )}
                            </button>
                        </div>
                    </div>
                </section>

                {/* Security & Browser Mode */}
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
                    <div>
                        <label className="block text-xs font-bold uppercase tracking-widest text-[var(--base-text-muted)] mb-3">Adapter Policy</label>
                        <div className="grid grid-cols-3 gap-2">
                            {[AdapterPolicy.STRICT, AdapterPolicy.BALANCED, AdapterPolicy.TRUSTED].map(p => (
                                <button
                                    key={p}
                                    onClick={() => {
                                        setPolicy(p);
                                        updateSettings({ approval_mode: p }).catch(() => {});
                                    }}
                                    className={`py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider border transition-all ${policy === p
                                        ? 'bg-[var(--brand-primary)]/20 border-brand-500 text-[var(--brand-primary)]'
                                        : 'bg-[var(--base-border)] border-[var(--base-border)] text-[var(--base-text-muted)] hover:bg-[var(--base-border)]'}`}
                                >{p}</button>
                            ))}
                        </div>
                    </div>
                </section>

                {/* System Utilities + Remote Access */}
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

                    {/* Remote Access / Ngrok */}
                    <div className="space-y-3">
                        <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest flex items-center gap-2">
                            <Globe size={12} /> Remote Monitoring
                        </label>
                        <div className="p-4 rounded-2xl bg-[var(--base-border)] border border-[var(--base-border)] space-y-3">
                            <p className="text-[9px] text-[var(--base-text-muted)] leading-relaxed">
                                Start an ngrok tunnel to monitor Autobot from your phone or any device over the internet.
                            </p>
                            {tunnelActive && tunnelUrl && (
                                <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                                    <div className="text-[9px] font-bold text-emerald-400 uppercase tracking-widest mb-1">Tunnel Active</div>
                                    <a
                                        href={tunnelUrl}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-sm font-mono text-emerald-400 hover:underline break-all"
                                    >
                                        {tunnelUrl}
                                    </a>
                                </div>
                            )}
                            <button
                                onClick={async () => {
                                    setTunnelLoading(true);
                                    try {
                                        if (tunnelActive) {
                                            await stopTunnel();
                                            setTunnelActive(false);
                                            setTunnelUrl(null);
                                        } else {
                                            const res = await startTunnel();
                                            setTunnelActive(true);
                                            setTunnelUrl(res.url);
                                        }
                                    } catch (e) {
                                        alert((e instanceof Error ? e.message : 'Failed'));
                                    } finally {
                                        setTunnelLoading(false);
                                    }
                                }}
                                disabled={tunnelLoading}
                                className={`w-full py-2 rounded-lg text-xs font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${
                                    tunnelActive
                                        ? 'bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20'
                                        : 'bg-[var(--brand-primary)]/10 border border-[var(--brand-primary)]/20 text-[var(--brand-primary)] hover:bg-[var(--brand-primary)]/20'
                                }`}
                            >
                                {tunnelLoading ? '...' : tunnelActive ? 'Stop Tunnel' : 'Start Tunnel'}
                            </button>
                        </div>
                    </div>

                    {/* Appearance */}
                    <div className="space-y-3">
                        <label className="text-xs font-bold text-[var(--base-text-muted)] uppercase tracking-widest">Theme</label>
                        <div className="flex items-center gap-3">
                            {(['light', 'dark'] as Theme[]).map(t => (
                                <button
                                    key={t}
                                    onClick={() => setTheme(t)}
                                    className={`px-4 py-2 rounded-xl border transition-all flex items-center gap-2 ${theme === t
                                        ? 'bg-[var(--brand-primary)]/10 border-[var(--brand-primary)] text-[var(--brand-primary)]'
                                        : 'bg-[var(--base-border)] border-[var(--base-border)] hover:bg-[var(--base-border)] text-[var(--base-text-muted)]'}`}
                                >
                                    <div className={`w-3 h-3 rounded-full border border-black/20 ${t === 'light' ? 'bg-white' : 'bg-[#0a0a0c]'}`} />
                                    <span className="text-xs font-bold uppercase tracking-widest capitalize">{t}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </section>
            </div>
        </motion.div>
    );
}

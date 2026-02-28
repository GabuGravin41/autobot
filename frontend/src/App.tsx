/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom';
import {
  Bot,
  Terminal,
  Play,
  History,
  Settings,
  MessageSquare,
  Zap,
  Cpu,
  Shield,
  Globe,
  MousePointer2,
  Keyboard,
  ChevronRight,
  Search,
  Plus,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  MoreVertical,
  ExternalLink,
  Image as ImageIcon,
  FileText,
  Send,
  ChevronDown,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  AlertCircle,
  Activity,
  Palette,
  Menu,
  X,
  User,
  LogOut,
  PlusCircle,
  Key,
  Database,
  Smartphone,
  LayoutDashboard,
  Sparkles,
  Download,
  Mail,
  Trash2,
  Monitor
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import {
  BrowserMode,
  AdapterPolicy,
  RunHistory,
  TaskStep,
  WorkflowPlan,
  Adapter,
  LLMModel,
  UserProfile
} from './types';
import {
  generatePlan,
  getStatus,
  getAdapters,
  getRuns,
  runPlan,
  cancelRun,
  connectLogStream,
  updateSettings,
  getBrowserScreenshotUrl,
  BackendStatus,
  BackendAdapter,
  BackendRun,
} from './services/apiService';

// --- Mock Data ---
const MOCK_WORKFLOWS: WorkflowPlan[] = [
  {
    id: 'tool_call_stress',
    name: 'Tool Call Stress Test',
    description: 'WhatsApp → Google Docs → Grok → Overleaf chain.',
    steps: [
      { action: 'open_url', args: { url: 'https://web.whatsapp.com' }, description: 'Open WhatsApp Web' },
      { action: 'find_element', args: { selector: 'chat_list' }, description: 'Locate chat list' },
      { action: 'click', args: { selector: 'target_chat' }, description: 'Select target contact' },
      { action: 'type', args: { text: 'Automated message from Autobot' }, description: 'Type message' },
      { action: 'click', args: { selector: 'send_btn' }, description: 'Send message' }
    ]
  },
  {
    id: 'website_builder',
    name: 'Website Builder',
    description: 'Open VS Code, Grok, and Google search for layout ideas.',
    steps: [
      { action: 'open_app', args: { app: 'VS Code' }, description: 'Launch VS Code' },
      { action: 'open_url', args: { url: 'https://grok.com' }, description: 'Open Grok for ideas' },
      { action: 'search', args: { query: 'modern landing page layouts' }, description: 'Search Google for inspiration' }
    ]
  },
  {
    id: 'research_paper',
    name: 'Research Paper Assistant',
    description: 'Open Grok, DeepSeek, Overleaf, and search for references.',
    steps: [
      { action: 'open_url', args: { url: 'https://overleaf.com' }, description: 'Open Overleaf' },
      { action: 'open_url', args: { url: 'https://deepseek.com' }, description: 'Open DeepSeek' },
      { action: 'search', args: { query: 'latest AI research papers 2026' }, description: 'Find references' }
    ]
  }
];

const MOCK_RUNS: RunHistory[] = [
  {
    id: 'run_1',
    planName: 'Tool Call Stress',
    timestamp: '2026-02-24 09:30',
    status: 'success',
    stepsCompleted: 8,
    totalSteps: 8,
    artifacts: { whatsapp_sent: true, pdf_downloaded: true },
    screenshots: ['https://picsum.photos/seed/run1/800/450'],
    logs: ['Starting workflow...', 'Opening WhatsApp...', 'Message sent.', 'Completed.']
  },
  {
    id: 'run_2',
    planName: 'Email Prioritizer',
    timestamp: '2026-02-23 14:15',
    status: 'failed',
    stepsCompleted: 3,
    totalSteps: 10,
    artifacts: {},
    screenshots: ['https://picsum.photos/seed/run2/800/450'],
    logs: ['Starting workflow...', 'Opening Gmail...', 'Timeout waiting for selector.']
  }
];

const MOCK_ADAPTERS: Adapter[] = [
  { name: 'whatsapp_web', description: 'Control WhatsApp Web', actions: ['open_chat', 'send_message'] },
  { name: 'google_docs_web', description: 'Automate Google Docs', actions: ['type_text', 'open_new_doc'] },
  { name: 'grok_web', description: 'Interface with xAI Grok', actions: ['ask_from_clipboard', 'copy_response'] },
  { name: 'overleaf_web', description: 'Manage LaTeX projects', actions: ['compile', 'download_pdf'] },
];

// --- Components ---

const NavItem = ({ icon: Icon, label, active, onClick }: { icon: any, label: string, active: boolean, onClick: () => void }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${active
      ? 'bg-brand-500/10 text-brand-400 border border-brand-500/20 shadow-[0_0_20px_rgba(34,197,94,0.1)]'
      : 'text-white/50 hover:text-white hover:bg-white/5'
      }`}
  >
    <Icon size={20} className={active ? 'animate-pulse' : ''} />
    <span className="font-medium">{label}</span>
    {active && <motion.div layoutId="active-pill" className="ml-auto w-1.5 h-1.5 rounded-full bg-brand-400 shadow-[0_0_10px_rgba(34,197,94,0.8)]" />}
  </button>
);

const StatCard = ({ label, value, icon: Icon, trend }: { label: string, value: string | number, icon: any, trend?: string }) => (
  <div className="glass-card p-6 rounded-2xl">
    <div className="flex justify-between items-start mb-4">
      <div className="p-2 rounded-lg bg-white/5 text-white/70">
        <Icon size={20} />
      </div>
      {trend && <span className="text-xs font-medium text-brand-400">{trend}</span>}
    </div>
    <div className="text-2xl font-bold mb-1">{value}</div>
    <div className="text-sm text-white/40">{label}</div>
  </div>
);

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const activeTab = location.pathname.substring(1) || 'dashboard';

  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [liveAdapters, setLiveAdapters] = useState<BackendAdapter[]>([]);
  const [liveRuns, setLiveRuns] = useState<BackendRun[]>([]);
  const [liveLogLines, setLiveLogLines] = useState<string[]>([]);
  const [backendOnline, setBackendOnline] = useState(false);
  const [screenshotUrl, setScreenshotUrl] = useState<string>('');
  const [browserMode, setBrowserMode] = useState<BrowserMode>(BrowserMode.HUMAN_PROFILE);
  const [policy, setPolicy] = useState<AdapterPolicy>(AdapterPolicy.BALANCED);
  const [isAutonomous, setIsAutonomous] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [theme, setTheme] = useState<'blue-violet' | 'emerald' | 'blue' | 'amber'>('blue-violet');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const [apiKey, setApiKey] = useState('');
  const [isApiKeySaved, setIsApiKeySaved] = useState(false);
  const [models, setModels] = useState<LLMModel[]>([
    { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro', provider: 'Google' },
    { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', provider: 'Google' },
    { id: 'deepseek/deepseek-chat', name: 'DeepSeek V3', provider: 'OpenRouter' },
    { id: 'deepseek/deepseek-r1', name: 'DeepSeek R1', provider: 'OpenRouter' },
    { id: 'gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
  ]);
  const [selectedModelId, setSelectedModelId] = useState('deepseek/deepseek-chat');


  const [user] = useState<UserProfile>({
    name: 'Alex Rivera',
    email: 'alex@autobot.ai',
    avatar: 'https://picsum.photos/seed/alex/100/100',
    role: 'System Architect'
  });


  const [chatMessages, setChatMessages] = useState<{ role: 'user' | 'bot', content: string, plan?: WorkflowPlan, artifact?: any }[]>([
    { role: 'bot', content: 'Hello! I am Autobot. How can I help you automate your computer today?' },
    {
      role: 'bot',
      content: 'I have analyzed your recent emails. Here is the most urgent one that requires your attention:',
      artifact: {
        type: 'email',
        subject: 'URGENT: Server Migration Schedule',
        from: 'DevOps Team',
        priority: 'High',
        summary: 'The production server migration is scheduled for tonight at 22:00 UTC. Please confirm your availability for the smoke tests.'
      }
    }
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeRun, setActiveRun] = useState<RunHistory | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [showReasoning, setShowReasoning] = useState<Record<number, boolean>>({});
  const [adapterSearch, setAdapterSearch] = useState('');
  const [showAuditLog, setShowAuditLog] = useState(false);
  const [selectedRun, setSelectedRun] = useState<RunHistory | null>(null);
  const [selectedArtifact, setSelectedArtifact] = useState<any | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Derived from live data to keep components working
  const runs = liveRuns;
  const artifacts = liveAdapters.map(a => ({
    id: a.name,
    title: a.description,
    type: 'adapter',
    timestamp: 'online'
  }));

  useEffect(() => {
    let disconnectLogs: (() => void) | null = null;

    const poll = async () => {
      try {
        const [status, adaptersData, runsData] = await Promise.all([
          getStatus(),
          getAdapters(),
          getRuns(),
        ]);
        setBackendStatus(status);
        setLiveAdapters(adaptersData.adapters);
        setLiveRuns(runsData.runs);
        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    };

    poll();
    const interval = setInterval(poll, 5000);

    // Real-time log streaming via WebSocket
    disconnectLogs = connectLogStream((line) => {
      setLiveLogLines(prev => [...prev.slice(-200), line]);
    });

    return () => {
      clearInterval(interval);
      disconnectLogs?.();
    };
  }, []);

  // Sync backend run state → activeRun card
  useEffect(() => {
    if (!backendStatus) return;
    if (backendStatus.run_status === 'running' && backendStatus.active_run_id) {
      setActiveRun(prev => prev ? { ...prev, logs: liveLogLines } : {
        id: backendStatus.active_run_id!,
        planName: 'Active Run',
        timestamp: new Date().toLocaleString(),
        status: 'running',
        stepsCompleted: liveLogLines.length,
        totalSteps: 0,
        artifacts: {},
        screenshots: [],
        logs: liveLogLines,
      });
    } else if (backendStatus.run_status === 'done' || backendStatus.run_status === 'failed') {
      setActiveRun(prev => prev ? {
        ...prev,
        status: backendStatus.run_status === 'done' ? 'success' : 'failed',
        logs: liveLogLines,
      } : null);
    }
  }, [backendStatus?.run_status, liveLogLines]);


  useEffect(() => {
    if (!backendOnline || !backendStatus?.browser?.active) {
      setScreenshotUrl('');
      return;
    }
    const updateScreenshot = () => {
      setScreenshotUrl(getBrowserScreenshotUrl());
    };
    updateScreenshot();
    const interval = setInterval(updateScreenshot, 3000);
    return () => clearInterval(interval);
  }, [backendOnline, backendStatus?.browser?.active]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const handleSaveApiKey = async () => {
    if (!apiKey.trim()) return;
    try {
      const { updateSettings: apiUpdateSettings } = await import('./services/apiService');
      const provider = selectedModelId.includes('gemini') ? 'gemini' : selectedModelId.includes('deepseek') ? 'openrouter' : 'openai';

      const updates: any = {
        llm_provider: provider,
        llm_model: selectedModelId,
      };

      if (provider === 'openrouter') updates.openrouter_api_key = apiKey;
      if (provider === 'openai') updates.openai_api_key = apiKey;

      await apiUpdateSettings(updates);
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
      const newModel: LLMModel = {
        id: name.toLowerCase().replace(/\s+/g, '-'),
        name,
        provider: 'Custom',
        isCustom: true
      };
      setModels([...models, newModel]);
    }
  };

  const handleAddWorkflow = () => {
    const name = prompt('Enter workflow name:');
    if (name) {
      const newWorkflow: WorkflowPlan = {
        id: `wf_${Date.now()}`,
        name,
        description: 'Custom user-defined workflow.',
        steps: [{ action: 'manual', args: {}, description: 'Initial custom step' }]
      };
      // In a real app, we'd update state. For now, we'll just simulate it.
      alert(`Workflow "${name}" created and added to library.`);
    }
  };

  // ── Chat: send user message → LLM planner → plan card ──────────────────
  const handleSendMessage = async () => {
    if (!chatInput.trim() || isGenerating) return;

    const userMsg = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsGenerating(true);

    try {
      // sendChat calls /api/chat → LLMBrain.generate_plan_draft or text planner
      const { sendChat } = await import('./services/apiService');
      const { reply, plan } = await sendChat(userMsg, {
        browser_mode: backendStatus?.browser?.mode,
        llm_enabled: backendStatus?.llm_enabled,
      });
      setChatMessages(prev => [...prev, {
        role: 'bot',
        content: reply,
        plan: plan ? {
          id: plan.id,
          name: plan.name,
          description: plan.description,
          steps: plan.steps,
        } : undefined,
      }]);
    } catch (error) {
      setChatMessages(prev => [...prev, {
        role: 'bot',
        content: `Sorry, I couldn't reach the Autobot backend. Make sure the server is running on port 8000. Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      }]);
    } finally {
      setIsGenerating(false);
    }
  };

  // ── Execute plan: call real backend, switch to dashboard for live view ────
  const executePlan = async (plan: WorkflowPlan) => {
    navigate('/dashboard');
    setIsPaused(false);

    // Optimistic UI: show a "starting" run card immediately
    const optimisticRun: RunHistory = {
      id: 'pending',
      planName: plan.name,
      timestamp: new Date().toLocaleString(),
      status: 'running',
      stepsCompleted: 0,
      totalSteps: plan.steps?.length || 0,
      artifacts: {},
      screenshots: [],
      logs: [`▶ Sending plan '${plan.name}' to engine...`],
    };
    setActiveRun(optimisticRun);

    try {
      const { runPlan: apiRunPlan } = await import('./services/apiService');
      const { run_id } = await apiRunPlan(plan as any);
      // Update with real run_id; backend WebSocket + polling will keep logs live
      setActiveRun(prev => prev ? { ...prev, id: run_id } : null);
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : 'Unknown error';
      setActiveRun(prev => prev ? {
        ...prev,
        status: 'failed',
        logs: [...(prev.logs || []), `✗ Failed to start plan: ${errMsg}`],
      } : null);
    }
  };

  // ── Abort the current run ────────────────────────────────────────────────
  const handleAbortRun = async () => {
    if (!activeRun || activeRun.status !== 'running') {
      setActiveRun(null);
      return;
    }
    try {
      if (activeRun.id && activeRun.id !== 'pending') {
        const { cancelRun: apiCancelRun } = await import('./services/apiService');
        await apiCancelRun(activeRun.id);
      }
    } catch (e) {
      // Ignore cancel errors — status will self-correct via polling
    } finally {
      setActiveRun(prev => prev ? { ...prev, status: 'failed', logs: [...(prev.logs || []), '⚠ Cancelled by user.'] } : null);
      setIsPaused(false);
    }
  };

  const handleDeleteRun = async (runId: string) => {
    if (!confirm('Are you sure you want to delete this run and all its history?')) return;
    try {
      const { deleteRun: apiDeleteRun } = await import('./services/apiService');
      await apiDeleteRun(runId);
      setLiveRuns(prev => prev.filter(r => r.id !== runId));
    } catch (e) {
      alert('Failed to delete run: ' + (e instanceof Error ? e.message : 'Unknown error'));
    }
  };

  const handleViewDetails = async (run: RunHistory) => {
    try {
      const { getRun: apiGetRun } = await import('./services/apiService');
      const details = await apiGetRun(run.id);
      setSelectedRun({ ...run, logs: details.logs || [] });
    } catch (e) {
      setSelectedRun(run); // fallback to basic data
    }
  };


  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-obsidian-bg flex items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-brand-500/10 blur-[120px] rounded-full -z-10 animate-pulse" />
        <div className="absolute bottom-0 left-0 w-[300px] h-[300px] bg-emerald-500/10 blur-[100px] rounded-full -z-10 animate-pulse" />

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-md glass-panel p-10 rounded-3xl space-y-8 relative z-10"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-brand-500 flex items-center justify-center shadow-[0_0_30px_rgba(var(--brand-500-rgb),0.5)]">
              <Bot className="text-black" size={32} />
            </div>
            <div className="text-center">
              <h1 className="text-3xl font-bold tracking-tight">AUTOBOT</h1>
              <p className="text-xs uppercase tracking-[0.3em] text-brand-400 font-bold">Secure Access</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-[10px] font-bold uppercase tracking-widest text-white/40 ml-1">System Identifier</label>
              <input type="text" placeholder="Username or Email" className="input-field" defaultValue="alex@autobot.ai" />
            </div>
            <div className="space-y-2">
              <label className="text-[10px] font-bold uppercase tracking-widest text-white/40 ml-1">Access Token</label>
              <input type="password" placeholder="••••••••" className="input-field" defaultValue="password" />
            </div>
          </div>

          <div className="space-y-4">
            <button
              onClick={() => setIsAuthenticated(true)}
              className="btn-primary w-full py-4 text-sm uppercase tracking-widest"
            >
              Initialize Session
            </button>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-white/5"></div>
              </div>
              <div className="relative flex justify-center text-[10px] uppercase tracking-widest">
                <span className="bg-obsidian-panel px-4 text-white/20">Or continue with</span>
              </div>
            </div>

            <button
              onClick={() => setIsAuthenticated(true)}
              className="w-full py-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-3 text-sm font-medium"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" />
                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
              Google Account
            </button>
          </div>

          <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-white/20">
            <button className="hover:text-brand-400 transition-colors">Forgot Token?</button>
            <button className="hover:text-brand-400 transition-colors">Request Access</button>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col md:flex-row font-sans overflow-hidden bg-obsidian-bg">
      {/* Mobile Header */}
      <div className="md:hidden h-16 glass-panel border-b border-white/5 flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-2">
          <Bot className="text-brand-500" size={24} />
          <span className="font-bold tracking-tight">AUTOBOT</span>
        </div>
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="p-2 rounded-lg bg-white/5 text-white/60"
        >
          {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Sidebar */}
      <aside className={`
        fixed inset-0 z-40 md:relative md:translate-x-0 transition-transform duration-300 ease-in-out
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        w-72 glass-panel border-r border-white/5 flex flex-col p-6
      `}>
        <div className="hidden md:flex items-center gap-3 mb-10 px-2">
          <motion.div
            animate={{
              rotate: [0, 10, -10, 0],
              scale: [1, 1.1, 1]
            }}
            transition={{ duration: 4, repeat: Infinity }}
            className="w-10 h-10 rounded-xl bg-brand-500 flex items-center justify-center shadow-[0_0_20px_rgba(var(--brand-500-rgb),0.4)]"
          >
            <Bot className="text-black" size={24} />
          </motion.div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">AUTOBOT</h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-brand-400 font-bold">Control Center</p>
          </div>
        </div>

        <nav className="flex-1 space-y-2">
          <NavItem icon={LayoutDashboard} label="Dashboard" active={activeTab === 'dashboard'} onClick={() => { navigate('/dashboard'); setIsSidebarOpen(false); }} />
          <NavItem icon={Sparkles} label="AI Planner" active={activeTab === 'planner'} onClick={() => { navigate('/planner'); setIsSidebarOpen(false); }} />
          <NavItem icon={Zap} label="Workflows" active={activeTab === 'workflows'} onClick={() => { navigate('/workflows'); setIsSidebarOpen(false); }} />
          <NavItem icon={History} label="Run History" active={activeTab === 'history'} onClick={() => { navigate('/history'); setIsSidebarOpen(false); }} />
          <NavItem icon={Settings} label="Settings" active={activeTab === 'settings'} onClick={() => { navigate('/settings'); setIsSidebarOpen(false); }} />
        </nav>

        <div className="mt-auto pt-6 border-t border-white/5 space-y-6">
          <div className="flex items-center justify-between px-2">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isAutonomous ? 'bg-brand-500 animate-pulse' : 'bg-white/20'}`} />
              <span className="text-xs font-bold text-white/60">Autonomous Mode</span>
            </div>
            <button
              onClick={() => setIsAutonomous(!isAutonomous)}
              className={`w-10 h-5 rounded-full transition-colors relative ${isAutonomous ? 'bg-brand-500' : 'bg-white/10'}`}
            >
              <motion.div
                animate={{ x: isAutonomous ? 22 : 2 }}
                className="absolute top-1 left-1 w-3 h-3 rounded-full bg-white shadow-sm"
              />
            </button>
          </div>

          {/* User Profile */}
          <div
            onClick={() => { navigate('/profile'); setIsSidebarOpen(false); }}
            className={`glass-card p-3 rounded-2xl flex items-center gap-3 group cursor-pointer transition-all ${activeTab === 'profile' ? 'bg-brand-500/10 border-brand-500/30' : ''}`}
          >
            <div className="relative">
              <img src={user.avatar} alt={user.name} className="w-10 h-10 rounded-xl object-cover border border-white/10" />
              <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 border-2 border-obsidian-panel flex items-center justify-center">
                <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-bold truncate">{user.name}</div>
              <div className="text-[10px] text-white/40 truncate">{user.role}</div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setIsAuthenticated(false); }}
              className="p-1.5 rounded-lg hover:bg-red-500/10 text-white/20 hover:text-red-400 transition-colors"
            >
              <LogOut size={14} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-[calc(100vh-64px)] md:h-screen overflow-y-auto bg-obsidian-bg relative custom-scrollbar">
        {/* Background Gradients */}
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-brand-500/5 blur-[120px] rounded-full -z-10" />
        <div className="absolute bottom-0 left-0 w-[300px] h-[300px] bg-emerald-500/5 blur-[100px] rounded-full -z-10" />

        <div className="max-w-7xl mx-auto p-6 md:p-10 pb-24">
          <AnimatePresence mode="wait">
            <Routes location={location}>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={
                <motion.div
                  key="dashboard"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-8"
                >
                  {activeRun && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="glass-panel p-8 rounded-3xl glow-border relative overflow-hidden group mb-8"
                    >
                      <div className="absolute inset-0 hologram-grid opacity-10 pointer-events-none" />
                      <div className="absolute -inset-2 bg-brand-500/5 blur-3xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />

                      <div className="relative z-10">
                        <div className="flex justify-between items-start mb-8">
                          <div>
                            <div className="flex items-center gap-3 mb-2">
                              <div className="px-2 py-0.5 rounded bg-brand-500/20 text-brand-400 text-[10px] font-bold uppercase tracking-wider flex items-center gap-2">
                                <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
                                Active Operation
                              </div>
                              <h3 className="text-2xl font-bold tracking-tight">{activeRun.planName}</h3>
                            </div>
                            <p className="text-white/40 text-sm">Initialized at {activeRun.timestamp} • Session ID: {activeRun.id}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-3xl font-mono font-bold text-brand-400">
                              {Math.round((activeRun.stepsCompleted / activeRun.totalSteps) * 100)}%
                            </div>
                            <div className="text-[10px] text-white/40 uppercase tracking-widest font-bold">Execution Progress</div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                          <div className="lg:col-span-2 space-y-6">
                            <div className="space-y-3">
                              <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-white/40">
                                <span>Step {activeRun.stepsCompleted} of {activeRun.totalSteps}</span>
                                <span className="text-brand-400">Processing...</span>
                              </div>
                              <div className="h-3 w-full bg-white/5 rounded-full overflow-hidden p-0.5 border border-white/10">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${(activeRun.stepsCompleted / activeRun.totalSteps) * 100}%` }}
                                  className="h-full bg-gradient-to-r from-brand-500 to-accent-cyan rounded-full shadow-[0_0_15px_rgba(var(--brand-500-rgb),0.5)]"
                                />
                              </div>
                            </div>

                            <div className="glass-panel bg-black/40 rounded-2xl p-6 h-[280px] flex flex-col">
                              <div className="flex items-center justify-between mb-4 pb-4 border-b border-white/5">
                                <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 flex items-center gap-2">
                                  <Terminal size={14} />
                                  Execution Logs
                                </div>
                                <div className="flex gap-1">
                                  <div className="w-2 h-2 rounded-full bg-red-500/20" />
                                  <div className="w-2 h-2 rounded-full bg-amber-500/20" />
                                  <div className="w-2 h-2 rounded-full bg-emerald-500/20" />
                                </div>
                              </div>
                              <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-2 font-mono text-[11px]">
                                {activeRun.logs.map((log, i) => (
                                  <motion.div
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    key={i}
                                    className="flex gap-3 group"
                                  >
                                    <span className="text-white/20">[{i.toString().padStart(2, '0')}]</span>
                                    <span className={i === activeRun.logs.length - 1 ? 'text-brand-400' : 'text-white/60'}>
                                      {log}
                                    </span>
                                  </motion.div>
                                ))}
                                <div className="animate-pulse text-brand-400/50">_</div>
                              </div>
                            </div>
                          </div>

                          <div className="space-y-6">
                            <div className="aspect-video rounded-2xl bg-black/60 border border-white/10 overflow-hidden relative group/view shadow-2xl">
                              <img
                                src={`https://picsum.photos/seed/${activeRun.id}/800/450`}
                                className="w-full h-full object-cover opacity-40 group-hover/view:opacity-60 transition-opacity duration-700"
                                alt="Live View"
                              />
                              <div className="scanline" />
                              <div className="absolute inset-0 hologram-grid opacity-20" />
                              <div className="absolute inset-0 flex items-center justify-center">
                                <div className="flex flex-col items-center gap-4">
                                  <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-black/60 backdrop-blur-xl border border-white/10 text-[10px] font-bold uppercase tracking-widest text-white shadow-2xl">
                                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                    Live Remote View
                                  </div>
                                  <div className="text-[10px] font-mono text-white/20 bg-black/40 px-2 py-1 rounded">
                                    9222:WS_ATTACHED
                                  </div>
                                </div>
                              </div>
                              <div className="absolute bottom-4 right-4 flex gap-2">
                                <button className="p-2 rounded-lg bg-black/60 border border-white/10 text-white/40 hover:text-white transition-colors">
                                  <Search size={14} />
                                </button>
                              </div>
                            </div>

                            <div className="p-4 rounded-2xl bg-brand-500/5 border border-brand-500/10 mb-4">
                              <div className="text-[9px] font-bold uppercase tracking-widest text-brand-400 mb-2">Latest Log</div>
                              <div className="text-xs text-white/60 leading-relaxed font-mono truncate">
                                {activeRun.logs.length > 0 ? activeRun.logs[activeRun.logs.length - 1] : 'Waiting for engine output...'}
                              </div>
                            </div>

                            {backendStatus?.browser?.active && screenshotUrl && (
                              <div className="space-y-4">
                                <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 flex items-center gap-2">
                                  <Monitor size={12} />
                                  Live Browser Feed
                                </div>
                                <div className="relative aspect-video rounded-2xl overflow-hidden border border-white/10 group/feed">
                                  <img
                                    src={screenshotUrl}
                                    alt="Live Feed"
                                    className="w-full h-full object-cover"
                                    onError={(e) => (e.currentTarget.style.display = 'none')}
                                  />
                                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover/feed:opacity-100 transition-opacity flex items-end p-4">
                                    <div className="text-[10px] font-mono text-white/80 truncate">
                                      {backendStatus.browser.url}
                                    </div>
                                  </div>
                                  <div className="absolute top-3 right-3 flex items-center gap-2 px-2 py-1 rounded bg-black/60 backdrop-blur-md border border-white/10">
                                    <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                                    <span className="text-[8px] font-bold text-white tracking-widest uppercase">Live</span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                      <h2 className="text-4xl font-bold tracking-tight mb-2">Command Center</h2>
                      <p className="text-white/40">Real-time overview of your autonomous operations.</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest ${backendOnline ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
                        {backendOnline ? 'Engine Online' : 'Engine Offline'}
                      </div>
                      <button
                        onClick={() => {
                          const logText = activeRun?.logs?.join('\n') || liveLogLines.join('\n');
                          const blob = new Blob([logText], { type: 'text/plain' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a'); a.href = url; a.download = `autobot-logs.txt`; a.click();
                          URL.revokeObjectURL(url);
                        }}
                        className="btn-secondary py-2 px-4 text-[10px] uppercase tracking-widest flex items-center gap-2"
                      >
                        <Download size={14} />
                        Export Logs
                      </button>
                      <button
                        onClick={() => navigate('/planner')}
                        className="btn-primary py-2 px-6 text-[10px] uppercase tracking-widest flex items-center gap-2 shadow-[0_0_20px_rgba(var(--brand-500-rgb),0.3)]"
                      >
                        <PlusCircle size={14} />
                        New Workflow
                      </button>
                    </div>
                  </header>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
                    <div className="glass-panel p-6 rounded-3xl border-brand-500/20 relative overflow-hidden group">
                      <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Activity size={48} />
                      </div>
                      <div className="text-[10px] font-bold text-brand-400 uppercase tracking-[0.2em] mb-1">Active Runs</div>
                      <div className="text-4xl font-bold tracking-tighter">{liveRuns.filter(r => r.status === 'running').length + (activeRun?.status === 'running' ? 1 : 0)}</div>
                      <div className="mt-4 flex items-center gap-2 text-[10px] text-emerald-400 font-bold">
                        <Zap size={10} />
                        +2.4% from last hour
                      </div>
                    </div>
                    <div className="glass-panel p-6 rounded-3xl border-white/5">
                      <div className="text-[10px] font-bold text-white/40 uppercase tracking-[0.2em] mb-1">Total Runs</div>
                      <div className="text-4xl font-bold tracking-tighter">{liveRuns.length}</div>
                      <div className="mt-4 flex items-center gap-2 text-[10px] text-white/20">
                        Stable performance
                      </div>
                    </div>
                    <div className="glass-panel p-6 rounded-3xl border-white/5">
                      <div className="text-[10px] font-bold text-white/40 uppercase tracking-[0.2em] mb-1">Adapters</div>
                      <div className="text-4xl font-bold tracking-tighter">{liveAdapters.length}</div>
                      <div className="mt-4 flex items-center gap-2 text-[10px] text-amber-400 font-bold">
                        84% of quota
                      </div>
                    </div>
                    <div className="glass-panel p-6 rounded-3xl border-white/5">
                      <div className="text-[10px] font-bold text-white/40 uppercase tracking-[0.2em] mb-1">Backend Uptime</div>
                      <div className="text-4xl font-bold tracking-tighter">{backendOnline ? '100%' : '0%'}</div>
                      <div className="mt-4 flex items-center gap-2 text-[10px] text-brand-400 font-bold">
                        This week
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    <div className="lg:col-span-2 space-y-6">
                      <div className="flex items-center justify-between px-2">
                        <h3 className="text-xl font-bold tracking-tight">Active Operations</h3>
                        <button
                          onClick={() => navigate('/history')}
                          className="text-[10px] font-bold uppercase tracking-widest text-brand-400 hover:underline"
                        >
                          View All
                        </button>
                      </div>
                      <div className="space-y-4">
                        {runs.map(run => (
                          <div
                            key={run.id}
                            onClick={() => {
                              setSelectedRun(run as any);
                            }}
                            className="glass-panel p-6 rounded-3xl border-white/5 hover:border-brand-500/30 transition-all group cursor-pointer"
                          >
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                              <div className="flex items-center gap-4">
                                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${run.status === 'running' ? 'bg-brand-500/10 text-brand-400 animate-pulse' :
                                  run.status === 'success' ? 'bg-emerald-500/10 text-emerald-400' :
                                    'bg-red-500/10 text-red-400'
                                  }`}>
                                  {run.status === 'running' ? <Play size={20} /> :
                                    run.status === 'success' ? <CheckCircle2 size={20} /> :
                                      <AlertCircle size={20} />}
                                </div>
                                <div>
                                  <div className="font-bold group-hover:text-brand-400 transition-colors">{run.planName}</div>
                                  <div className="text-[10px] text-white/40 uppercase tracking-widest flex items-center gap-2 mt-1">
                                    <span>{run.id}</span>
                                    <span className="w-1 h-1 rounded-full bg-white/20" />
                                    <span>{run.timestamp}</span>
                                  </div>
                                </div>
                              </div>
                              <div className="flex items-center gap-6">
                                <div className="text-right hidden xs:block">
                                  <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 mb-1">Progress</div>
                                  <div className="flex items-center gap-3">
                                    <div className="w-24 h-1.5 bg-white/5 rounded-full overflow-hidden">
                                      <div
                                        className="h-full bg-brand-500 transition-all duration-500"
                                        style={{ width: `${run.progress}%` }}
                                      />
                                    </div>
                                    <span className="text-xs font-mono">{run.progress}%</span>
                                  </div>
                                </div>
                                <ChevronRight size={16} className="text-white/20 group-hover:text-brand-400 transition-colors" />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-6">
                      <h3 className="text-xl font-bold tracking-tight px-2">Recent Artifacts</h3>
                      <div className="glass-panel p-6 rounded-3xl border-white/5 space-y-6">
                        {artifacts.map((artifact, idx) => (
                          <div key={artifact.id} className={`flex items-start gap-4 ${idx !== artifacts.length - 1 ? 'pb-6 border-bottom border-white/5' : ''}`}>
                            <div className="p-2 rounded-lg bg-white/5 text-white/40">
                              {artifact.type === 'email' ? <Mail size={16} /> : <FileText size={16} />}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-bold truncate">{artifact.title}</div>
                              <div className="text-[10px] text-white/40 uppercase tracking-widest mt-0.5">{artifact.timestamp}</div>
                              <div className="mt-3 flex items-center gap-2">
                                <button
                                  onClick={() => setSelectedArtifact(artifact)}
                                  className="px-2 py-1 rounded bg-brand-500/10 text-brand-400 text-[9px] font-bold uppercase tracking-widest hover:bg-brand-500/20 transition-colors"
                                >
                                  View
                                </button>
                                <button className="px-2 py-1 rounded bg-white/5 text-white/40 text-[9px] font-bold uppercase tracking-widest hover:bg-white/10 transition-colors">
                                  Dismiss
                                </button>
                              </div>
                            </div>
                          </div>
                        ))}
                        <button className="w-full py-3 rounded-2xl bg-white/5 hover:bg-white/10 text-[10px] font-bold uppercase tracking-widest transition-all">
                          Open Artifact Vault
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              } />

              <Route path="/planner" element={
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
                    <div className="flex-1 overflow-y-auto p-6 space-y-6">
                      {chatMessages.map((msg, i) => (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          key={i}
                          className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div className={`max-w-[80%] p-4 rounded-2xl ${msg.role === 'user'
                            ? 'bg-brand-500 text-black font-medium'
                            : 'bg-white/5 border border-white/10'
                            }`}>
                            <p className="text-sm leading-relaxed">{msg.content}</p>

                            {msg.artifact && msg.artifact.type === 'email' && (
                              <div className="mt-4 p-4 rounded-xl bg-white/5 border border-white/10 space-y-3">
                                <div className="flex justify-between items-start">
                                  <div className="flex items-center gap-2">
                                    <div className="p-1.5 rounded-lg bg-red-500/10 text-red-400">
                                      <AlertCircle size={14} />
                                    </div>
                                    <span className="text-[10px] font-bold uppercase tracking-widest text-red-400">Urgent Email</span>
                                  </div>
                                  <span className="text-[10px] text-white/20 font-mono">ID: EM-992</span>
                                </div>
                                <div>
                                  <h4 className="text-sm font-bold text-white/90">{msg.artifact.subject}</h4>
                                  <p className="text-[10px] text-white/40">From: {msg.artifact.from}</p>
                                </div>
                                <p className="text-xs text-white/60 leading-relaxed italic">"{msg.artifact.summary}"</p>
                                <div className="flex gap-2 pt-2">
                                  <button
                                    onClick={() => alert('Drafting response in Gmail...')}
                                    className="flex-1 py-1.5 rounded-lg bg-brand-500 text-black text-[10px] font-bold hover:bg-brand-400 transition-colors"
                                  >
                                    Draft Response
                                  </button>
                                  <button
                                    onClick={() => alert('Opening Gmail in new tab...')}
                                    className="flex-1 py-1.5 rounded-lg bg-white/5 border border-white/10 text-[10px] font-bold hover:bg-white/10 transition-colors"
                                  >
                                    Open in Gmail
                                  </button>
                                </div>
                              </div>
                            )}

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
                                        {" Analyzing user intent... Mapping to available adapters... Validating security policy... Constructing multi-step workflow plan."}
                                        <span className="text-accent-amber">{"</think>"}</span>
                                      </div>
                                    </motion.div>
                                  )}
                                </AnimatePresence>
                              </div>
                            )}

                            {msg.plan && (
                              <div className="mt-4 p-4 rounded-xl bg-black/40 border border-white/10 space-y-4">
                                <div className="flex items-center justify-between">
                                  <h4 className="text-xs font-bold uppercase tracking-widest text-brand-400">Proposed Plan</h4>
                                  <span className="text-[10px] text-white/40">{(msg.plan.steps?.length || 0)} steps</span>
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
                                  onClick={() => executePlan(msg.plan!)}
                                  className="w-full py-2 rounded-lg bg-brand-500 text-black text-xs font-bold hover:bg-brand-400 transition-colors flex items-center justify-center gap-2"
                                >
                                  <Play size={14} />
                                  Execute Plan
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

                    <div className="p-6 bg-white/[0.02] border-t border-white/5">
                      <div className="relative">
                        <input
                          type="text"
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                          placeholder="e.g. Find the most urgent email and send me a summary on WhatsApp"
                          className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-6 pr-16 text-sm focus:outline-none focus:border-brand-500/50 transition-colors"
                        />
                        <button
                          onClick={handleSendMessage}
                          disabled={isGenerating || !chatInput.trim()}
                          className="absolute right-3 top-1/2 -translate-y-1/2 w-10 h-10 rounded-xl bg-brand-500 flex items-center justify-center text-black hover:bg-brand-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <Send size={18} />
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              } />

              <Route path="/workflows" element={
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
                      <p className="text-white/40">Launch pre-configured automation sequences.</p>
                    </div>
                    <div className="flex items-center gap-2 bg-white/5 p-1 rounded-xl border border-white/10">
                      <button className="px-4 py-1.5 rounded-lg bg-white/10 text-[10px] font-bold uppercase tracking-widest">All</button>
                      <button className="px-4 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest text-white/40 hover:text-white transition-colors">Social</button>
                      <button className="px-4 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest text-white/40 hover:text-white transition-colors">Productivity</button>
                      <button className="px-4 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest text-white/40 hover:text-white transition-colors">DevOps</button>
                    </div>
                  </header>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {MOCK_WORKFLOWS.map((workflow) => (
                      <div key={workflow.id} className="glass-panel p-8 rounded-3xl border-white/5 hover:border-brand-500/30 transition-all group flex flex-col h-full">
                        <div className="flex items-start justify-between mb-6">
                          <div className="w-14 h-14 rounded-2xl bg-brand-500/10 flex items-center justify-center group-hover:bg-brand-500/20 transition-colors">
                            <Zap size={28} className="text-brand-400" />
                          </div>
                          <div className="flex gap-1">
                            <span className="px-2 py-0.5 rounded bg-white/5 text-[9px] font-bold uppercase tracking-widest text-white/40">v2.1</span>
                          </div>
                        </div>

                        <h3 className="text-xl font-bold mb-3 group-hover:text-brand-400 transition-colors">{workflow.name}</h3>
                        <p className="text-sm text-white/40 mb-8 leading-relaxed flex-1">{workflow.description}</p>

                        <div className="space-y-6">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                              <div className="text-[9px] font-bold uppercase tracking-widest text-white/20">Est. Time</div>
                              <div className="text-xs font-mono">~4.5 min</div>
                              <Terminal size={14} />
                            </div>
                          </div>
                          <button
                            onClick={() => executePlan({ ...workflow, steps: workflow.steps || [] })}
                            className="w-full py-4 rounded-xl bg-brand-500 text-black font-bold text-xs uppercase tracking-widest hover:bg-brand-400 transition-all flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(var(--brand-500-rgb),0.2)] active:scale-[0.98]"
                          >
                            <Play size={16} fill="currentColor" />
                            Initialize Run
                          </button>
                        </div>
                      </div>
                    ))}

                    <button
                      onClick={handleAddWorkflow}
                      className="glass-panel p-8 rounded-3xl border-dashed border-white/10 hover:border-brand-500/30 hover:bg-brand-500/5 transition-all flex flex-col items-center justify-center gap-4 group min-h-[400px]"
                    >
                      <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-brand-500/10 transition-colors">
                        <Plus size={32} className="text-white/20 group-hover:text-brand-400" />
                      </div>
                      <div className="text-center">
                        <div className="font-bold text-lg">Custom Workflow</div>
                        <div className="text-xs text-white/40 mt-1">Build your own sequence</div>
                      </div>
                    </button>
                  </div>
                </motion.div>
              } />

              <Route path="/history" element={
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
                              <div className={`flex items-center gap-2 ${run.status === 'success' ? 'text-emerald-400' : 'text-red-400'}`}>
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
                                  <div className="h-full bg-brand-500" style={{ width: `${run.progress}%` }} />
                                </div>
                                <span className="text-[10px] text-white/40">{run.progress}%</span>
                              </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                              <div className="flex items-center justify-end gap-2">
                                <button
                                  onClick={() => handleViewDetails(run)}
                                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-brand-400 transition-colors"
                                  title="View Details"
                                >
                                  <Eye size={16} />
                                </button>
                                <button
                                  onClick={() => handleDeleteRun(run.id)}
                                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-red-400 transition-colors"
                                  title="Delete Record"
                                >
                                  <Trash2 size={16} />
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              } />

              <Route path="/settings" element={
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
                    <section className="glass-panel p-8 rounded-3xl space-y-6">
                      <h3 className="text-xl font-bold flex items-center gap-2">
                        <Shield size={20} className="text-brand-400" />
                        Security & Policy
                      </h3>

                      <div className="space-y-6">
                        <div>
                          <label className="block text-xs font-bold uppercase tracking-widest text-white/40 mb-3">Browser Mode</label>
                          <div className="grid grid-cols-1 gap-3">
                            <button
                              onClick={async () => {
                                setBrowserMode(BrowserMode.HUMAN_PROFILE);
                                const { updateSettings: apiUpdateSettings } = await import('./services/apiService');
                                await apiUpdateSettings({ browser_mode: 'human_profile' });
                              }}
                              className={`p-4 rounded-xl border text-left transition-all ${browserMode === BrowserMode.HUMAN_PROFILE
                                ? 'bg-brand-500/10 border-brand-500/50 text-brand-400'
                                : 'bg-white/5 border-white/5 text-white/60 hover:bg-white/10'
                                }`}
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-bold">Human Profile</span>
                                {browserMode === BrowserMode.HUMAN_PROFILE && <CheckCircle2 size={16} />}
                              </div>
                              <p className="text-[10px] leading-relaxed opacity-60">Uses your existing Chrome profile with real cookies and history. Most reliable for security.</p>
                            </button>
                          </div>
                        </div>

                        <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5">
                          <div>
                            <div className="text-sm font-bold">Open in New Tabs</div>
                            <div className="text-[10px] text-white/40">Always open URLs in a new browser tab</div>
                          </div>
                          <button className="w-10 h-5 rounded-full bg-brand-500 relative">
                            <div className="absolute top-1 right-1 w-3 h-3 rounded-full bg-white" />
                          </button>
                        </div>
                      </div>
                    </section>

                    <section className="glass-panel p-8 rounded-3xl space-y-6">
                      <h3 className="text-xl font-bold flex items-center gap-2">
                        <Shield size={20} className="text-brand-400" />
                        Adapter Policy
                      </h3>

                      <div className="space-y-4">
                        <div>
                          <label className="block text-xs font-bold uppercase tracking-widest text-white/40 mb-3">Policy Level</label>
                          <div className="grid grid-cols-3 gap-2">
                            {[AdapterPolicy.STRICT, AdapterPolicy.BALANCED, AdapterPolicy.TRUSTED].map(p => (
                              <button
                                key={p}
                                onClick={() => setPolicy(p)}
                                className={`py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider border transition-all ${policy === p
                                  ? 'bg-brand-500/10 border-brand-500 text-brand-400'
                                  : 'bg-white/5 border-white/5 text-white/40 hover:bg-white/10'
                                  }`}
                              >
                                {p}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="p-4 rounded-xl bg-brand-500/5 border border-brand-500/10">
                          <div className="flex items-center gap-2 mb-2">
                            <Zap size={14} className="text-brand-400" />
                            <span className="text-xs font-bold text-brand-400">Policy Impact</span>
                          </div>
                          <p className="text-[10px] text-white/60 leading-relaxed">
                            {policy === AdapterPolicy.STRICT && "All sensitive actions (sending messages, downloads) require manual confirmation."}
                            {policy === AdapterPolicy.BALANCED && "Sensitive actions are allowed if they match the current plan context."}
                            {policy === AdapterPolicy.TRUSTED && "No restrictions. Autobot will execute all actions autonomously."}
                          </p>
                        </div>
                      </div>
                    </section>

                    <section className="glass-panel p-8 rounded-3xl space-y-6 md:col-span-2">
                      <h3 className="text-xl font-bold flex items-center gap-2">
                        <Palette size={20} className="text-brand-400" />
                        Interface & Theming
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-4">
                          <label className="text-xs font-bold text-white/40 uppercase tracking-widest">Accent Color</label>
                          <div className="grid grid-cols-2 gap-3">
                            {(['blue-violet', 'emerald', 'blue', 'amber'] as const).map((t) => (
                              <button
                                key={t}
                                onClick={() => setTheme(t)}
                                className={`p-4 rounded-2xl border transition-all flex flex-col items-center gap-2 ${theme === t ? 'bg-brand-500/10 border-brand-500 shadow-[0_0_20px_rgba(var(--brand-500-rgb),0.2)]' : 'bg-white/5 border-white/10 hover:bg-white/10'
                                  }`}
                              >
                                <div className={`w-6 h-6 rounded-full ${t === 'blue-violet' ? 'bg-[#8b5cf6]' :
                                  t === 'emerald' ? 'bg-[#22c55e]' :
                                    t === 'blue' ? 'bg-[#3b82f6]' : 'bg-[#f59e0b]'
                                  }`} />
                                <span className="text-[10px] font-bold uppercase tracking-widest capitalize">{t.replace('-', ' ')}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="space-y-4">
                          <label className="text-xs font-bold text-white/40 uppercase tracking-widest">Visual Effects</label>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10">
                              <div>
                                <div className="text-sm font-bold">Holographic Overlays</div>
                                <div className="text-[10px] text-white/40">Grid and glow effects on active panels</div>
                              </div>
                              <button className="w-10 h-5 rounded-full bg-brand-500 relative">
                                <div className="absolute right-1 top-1 w-3 h-3 rounded-full bg-white" />
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </section>

                    {/* AI Configuration */}
                    <section className="glass-panel p-8 rounded-3xl space-y-6 md:col-span-2">
                      <div className="flex items-center justify-between">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                          <Sparkles size={20} className="text-brand-400" />
                          AI Configuration
                        </h3>
                        <button
                          onClick={handleAddModel}
                          className="btn-secondary py-2 px-4 text-[10px] uppercase tracking-widest flex items-center gap-2"
                        >
                          <PlusCircle size={14} />
                          Add Model
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
                                  ? 'bg-brand-500/10 border-brand-500 text-brand-400 shadow-[0_0_20px_rgba(var(--brand-500-rgb),0.1)]'
                                  : 'bg-white/5 border-white/10 hover:bg-white/10'
                                  }`}
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
                              <Key size={18} />
                              <span className="text-sm font-bold">Secure Token Vault</span>
                            </div>
                            <p className="text-[10px] text-white/40 leading-relaxed">
                              Tokens are encrypted and stored in the secure backend vault. They are never exposed to the client after initial entry.
                            </p>
                            <div className="relative">
                              <input
                                type="password"
                                placeholder={isApiKeySaved ? "••••••••••••••••" : "Enter API Token..."}
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                disabled={isApiKeySaved}
                                className="input-field pr-12"
                              />
                              {isApiKeySaved ? (
                                <div className="absolute right-3 top-1/2 -translate-y-1/2 text-emerald-500">
                                  <CheckCircle2 size={18} />
                                </div>
                              ) : (
                                <button
                                  onClick={handleSaveApiKey}
                                  disabled={!apiKey.trim()}
                                  className="absolute right-2 top-1/2 -translate-y-1/2 btn-primary py-1.5 px-3 text-[10px] uppercase tracking-widest"
                                >
                                  Save
                                </button>
                              )}
                            </div>
                            {isApiKeySaved && (
                              <button
                                onClick={() => setIsApiKeySaved(false)}
                                className="text-[10px] font-bold uppercase tracking-widest text-white/20 hover:text-red-400 transition-colors"
                              >
                                Reset Token
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    </section>
                  </div>
                </motion.div>
              } />

              <Route path="/profile" element={
                <motion.div
                  key="profile"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-10"
                >
                  <header className="flex flex-col md:flex-row md:items-end gap-6">
                    <div className="relative group">
                      <img src={user.avatar} alt={user.name} className="w-32 h-32 rounded-3xl object-cover border-2 border-brand-500/20 shadow-2xl" />
                      <button className="absolute inset-0 flex items-center justify-center bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-3xl">
                        <ImageIcon size={24} />
                      </button>
                    </div>
                    <div className="flex-1">
                      <h2 className="text-4xl font-bold tracking-tight mb-1">{user.name}</h2>
                      <p className="text-brand-400 font-bold uppercase tracking-widest text-xs mb-4">{user.role}</p>
                      <div className="flex flex-wrap gap-2">
                        <span className="px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold uppercase tracking-widest">Verified Human</span>
                        <span className="px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-400 text-[10px] font-bold uppercase tracking-widest">Admin Access</span>
                      </div>
                    </div>
                    <button
                      onClick={() => setIsAuthenticated(false)}
                      className="btn-secondary border-red-500/20 text-red-400 hover:bg-red-500/10 hover:border-red-500/30 flex items-center gap-2"
                    >
                      <LogOut size={18} />
                      Terminate Session
                    </button>
                  </header>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <div className="md:col-span-2 space-y-8">
                      <section className="glass-panel p-8 rounded-3xl space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                          <User size={20} className="text-brand-400" />
                          Personal Information
                        </h3>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                          <div className="space-y-2">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-white/40 ml-1">Full Name</label>
                            <input type="text" className="input-field" defaultValue={user.name} />
                          </div>
                          <div className="space-y-2">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-white/40 ml-1">Email Address</label>
                            <input type="email" className="input-field" defaultValue={user.email} />
                          </div>
                          <div className="space-y-2">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-white/40 ml-1">Timezone</label>
                            <select className="input-field appearance-none">
                              <option>Pacific Time (PT)</option>
                              <option>Eastern Time (ET)</option>
                              <option>Greenwich Mean Time (GMT)</option>
                            </select>
                          </div>
                          <div className="space-y-2">
                            <label className="text-[10px] font-bold uppercase tracking-widest text-white/40 ml-1">Language</label>
                            <select className="input-field appearance-none">
                              <option>English (US)</option>
                              <option>Spanish</option>
                              <option>French</option>
                            </select>
                          </div>
                        </div>
                        <div className="pt-4">
                          <button className="btn-primary">Save Changes</button>
                        </div>
                      </section>

                      <section className="glass-panel p-8 rounded-3xl space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                          <Smartphone size={20} className="text-brand-400" />
                          Device Authorization
                        </h3>
                        <div className="space-y-4">
                          <div className="flex items-center justify-between p-4 rounded-2xl bg-white/5 border border-white/10">
                            <div className="flex items-center gap-4">
                              <div className="p-3 rounded-xl bg-brand-500/10 text-brand-400">
                                <Smartphone size={20} />
                              </div>
                              <div>
                                <div className="text-sm font-bold">iPhone 15 Pro</div>
                                <div className="text-[10px] text-white/40">Last active: 2 minutes ago • San Francisco, CA</div>
                              </div>
                            </div>
                            <button className="text-[10px] font-bold uppercase tracking-widest text-red-400 hover:underline">Revoke</button>
                          </div>
                          <div className="flex items-center justify-between p-4 rounded-2xl bg-white/5 border border-white/10 opacity-50">
                            <div className="flex items-center gap-4">
                              <div className="p-3 rounded-xl bg-white/10 text-white/40">
                                <Globe size={20} />
                              </div>
                              <div>
                                <div className="text-sm font-bold">MacBook Pro 16"</div>
                                <div className="text-[10px] text-white/40">Last active: 3 days ago • London, UK</div>
                              </div>
                            </div>
                            <button className="text-[10px] font-bold uppercase tracking-widest text-white/20 hover:text-white">Authorize</button>
                          </div>
                        </div>
                      </section>
                    </div>

                    <div className="space-y-8">
                      <section className="glass-panel p-8 rounded-3xl space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                          <Activity size={20} className="text-brand-400" />
                          Usage Stats
                        </h3>
                        <div className="space-y-6">
                          <div className="space-y-2">
                            <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-white/40">
                              <span>Monthly API Credits</span>
                              <span>84%</span>
                            </div>
                            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                              <div className="h-full w-[84%] bg-brand-500 shadow-[0_0_10px_rgba(var(--brand-500-rgb),0.5)]" />
                            </div>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-white/40">
                              <span>Storage Capacity</span>
                              <span>12%</span>
                            </div>
                            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                              <div className="h-full w-[12%] bg-accent-cyan shadow-[0_0_10px_rgba(6,182,212,0.5)]" />
                            </div>
                          </div>
                        </div>
                        <div className="pt-4 grid grid-cols-2 gap-4">
                          <div className="text-center p-4 rounded-2xl bg-white/5">
                            <div className="text-xl font-bold">1,284</div>
                            <div className="text-[10px] text-white/40 uppercase tracking-widest">Tasks Run</div>
                          </div>
                          <div className="text-center p-4 rounded-2xl bg-white/5">
                            <div className="text-xl font-bold">42h</div>
                            <div className="text-[10px] text-white/40 uppercase tracking-widest">Backend Uptime</div>
                          </div>
                        </div>
                      </section>

                      <section className="glass-panel p-8 rounded-3xl space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                          <Lock size={20} className="text-brand-400" />
                          Security Level
                        </h3>
                        <div className="flex items-center gap-4">
                          <div className="w-16 h-16 rounded-full border-4 border-brand-500/20 flex items-center justify-center relative">
                            <div className="absolute inset-0 rounded-full border-4 border-brand-500 border-t-transparent animate-spin" />
                            <span className="text-lg font-bold">A+</span>
                          </div>
                          <div>
                            <div className="text-sm font-bold">High Security</div>
                            <div className="text-[10px] text-white/40 leading-relaxed">Your account is protected by 2FA and hardware keys.</div>
                          </div>
                        </div>
                        <button className="btn-secondary w-full py-2 text-xs">Security Audit</button>
                      </section>
                    </div>
                  </div>
                </motion.div>
              } />
            </Routes>
          </AnimatePresence>
        </div>
      </main>

      {/* Status Bar */}
      <footer className="fixed bottom-0 left-0 md:left-72 right-0 h-10 glass-panel border-t border-white/5 flex items-center px-4 md:px-8 justify-between z-20">
        <div className="flex items-center gap-4 md:gap-6">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-brand-400 shadow-[0_0_8px_rgba(var(--brand-500-rgb),0.6)]" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-white/60 hidden sm:inline">System Ready</span>
          </div>
          <div className="flex items-center gap-2">
            <MousePointer2 size={12} className="text-white/30" />
            <span className="text-[10px] text-white/40 hidden sm:inline">Mouse: Active</span>
          </div>
          <div className="flex items-center gap-2">
            <Keyboard size={12} className="text-white/30" />
            <span className="text-[10px] text-white/40 hidden sm:inline">Keyboard: Active</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] text-white/30 font-mono hidden xs:inline">v1.2.4-stable</span>
          <div className="h-4 w-[1px] bg-white/10 hidden xs:inline" />
          <div className="flex items-center gap-2 text-white/40 hover:text-white transition-colors cursor-pointer">
            <span className="text-[10px] font-bold uppercase tracking-widest">Docs</span>
            <ExternalLink size={10} />
          </div>
        </div>
      </footer>

      {/* Detail Modals */}
      <AnimatePresence>
        {selectedRun && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedRun(null)}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="w-full max-w-2xl glass-panel p-8 rounded-3xl relative z-10 overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-6">
                <button onClick={() => setSelectedRun(null)} className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors">
                  <X size={20} />
                </button>
              </div>
              <div className="flex items-center gap-4 mb-8">
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${selectedRun.status === 'success' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                  }`}>
                  {selectedRun.status === 'success' ? <CheckCircle2 size={28} /> : <XCircle size={28} />}
                </div>
                <div>
                  <h3 className="text-2xl font-bold">{selectedRun.planName}</h3>
                  <p className="text-white/40 text-sm">{selectedRun.timestamp} • {selectedRun.id}</p>
                </div>
              </div>
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-2xl bg-white/5">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 mb-1">Steps</div>
                    <div className="text-xl font-bold">{selectedRun.stepsCompleted} / {selectedRun.totalSteps}</div>
                  </div>
                  <div className="p-4 rounded-2xl bg-white/5">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-white/40 mb-1">Status</div>
                    <div className={`text-xl font-bold uppercase tracking-widest ${selectedRun.status === 'success' ? 'text-emerald-400' : 'text-red-400'
                      }`}>{selectedRun.status}</div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="text-[10px] font-bold uppercase tracking-widest text-white/40">Execution Logs</div>
                  <div className="glass-panel bg-black/40 rounded-xl p-4 h-48 overflow-y-auto font-mono text-xs space-y-2 custom-scrollbar">
                    {selectedRun.logs.map((log, i) => (
                      <div key={i} className="flex gap-3">
                        <span className="text-white/20">[{i}]</span>
                        <span className="text-white/60">{log}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}

        {selectedArtifact && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedArtifact(null)}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="w-full max-w-lg glass-panel p-8 rounded-3xl relative z-10"
            >
              <div className="absolute top-0 right-0 p-6">
                <button onClick={() => setSelectedArtifact(null)} className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors">
                  <X size={20} />
                </button>
              </div>
              <div className="flex items-center gap-4 mb-8">
                <div className="w-14 h-14 rounded-2xl bg-brand-500/10 text-brand-400 flex items-center justify-center">
                  {selectedArtifact.type === 'email' ? <Mail size={28} /> : <FileText size={28} />}
                </div>
                <div>
                  <h3 className="text-2xl font-bold">{selectedArtifact.title}</h3>
                  <p className="text-white/40 text-sm">Generated {selectedArtifact.timestamp}</p>
                </div>
              </div>
              <div className="space-y-6">
                <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
                  <div className="text-[10px] font-bold uppercase tracking-widest text-brand-400 mb-4">Content Preview</div>
                  <div className="text-sm text-white/60 leading-relaxed">
                    This is a simulated preview of the artifact content. In a real application, this would display the actual data retrieved from the backend or generated by the AI agent.
                  </div>
                </div>
                <div className="flex gap-3">
                  <button className="btn-primary flex-1 py-3 text-xs uppercase tracking-widest">Download</button>
                  <button className="btn-secondary flex-1 py-3 text-xs uppercase tracking-widest">Share</button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

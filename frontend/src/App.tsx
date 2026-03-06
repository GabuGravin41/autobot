/**
 * App.tsx — Slim shell (was 102KB monolith, now ~200 lines)
 * All page content lives in src/components/*.tsx
 */

import React, { useState, useEffect } from 'react';
import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom';
import { AnimatePresence, motion } from 'motion/react';
import { Bot, MousePointer2, Keyboard, ExternalLink, LogOut } from 'lucide-react';

// Types
import {
  BrowserMode, AdapterPolicy, RunHistory, WorkflowPlan, LLMModel, UserProfile,
} from './types';

// Services
import {
  getStatus, getAdapters, getRuns, getWorkflows, runPlan, cancelRun,
  connectLogStream, updateSettings, getBrowserScreenshotUrl, submitHumanInput,
  getRun as apiGetRun, deleteRun as apiDeleteRun, sendChat,
  runAutonomous, cancelAutonomous, getAutonomousStatus,
  BackendStatus, BackendAdapter, BackendRun, BackendWorkflow,
} from './services/apiService';

// Components
import Sidebar from './components/Sidebar';
import DashboardPage from './components/DashboardPage';
import ChatPanel from './components/ChatPanel';
import WorkflowsPage from './components/WorkflowsPage';
import HistoryPage from './components/HistoryPage';
import SettingsPage from './components/SettingsPage';
import ProfilePage from './components/ProfilePage';
import Modals from './components/Modals';

// ── Constants ────────────────────────────────────────────────────────────────
const MOCK_USER: UserProfile = {
  name: 'Alex Rivera',
  email: 'alex@autobot.ai',
  avatar: 'https://picsum.photos/seed/alex/100/100',
  role: 'System Architect',
};

const DEFAULT_MODELS: LLMModel[] = [
  { id: 'google/gemini-2.0-flash-001', name: 'Gemini 2.0 Flash', provider: 'Google' },
  { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro', provider: 'Google' },
  { id: 'deepseek/deepseek-chat', name: 'DeepSeek V3', provider: 'OpenRouter' },
  { id: 'gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
];

export default function App() {
  // ── Auth ─────────────────────────────────────────────────────────────────
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  // ── Backend state ─────────────────────────────────────────────────────────
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [liveAdapters, setLiveAdapters] = useState<BackendAdapter[]>([]);
  const [liveRuns, setLiveRuns] = useState<RunHistory[]>([]);
  const [liveLogLines, setLiveLogLines] = useState<string[]>([]);
  const [workflows, setWorkflows] = useState<BackendWorkflow[]>([]);
  const [backendOnline, setBackendOnline] = useState(false);
  const [screenshotUrl, setScreenshotUrl] = useState('');
  const [activeRun, setActiveRun] = useState<RunHistory | null>(null);

  // ── UI state ──────────────────────────────────────────────────────────────
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [browserMode, setBrowserMode] = useState<BrowserMode>(BrowserMode.HUMAN_PROFILE);
  const [policy, setPolicy] = useState<AdapterPolicy>(AdapterPolicy.BALANCED);
  const [models, setModels] = useState<LLMModel[]>(DEFAULT_MODELS);
  const [selectedModelId, setSelectedModelId] = useState('google/gemini-2.0-flash-001');

  // ── Chat / planner ────────────────────────────────────────────────────────
  const [chatMessages, setChatMessages] = useState<any[]>([
    { role: 'bot', content: 'Hello! I am Autobot. How can I help you automate your computer today?' },
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showReasoning, setShowReasoning] = useState<Record<number, boolean>>({});

  // ── Modals ────────────────────────────────────────────────────────────────
  const [selectedRun, setSelectedRun] = useState<RunHistory | null>(null);
  const [selectedArtifact, setSelectedArtifact] = useState<any | null>(null);

  // ── Effects ───────────────────────────────────────────────────────────────
  useEffect(() => {
    let disconnect: (() => void) | null = null;
    const poll = async () => {
      try {
        const [status, adapters, runs, wfs] = await Promise.all([
          getStatus(), getAdapters(), getRuns(),
          getWorkflows().catch(() => ({ workflows: [] })),
        ]);
        setBackendStatus(status);
        setLiveAdapters(adapters.adapters);
        setLiveRuns(runs.runs as any as RunHistory[]);
        if (wfs.workflows?.length) setWorkflows(wfs.workflows);
        setBackendOnline(true);
      } catch { setBackendOnline(false); }
    };
    poll();
    const interval = setInterval(poll, 5000);
    disconnect = connectLogStream(
      (line) => setLiveLogLines(prev => [...prev.slice(-199), line]),
      undefined,
      { usePollingFallback: true, onLogsSnapshot: (logs) => setLiveLogLines(logs.slice(-200)) },
    );
    return () => { clearInterval(interval); disconnect?.(); };
  }, []);

  useEffect(() => {
    if (!backendStatus) return;
    const { run_status, active_run_id } = backendStatus;

    if (run_status === 'running' && active_run_id) {
      // If we don't have a local activeRun or it's just a placeholder, fetch full details
      if (!activeRun || (activeRun.id !== active_run_id && activeRun.id !== 'pending')) {
        apiGetRun(active_run_id).then(run => {
          setActiveRun({ ...run, logs: liveLogLines.length > run.logs?.length ? liveLogLines : (run.logs || []) });
        }).catch(() => {
          // Fallback to placeholder if fetch fails
          setActiveRun({
            id: active_run_id,
            planName: 'Active Operation',
            timestamp: new Date().toLocaleString(),
            status: 'running',
            stepsCompleted: 0,
            totalSteps: 0,
            progress: 0,
            artifacts: {},
            screenshots: [],
            logs: liveLogLines,
          });
        });
      } else {
        // Just sync logs and status
        setActiveRun(prev => prev ? { ...prev, status: 'running', logs: liveLogLines } : null);
      }
    } else if (run_status === 'done' || run_status === 'failed' || run_status === 'cancelled') {
      const finalStatus = run_status === 'done' ? 'success' : (run_status === 'cancelled' ? 'failed' : 'failed');
      setActiveRun(prev => prev ? { ...prev, status: finalStatus, logs: liveLogLines } : null);
    } else if (run_status === 'idle') {
      setActiveRun(prev => (prev?.id === 'pending' ? prev : null));
    }
  }, [backendStatus?.run_status, backendStatus?.active_run_id, liveLogLines]);

  useEffect(() => {
    if (!backendOnline || !backendStatus?.browser?.active) return;
    setScreenshotUrl(getBrowserScreenshotUrl());
    const interval = setInterval(() => setScreenshotUrl(getBrowserScreenshotUrl()), 3000);
    return () => clearInterval(interval);
  }, [backendOnline, backendStatus?.browser?.active]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleSendMessage = async () => {
    if (!chatInput.trim() || isGenerating) return;
    const userMsg = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsGenerating(true);
    try {

      const { reply, plan } = await sendChat(userMsg, {
        browser_mode: backendStatus?.browser?.mode,
        llm_enabled: backendStatus?.llm_enabled,
      });
      setChatMessages(prev => [...prev, { role: 'bot', content: reply, plan: plan ? { id: plan.id, name: plan.name, description: plan.description, steps: plan.steps } : undefined }]);
    } catch (error) {
      setChatMessages(prev => [...prev, { role: 'bot', content: `Sorry, I couldn't reach the Autobot backend. Error: ${error instanceof Error ? error.message : 'Unknown'}` }]);
    } finally { setIsGenerating(false); }
  };

  const executePlan = async (plan: WorkflowPlan) => {
    navigate('/dashboard');
    const optimistic: RunHistory = { id: 'pending', planName: plan.name, timestamp: new Date().toLocaleString(), status: 'running', stepsCompleted: 0, totalSteps: plan.steps?.length || 0, artifacts: {}, screenshots: [], logs: [`▶ Sending plan '${plan.name}' to engine...`] };
    setActiveRun(optimistic);
    try {
      const { run_id } = await runPlan(plan as any);
      setActiveRun(prev => prev ? { ...prev, id: run_id } : null);
    } catch (error) {
      setActiveRun(prev => prev ? { ...prev, status: 'failed', logs: [...(prev.logs || []), `✗ Failed: ${error instanceof Error ? error.message : 'Unknown'}`] } : null);
    }
  };

  const handleAbortRun = async () => {
    try {
      if (activeRun?.id && activeRun.id !== 'pending') {
        await cancelRun(activeRun.id);
      }
      await cancelAutonomous();
    } catch (_) { }
    setActiveRun(prev => prev ? { ...prev, status: 'failed', logs: [...(prev.logs || []), '⚠ Aborted.'] } : null);
  };

  const handleDeleteRun = async (runId: string) => {
    if (!confirm('Delete this run?')) return;
    try { await apiDeleteRun(runId); setLiveRuns(prev => prev.filter(r => r.id !== runId)); } catch (e) { alert('Failed: ' + (e instanceof Error ? e.message : '')); }
  };

  const handleViewDetails = async (run: RunHistory) => {
    try { const d = await apiGetRun(run.id); setSelectedRun({ ...run, logs: d.logs || [] }); }
    catch (_) { setSelectedRun(run); }
  };

  // ── Login screen ──────────────────────────────────────────────────────────
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen  flex items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-[var(--brand-primary)]/20 blur-[120px] rounded-full -z-10 animate-pulse" />
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="w-full max-w-md glass-panel p-10 rounded-3xl space-y-8">
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-[var(--brand-primary)] flex items-center justify-center shadow-xl shadow-[var(--brand-primary)]/30">
              <Bot className="text-white" size={32} />
            </div>
            <div className="text-center">
              <h1 className="text-3xl font-bold tracking-tight">AUTOBOT</h1>
              <p className="text-xs uppercase tracking-[0.3em] text-[var(--brand-primary)] font-bold">Secure Access</p>
            </div>
          </div>
          <div className="space-y-4">
            <input type="text" placeholder="Username or Email" className="input-field w-full" defaultValue="alex@autobot.ai" />
            <input type="password" placeholder="••••••••" className="input-field w-full" defaultValue="password" />
          </div>
          <button onClick={() => setIsAuthenticated(true)} className="btn-primary w-full py-4 text-sm uppercase tracking-widest">
            Initialize Session
          </button>
        </motion.div>
      </div>
    );
  }

  // ── Main layout ───────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col md:flex-row font-sans overflow-hidden ">
      <Sidebar
        user={MOCK_USER}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
        isCollapsed={isSidebarCollapsed}
        setIsCollapsed={setIsSidebarCollapsed}
        onLogout={() => setIsAuthenticated(false)}
      />

      <main className="flex-1 h-[calc(100vh-64px)] md:h-screen overflow-y-auto  relative custom-scrollbar">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-[var(--brand-primary)]/20 blur-[120px] rounded-full -z-10" />
        <div className="max-w-7xl mx-auto p-6 md:p-10 pb-24">
          <AnimatePresence mode="wait">
            <Routes location={location}>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={
                <DashboardPage
                  backendOnline={backendOnline} backendStatus={backendStatus}
                  activeRun={activeRun} liveRuns={liveRuns} liveAdapters={liveAdapters}
                  liveLogLines={liveLogLines} screenshotUrl={screenshotUrl}
                  onRefreshScreenshot={() => setScreenshotUrl(getBrowserScreenshotUrl())}
                  onAbortRun={handleAbortRun}
                  onSelectRun={handleViewDetails}
                  onSelectArtifact={setSelectedArtifact}
                />
              } />
              <Route path="/planner" element={
                <ChatPanel
                  messages={chatMessages} input={chatInput} setInput={setChatInput}
                  isGenerating={isGenerating} onSend={handleSendMessage}
                  onExecutePlan={executePlan}
                  showReasoning={showReasoning} setShowReasoning={setShowReasoning}
                />
              } />
              <Route path="/workflows" element={
                <WorkflowsPage
                  workflows={workflows}
                  isRunning={backendStatus?.run_status === 'running'}
                  onRunStart={setActiveRun}
                  onRunIdUpdate={(id) => setActiveRun(prev => prev ? { ...prev, id } : null)}
                  onRunError={(msg) => setActiveRun(prev => prev ? { ...prev, status: 'failed', logs: [...(prev.logs || []), 'Failed: ' + msg] } : null)}
                />
              } />
              <Route path="/history" element={
                <HistoryPage
                  runs={liveRuns}
                  onViewDetails={handleViewDetails}
                  onDeleteRun={handleDeleteRun}
                />
              } />
              <Route path="/settings" element={
                <SettingsPage
                  browserMode={browserMode} setBrowserMode={setBrowserMode}
                  policy={policy} setPolicy={setPolicy}
                  theme={theme} setTheme={setTheme}
                  models={models} setModels={setModels}
                  selectedModelId={selectedModelId} setSelectedModelId={setSelectedModelId}
                />
              } />
              <Route path="/profile" element={
                <ProfilePage
                  user={MOCK_USER}
                  onLogout={() => setIsAuthenticated(false)}
                />
              } />
            </Routes>
          </AnimatePresence>
        </div>
      </main>

      {/* Status bar */}
      <footer className={`fixed bottom-0 left-0 ${isSidebarCollapsed ? 'md:left-20' : 'md:left-72'} right-0 h-10 glass-panel border-t border-[var(--base-border)] flex items-center px-4 md:px-8 justify-between z-20 transition-all duration-300`}>
        <div className="flex items-center gap-4 md:gap-6">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--brand-primary)] shadow-md shadow-[var(--brand-primary)]/40" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] hidden sm:inline">System Ready</span>
          </div>
          <div className="flex items-center gap-2"><MousePointer2 size={12} className="text-[var(--base-text-muted)]" /><span className="text-[10px] text-[var(--base-text-muted)] hidden sm:inline">Mouse: Active</span></div>
          <div className="flex items-center gap-2"><Keyboard size={12} className="text-[var(--base-text-muted)]" /><span className="text-[10px] text-[var(--base-text-muted)] hidden sm:inline">Keyboard: Active</span></div>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] text-[var(--base-text-muted)] font-mono hidden xs:inline">v2.0.0-alpha</span>
          <div className="flex items-center gap-2 text-[var(--base-text-muted)] hover:text-[var(--base-text)] transition-colors cursor-pointer">
            <span className="text-[10px] font-bold uppercase tracking-widest">Docs</span>
            <ExternalLink size={10} />
          </div>
        </div>
      </footer>

      <Modals
        backendStatus={backendStatus}
        selectedRun={selectedRun} onCloseRun={() => setSelectedRun(null)}
        selectedArtifact={selectedArtifact} onCloseArtifact={() => setSelectedArtifact(null)}
      />
    </div>
  );
}

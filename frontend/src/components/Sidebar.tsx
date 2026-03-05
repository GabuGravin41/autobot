import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'motion/react';
import {
    Bot, LayoutDashboard, Sparkles, Zap, History, Settings, LogOut, X, Menu,
} from 'lucide-react';
import { UserProfile } from '../types';

interface SidebarProps {
    user: UserProfile;
    isOpen: boolean;
    setIsOpen: (v: boolean) => void;
    isAutonomous: boolean;
    setIsAutonomous: (v: boolean) => void;
    onLogout: () => void;
}

const NavItem = ({
    icon: Icon, label, active, onClick,
}: { icon: any; label: string; active: boolean; onClick: () => void }) => (
    <button
        onClick={onClick}
        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${active
                ? 'bg-brand-500/10 text-brand-400 border border-brand-500/20 shadow-[0_0_20px_rgba(34,197,94,0.1)]'
                : 'text-white/50 hover:text-white hover:bg-white/5'
            }`}
    >
        <Icon size={20} className={active ? 'animate-pulse' : ''} />
        <span className="font-medium">{label}</span>
        {active && (
            <motion.div
                layoutId="active-pill"
                className="ml-auto w-1.5 h-1.5 rounded-full bg-brand-400 shadow-[0_0_10px_rgba(34,197,94,0.8)]"
            />
        )}
    </button>
);

export default function Sidebar({
    user, isOpen, setIsOpen, isAutonomous, setIsAutonomous, onLogout,
}: SidebarProps) {
    const navigate = useNavigate();
    const location = useLocation();
    const activeTab = location.pathname.substring(1) || 'dashboard';

    const go = (path: string) => { navigate(path); setIsOpen(false); };

    return (
        <>
            {/* Mobile header */}
            <div className="md:hidden h-16 glass-panel border-b border-white/5 flex items-center justify-between px-6 z-50">
                <div className="flex items-center gap-2">
                    <Bot className="text-brand-500" size={24} />
                    <span className="font-bold tracking-tight">AUTOBOT</span>
                </div>
                <button onClick={() => setIsOpen(!isOpen)} className="p-2 rounded-lg bg-white/5 text-white/60">
                    {isOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>

            {/* Sidebar panel */}
            <aside className={`
        fixed inset-0 z-40 md:relative md:translate-x-0 transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        w-72 glass-panel border-r border-white/5 flex flex-col p-6
      `}>
                <div className="hidden md:flex items-center gap-3 mb-10 px-2">
                    <motion.div
                        animate={{ rotate: [0, 10, -10, 0], scale: [1, 1.1, 1] }}
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
                    <NavItem icon={LayoutDashboard} label="Dashboard" active={activeTab === 'dashboard'} onClick={() => go('/dashboard')} />
                    <NavItem icon={Sparkles} label="AI Planner" active={activeTab === 'planner'} onClick={() => go('/planner')} />
                    <NavItem icon={Zap} label="Workflows" active={activeTab === 'workflows'} onClick={() => go('/workflows')} />
                    <NavItem icon={History} label="Run History" active={activeTab === 'history'} onClick={() => go('/history')} />
                    <NavItem icon={Settings} label="Settings" active={activeTab === 'settings'} onClick={() => go('/settings')} />
                </nav>

                <div className="mt-auto pt-6 border-t border-white/5 space-y-6">
                    {/* Autonomous mode toggle */}
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

                    {/* User card */}
                    <div
                        onClick={() => go('/profile')}
                        className={`glass-card p-3 rounded-2xl flex items-center gap-3 group cursor-pointer transition-all ${activeTab === 'profile' ? 'bg-brand-500/10 border-brand-500/30' : ''
                            }`}
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
                            onClick={(e) => { e.stopPropagation(); onLogout(); }}
                            className="p-1.5 rounded-lg hover:bg-red-500/10 text-white/20 hover:text-red-400 transition-colors"
                        >
                            <LogOut size={14} />
                        </button>
                    </div>
                </div>
            </aside>
        </>
    );
}

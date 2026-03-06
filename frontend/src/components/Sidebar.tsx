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
    isCollapsed: boolean;
    setIsCollapsed: (v: boolean) => void;
    onLogout: () => void;
}

const NavItem = ({
    icon: Icon, label, active, collapsed, onClick,
}: { icon: any; label: string; active: boolean; collapsed: boolean; onClick: () => void }) => (
    <button
        onClick={onClick}
        title={collapsed ? label : undefined}
        className={`w-full flex items-center ${collapsed ? 'justify-center p-3' : 'gap-3 px-4 py-3'} rounded-xl transition-all duration-300 ${active
            ? 'bg-[var(--brand-primary)]/10 text-[var(--brand-primary)]'
            : 'text-[var(--base-text-muted)] hover:text-[var(--base-text)] hover:bg-[var(--base-border)]'
            }`}
    >
        <Icon size={20} className={active ? 'animate-pulse' : ''} />
        {!collapsed && <span className="font-medium">{label}</span>}
        {active && !collapsed && (
            <motion.div
                layoutId="active-pill"
                className="ml-auto w-1.5 h-1.5 rounded-full bg-[var(--brand-primary)] shadow-[0_0_10px_rgba(59,130,246,0.8)]"
            />
        )}
    </button>
);

export default function Sidebar({
    user, isOpen, setIsOpen, isCollapsed, setIsCollapsed, onLogout,
}: SidebarProps) {
    const navigate = useNavigate();
    const location = useLocation();
    const activeTab = location.pathname.substring(1) || 'dashboard';

    const go = (path: string) => { navigate(path); setIsOpen(false); };

    return (
        <>
            {/* Mobile header */}
            <div className="md:hidden h-16 glass-panel border-b border-[var(--base-border)] flex items-center justify-between px-6 z-50">
                <div className="flex items-center gap-2">
                    <Bot className="text-[var(--brand-primary)]" size={24} />
                    <span className="font-bold tracking-tight">AUTOBOT</span>
                </div>
                <button onClick={() => setIsOpen(!isOpen)} className="p-2 rounded-lg bg-[var(--base-border)] text-[var(--base-text-muted)] hover:text-[var(--base-text)]">
                    {isOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>

            {/* Sidebar panel */}
            <aside className={`
        fixed inset-0 z-40 md:relative md:translate-x-0 transition-all duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        ${isCollapsed ? 'md:w-20' : 'md:w-72'} w-72 glass-panel border-r border-[var(--base-border)] flex flex-col p-4 md:p-6
      `}>
                <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'justify-between'} mb-10 px-2`}>
                    <div className="hidden md:flex items-center gap-3 cursor-pointer" onClick={() => go('/dashboard')}>
                        <motion.div
                            animate={{ rotate: [0, 10, -10, 0], scale: [1, 1.1, 1] }}
                            transition={{ duration: 4, repeat: Infinity }}
                            className="w-10 h-10 rounded-xl bg-[var(--brand-primary)] flex items-center justify-center shadow-lg shadow-[var(--brand-primary)]/20"
                        >
                            <Bot className="text-white" size={24} />
                        </motion.div>
                        {!isCollapsed && (
                            <div>
                                <h1 className="text-xl font-bold tracking-tight">AUTOBOT</h1>
                                <p className="text-[10px] uppercase tracking-[0.2em] text-[var(--brand-primary)] font-bold">Control Center</p>
                            </div>
                        )}
                    </div>
                    {/* Desktop collapse toggle */}
                    <button
                        onClick={() => setIsCollapsed(!isCollapsed)}
                        className="hidden md:flex p-1.5 rounded-lg bg-[var(--base-border)] text-[var(--base-text-muted)] hover:text-[var(--base-text)] transition-colors"
                    >
                        <Menu size={16} />
                    </button>
                </div>

                <nav className="flex-1 space-y-2">
                    <NavItem icon={LayoutDashboard} label="Dashboard" active={activeTab === 'dashboard'} collapsed={isCollapsed} onClick={() => go('/dashboard')} />
                    <NavItem icon={Sparkles} label="AI Planner" active={activeTab === 'planner'} collapsed={isCollapsed} onClick={() => go('/planner')} />
                    <NavItem icon={Zap} label="Workflows" active={activeTab === 'workflows'} collapsed={isCollapsed} onClick={() => go('/workflows')} />
                    <NavItem icon={History} label="Run History" active={activeTab === 'history'} collapsed={isCollapsed} onClick={() => go('/history')} />
                    <NavItem icon={Settings} label="Settings" active={activeTab === 'settings'} collapsed={isCollapsed} onClick={() => go('/settings')} />
                </nav>

                <div className="mt-auto pt-6 border-t border-[var(--base-border)] space-y-6">
                    {/* User card */}
                    <div
                        onClick={() => go('/profile')}
                        className={`flex items-center ${isCollapsed ? 'justify-center p-2' : 'gap-3 p-3'} rounded-2xl group cursor-pointer transition-all hover:bg-[var(--base-border)] ${activeTab === 'profile' ? 'bg-[var(--brand-primary)]/10 text-[var(--brand-primary)]' : ''
                            }`}
                        title={isCollapsed ? user.name : undefined}
                    >
                        <div className="relative flex-shrink-0">
                            <img src={user.avatar} alt={user.name} className="w-10 h-10 rounded-xl object-cover border border-[var(--base-border)]" />
                            <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 border-2 border-white dark:border-[#0a0a0c] flex items-center justify-center">
                                <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                            </div>
                        </div>
                        {!isCollapsed && (
                            <>
                                <div className="flex-1 min-w-0">
                                    <div className="text-xs font-bold truncate text-[var(--base-text)] group-hover:text-[var(--brand-primary)]">{user.name}</div>
                                    <div className="text-[10px] text-[var(--base-text-muted)] truncate">{user.role}</div>
                                </div>
                                <button
                                    onClick={(e) => { e.stopPropagation(); onLogout(); }}
                                    className="p-1.5 rounded-lg hover:bg-red-500/10 text-[var(--base-text-muted)] hover:text-red-500 transition-colors"
                                >
                                    <LogOut size={14} />
                                </button>
                            </>
                        )}
                    </div>
                </div>
            </aside>
        </>
    );
}

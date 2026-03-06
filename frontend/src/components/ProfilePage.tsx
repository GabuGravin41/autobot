import React from 'react';
import { motion } from 'motion/react';
import {
    User, ImageIcon, LogOut, Smartphone, Globe, Activity, Lock,
} from 'lucide-react';
import { UserProfile } from '../types';

interface ProfilePageProps {
    user: UserProfile;
    onLogout: () => void;
}

export default function ProfilePage({ user, onLogout }: ProfilePageProps) {
    return (
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
                    <button className="absolute inset-0 flex items-center justify-center bg-[var(--base-border)] opacity-0 group-hover:opacity-100 transition-opacity rounded-3xl">
                        <ImageIcon size={24} />
                    </button>
                </div>
                <div className="flex-1">
                    <h2 className="text-4xl font-bold tracking-tight mb-1">{user.name}</h2>
                    <p className="text-[var(--brand-primary)] font-bold uppercase tracking-widest text-xs mb-4">{user.role}</p>
                    <div className="flex flex-wrap gap-2">
                        <span className="px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold uppercase tracking-widest">Verified Human</span>
                        <span className="px-3 py-1 rounded-full bg-[var(--brand-primary)]/20 border border-brand-500/20 text-[var(--brand-primary)] text-[10px] font-bold uppercase tracking-widest">Admin Access</span>
                    </div>
                </div>
                <button
                    onClick={onLogout}
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
                            <User size={20} className="text-[var(--brand-primary)]" />
                            Personal Information
                        </h3>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] ml-1">Full Name</label>
                                <input type="text" className="input-field" defaultValue={user.name} />
                            </div>
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)] ml-1">Email Address</label>
                                <input type="email" className="input-field" defaultValue={user.email} />
                            </div>
                        </div>
                        <div className="pt-4">
                            <button className="btn-primary">Save Changes</button>
                        </div>
                    </section>

                    <section className="glass-panel p-8 rounded-3xl space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <Smartphone size={20} className="text-[var(--brand-primary)]" />
                            Device Authorization
                        </h3>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between p-4 rounded-2xl bg-[var(--base-border)] border border-[var(--base-border)]">
                                <div className="flex items-center gap-4">
                                    <div className="p-3 rounded-xl bg-[var(--brand-primary)]/20 text-[var(--brand-primary)]">
                                        <Smartphone size={20} />
                                    </div>
                                    <div>
                                        <div className="text-sm font-bold">iPhone 15 Pro</div>
                                        <div className="text-[10px] text-[var(--base-text-muted)]">Last active: 2 minutes ago • San Francisco, CA</div>
                                    </div>
                                </div>
                                <button className="text-[10px] font-bold uppercase tracking-widest text-red-400 hover:underline">Revoke</button>
                            </div>
                        </div>
                    </section>
                </div>

                <div className="space-y-8">
                    <section className="glass-panel p-8 rounded-3xl space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <Activity size={20} className="text-[var(--brand-primary)]" />
                            Usage Stats
                        </h3>
                        <div className="space-y-6">
                            <div className="space-y-2">
                                <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-[var(--base-text-muted)]">
                                    <span>Monthly API Credits</span>
                                    <span>84%</span>
                                </div>
                                <div className="h-1.5 w-full bg-[var(--base-border)] rounded-full overflow-hidden">
                                    <div className="h-full w-[84%] bg-[var(--brand-primary)] shadow-[0_0_10px_rgba(var(--brand-500-rgb),0.5)]" />
                                </div>
                            </div>
                        </div>
                    </section>

                    <section className="glass-panel p-8 rounded-3xl space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                            <Lock size={20} className="text-[var(--brand-primary)]" />
                            Security Level
                        </h3>
                        <div className="flex items-center gap-4">
                            <div className="w-16 h-16 rounded-full border-4 border-brand-500/20 flex items-center justify-center relative">
                                <div className="absolute inset-0 rounded-full border-4 border-brand-500 border-t-transparent animate-spin" />
                                <span className="text-lg font-bold">A+</span>
                            </div>
                            <div>
                                <div className="text-sm font-bold">High Security</div>
                                <div className="text-[10px] text-[var(--base-text-muted)] leading-relaxed">Your account is protected by 2FA.</div>
                            </div>
                        </div>
                    </section>
                </div>
            </div>
        </motion.div>
    );
}

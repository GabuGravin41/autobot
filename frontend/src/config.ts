// Frontend API configuration — reads from Vite env vars so the same build
// works locally (pointing to localhost) and on Vercel (pointing to your tunnel URL).
export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";
export const WS_BASE = (import.meta.env.VITE_WS_BASE as string | undefined)?.replace(/\/$/, "") ??
    (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host;

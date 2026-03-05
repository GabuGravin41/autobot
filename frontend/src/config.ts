// Frontend API configuration — reads from Vite env vars so the same build
// works locally (pointing to localhost) and on Vercel (pointing to your tunnel URL).
export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";
export const WS_BASE = (import.meta.env.VITE_WS_BASE as string | undefined)?.replace(/\/$/, "") ?? "ws://127.0.0.1:8000";

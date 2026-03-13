// Replaced Supabase with custom WebSocket implementation

// Single session WebSocket — both roles connect to the same server for sync
const SESSION_WS = import.meta.env.VITE_SESSION_WS_URL || import.meta.env.VITE_WS_URL || '';

type Role = 'director' | 'matcher';

class WSChannel {
  ws: WebSocket | null = null;
  sessionId: string;
  url: string;
  handlers: Array<{ event: string; cb: (arg: any) => void }> = [];
  queue: string[] = [];
  isConnected = false;

  constructor(sessionId: string, url: string) {
    this.sessionId = sessionId;
    this.url = url;
    this.connect();
  }

  connect() {
    console.log(`[WS] Connecting to ${this.url} for session ${this.sessionId}`);
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('[WS] Connected');
      this.isConnected = true;
      // Join Session
      this.ws?.send(JSON.stringify({ type: 'join', session: this.sessionId }));
      // Flush queue
      while (this.queue.length > 0) {
        this.ws?.send(this.queue.shift()!);
      }
    };

    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        // Dispatch to handlers
        // msg format from backend: { event: '...', payload: ... }
        this.handlers.forEach(h => {
          if (h.event === msg.event) {
            h.cb({ payload: msg.payload });
          }
        });
      } catch (e) {
        console.error('[WS] Parse error', e);
      }
    };

    this.ws.onclose = () => {
      console.log('[WS] Disconnected. Reconnecting in 3s...');
      this.isConnected = false;
      setTimeout(() => this.connect(), 3000);
    };
  }

  on(_type: 'broadcast', filter: { event: string }, cb: (arg: any) => void) {
    this.handlers.push({ event: filter.event, cb });
    return this;
  }

  send(msg: { type: 'broadcast'; event: string; payload: any }) {
    const raw = JSON.stringify(msg);
    if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(raw);
    } else {
      this.queue.push(raw);
    }
  }

  subscribe(cb?: (status: string) => void) {
    // Mock subscribe for compatibility
    cb && cb('SUBSCRIBED');
    return this;
  }
}

// Singleton map to reuse channels per session + URL
const channels = new Map<string, WSChannel>();

export function joinSession(sessionId: string, _role?: Role): WSChannel {
  if (!SESSION_WS) throw new Error('VITE_SESSION_WS_URL is missing');
  const key = `${SESSION_WS}::${sessionId}`;
  if (channels.has(key)) return channels.get(key)!;
  const channel = new WSChannel(sessionId, SESSION_WS);
  channels.set(key, channel);
  return channel;
}

export async function signalStart(channel: any, startAt: number, trialIndex: number, mapNumber: number, durationSec?: number) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'start', payload: { startAt, trialIndex, mapNumber, durationSec } });
}

export async function signalTrialEnd(channel: any) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'trial_end', payload: { at: Date.now() } });
}

export async function signalFormSubmitted(channel: any, role: 'director' | 'matcher') {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'forms_submitted', payload: { role, at: Date.now() } });
}

export async function signalTrialPrepare(channel: any, trialIndex: number, mapNumber: number, durationSec?: number) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'trial_prepare', payload: { trialIndex, mapNumber, durationSec, at: Date.now() } });
}

// Uniform sync
export type SyncPhase = 'idle' | 'countdown' | 'running' | 'forms';
export type SyncState = {
  ts: number;
  trialIndex: number;
  mapSet: 1 | 2;
  durationSec: number;
  mapNumber: number;
  startAt: number | null;
  phase: SyncPhase;
};

export async function signalSyncRequest(channel: any) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'sync_request', payload: { at: Date.now() } });
}

export async function signalSyncState(channel: any, state: SyncState) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'sync_state', payload: state });
}


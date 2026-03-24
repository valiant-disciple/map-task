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
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000; // start at 1s, back off

  constructor(sessionId: string, url: string) {
    this.sessionId = sessionId;
    this.url = url;
    this.connect();
  }

  private startPing() {
    this.stopPing();
    // Send a lightweight ping every 25s to keep Render from sleeping the connection
    this.pingTimer = setInterval(() => {
      if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 25000);
  }

  private stopPing() {
    if (this.pingTimer) { clearInterval(this.pingTimer); this.pingTimer = null; }
  }

  connect() {
    // Clear any pending reconnect
    if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }

    console.log(`[WS] Connecting to ${this.url} for session ${this.sessionId}`);
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('[WS] Connected');
      this.isConnected = true;
      this.reconnectDelay = 1000; // reset backoff on success
      // Join Session
      this.ws?.send(JSON.stringify({ type: 'join', session: this.sessionId }));
      // Flush queue
      while (this.queue.length > 0) {
        this.ws?.send(this.queue.shift()!);
      }
      this.startPing();
    };

    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === 'pong') return; // ignore keepalive responses
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
      console.log(`[WS] Disconnected. Reconnecting in ${this.reconnectDelay}ms...`);
      this.isConnected = false;
      this.stopPing();
      this.reconnectTimer = setTimeout(() => this.connect(), this.reconnectDelay);
      // Backoff: 1s → 2s → 4s → cap at 5s
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 5000);
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror, so just log
      console.warn('[WS] Connection error');
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

// ── Clock offset measurement (NTP-style via WS) ──
// Each measurement session uses a unique pingId so that both sides can
// measure simultaneously without cross-contaminating each other's samples.

export async function signalClockPing(channel: any, pingId: string) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'clock_ping', payload: { pingId, t1: Date.now() } });
}

export async function signalClockPong(channel: any, t1: number, pingId: string) {
  if (!channel) return;
  const t2 = Date.now();
  channel.send({ type: 'broadcast', event: 'clock_pong', payload: { pingId, t1, t2, t3: Date.now() } });
}

export type ClockOffsetResult = {
  offsetMs: number;   // peer clock - our clock (positive = peer ahead)
  rttMs: number;
  samples: number;
};

/**
 * Measure clock offset to peer via NTP-style ping/pong.
 * Safe to call from both sides simultaneously — each session uses a
 * unique pingId and only processes matching responses.
 */
export function measureClockOffset(
  channel: any,
  numSamples = 5,
  intervalMs = 300,
  timeoutMs = 8000,
): Promise<ClockOffsetResult> {
  return new Promise((resolve) => {
    const pingId = Math.random().toString(36).slice(2, 10);
    const offsets: number[] = [];
    const rtts: number[] = [];
    let sent = 0;
    let resolved = false;

    const finish = () => {
      if (resolved) return;
      resolved = true;
      clearInterval(timer);
      clearTimeout(timeout);

      if (offsets.length === 0) {
        // No pongs received — peer likely not connected yet
        console.warn('[Sync] Clock offset measurement timed out (0 samples). Using offset=0.');
        resolve({ offsetMs: 0, rttMs: -1, samples: 0 });
        return;
      }

      const sorted = [...offsets].sort((a, b) => a - b);
      const trim = Math.max(1, Math.floor(sorted.length * 0.2));
      const trimmed = sorted.slice(trim, sorted.length - trim);
      const avgOffset = trimmed.length > 0
        ? trimmed.reduce((a, b) => a + b, 0) / trimmed.length
        : offsets.reduce((a, b) => a + b, 0) / offsets.length;
      const avgRtt = rtts.reduce((a, b) => a + b, 0) / rtts.length;

      resolve({
        offsetMs: Math.round(avgOffset * 10) / 10,
        rttMs: Math.round(avgRtt * 10) / 10,
        samples: offsets.length,
      });
    };

    const handler = ({ payload }: any) => {
      if (payload?.pingId !== pingId) return;

      const t4 = Date.now();
      const { t1, t2, t3 } = payload;
      const rtt = (t4 - t1) - (t3 - t2);
      const offset = ((t2 - t1) + (t3 - t4)) / 2;
      offsets.push(offset);
      rtts.push(rtt);

      if (offsets.length >= numSamples) finish();
    };

    channel.on('broadcast', { event: 'clock_pong' }, handler);

    const timer = setInterval(() => {
      if (sent >= numSamples) {
        clearInterval(timer);
        return;
      }
      signalClockPing(channel, pingId);
      sent++;
    }, intervalMs);

    // Timeout: resolve with whatever samples we have (or 0)
    const timeout = setTimeout(finish, timeoutMs);
  });
}


// Replaced Supabase with custom WebSocket implementation
import type { EventRecord } from '../types';

// Use standard WebSocket — configurable via VITE_WS_URL env var
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:3000';

class WSChannel {
  ws: WebSocket | null = null;
  sessionId: string;
  handlers: Array<{ event: string; cb: (arg: any) => void }> = [];
  queue: string[] = [];
  isConnected = false;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
    this.connect();
  }

  connect() {
    console.log(`[WS] Connecting to ${WS_URL} for session ${this.sessionId}`);
    this.ws = new WebSocket(WS_URL);

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

// Singleton map to reuse channels per session
const channels = new Map<string, WSChannel>();

export function joinSession(sessionId: string): WSChannel {
  if (channels.has(sessionId)) return channels.get(sessionId)!;
  const channel = new WSChannel(sessionId);
  channels.set(sessionId, channel);
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

export async function signalEvt(channel: any, rec: EventRecord, from: string) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'evt', payload: { rec, from } });
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

export async function signalAudioChunk(
  channel: any,
  payload: {
    trialIndex: number;
    chunkIndex: number;
    totalChunks: number;
    data: string; // base64
    filename: string;
  }
) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'audio_chunk', payload });
}

// HR Integration signals
export async function signalBaselineStart(channel: any) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'baseline_start', payload: { at: Date.now() } });
}

export async function signalBaselineComplete(channel: any, role: 'director' | 'matcher', avgBpm: number) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'baseline_complete', payload: { role, avgBpm, at: Date.now() } });
}

export async function signalHRData(
  channel: any,
  payload: {
    trialIndex: number;
    role: 'director' | 'matcher';
    data: string; // CSV data as base64 or raw string
  }
) {
  if (!channel) return;
  channel.send({ type: 'broadcast', event: 'hr_data', payload });
}
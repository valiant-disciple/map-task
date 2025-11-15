import { createClient, RealtimeChannel } from '@supabase/supabase-js';
import type { EventRecord } from '../types';

const url = import.meta.env.VITE_SUPABASE_URL;
const key = import.meta.env.VITE_SUPABASE_ANON_KEY;

let supabase: ReturnType<typeof createClient> | null = null;
if (url && key) {
  supabase = createClient(url, key);
} else {
  console.warn('[supabase] Realtime disabled: set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY');
}

export const realtimeEnabled = !!supabase;

// Local same-browser fallback (for single-machine testing)
class LocalChannel {
  bc: BroadcastChannel;
  handlers: Array<{ event: string; cb: (arg: any) => void }> = [];
  constructor(name: string) {
    this.bc = new BroadcastChannel(name);
    this.bc.onmessage = (ev) => {
      const { event, payload } = ev.data || {};
      this.handlers.forEach(h => { if (h.event === event) h.cb({ payload }); });
    };
  }
  on(_type: 'broadcast', filter: { event: string }, cb: (arg: any) => void) {
    this.handlers.push({ event: filter.event, cb });
    return this as any;
  }
  async send(msg: { type: 'broadcast'; event: string; payload: any }) {
    this.bc.postMessage({ event: msg.event, payload: msg.payload });
  }
  subscribe(cb?: (status: string) => void) {
    cb && cb('SUBSCRIBED');
    return this as any;
  }
}

export function joinSession(sessionId: string): RealtimeChannel | null {
  if (supabase) {
    const channel = supabase.channel(`session:${sessionId}`, { config: { broadcast: { ack: true } } });
    channel.subscribe((status) => {
      console.info(`[supabase] channel session:${sessionId} status=${status}`);
      return null;
    });
    return channel;
  }
  const ch = new LocalChannel(`session:${sessionId}`);
  ch.subscribe((status) => console.info(`[local-bus] session:${sessionId} status=${status}`));
  return ch as unknown as RealtimeChannel;
}

export async function signalStart(channel: RealtimeChannel | null, startAt: number, trialIndex: number, mapNumber: number) {
  if (!channel) return;
  await (channel as any).send({ type: 'broadcast', event: 'start', payload: { startAt, trialIndex, mapNumber } });
}

export async function signalTrialEnd(channel: RealtimeChannel | null) {
  if (!channel) return;
  await (channel as any).send({ type: 'broadcast', event: 'trial_end', payload: { at: Date.now() } });
}

export async function signalFormSubmitted(channel: RealtimeChannel | null, role: 'director'|'matcher') {
  if (!channel) return;
  await (channel as any).send({ type: 'broadcast', event: 'forms_submitted', payload: { role, at: Date.now() } });
}

export async function signalEvt(channel: RealtimeChannel | null, rec: EventRecord, from: string) {
  if (!channel) return;
  await (channel as any).send({ type: 'broadcast', event: 'evt', payload: { rec, from } });
}

export async function signalTrialPrepare(channel: RealtimeChannel | null, trialIndex: number, mapNumber: number) {
  if (!channel) return;
  await (channel as any).send({ type: 'broadcast', event: 'trial_prepare', payload: { trialIndex, mapNumber, at: Date.now() } });
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

export async function signalSyncRequest(channel: RealtimeChannel | null) {
  if (!channel) return;
  await (channel as any).send({ type: 'broadcast', event: 'sync_request', payload: { at: Date.now() } });
}

export async function signalSyncState(channel: RealtimeChannel | null, state: SyncState) {
  if (!channel) return;
  await (channel as any).send({ type: 'broadcast', event: 'sync_state', payload: state });
}
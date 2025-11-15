export type Role = 'director' | 'matcher';
export type MapSet = 1 | 2;

export interface SessionState {
  sessionId: string | null;
  participantId: string | null;
  role: Role | null;
  trialIndex: number;      // 1..8
  durationSec: number;     // per-trial duration
  mapSet: MapSet;          // 1 or 2
  trialTotal: number;      // 8
  warmupCount: number;     // 2
}

export interface EventRecord {
  t: number;
  type: string;
  role?: Role;
  payload?: any;
}
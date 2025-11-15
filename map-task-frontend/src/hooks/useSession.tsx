import React, { createContext, useContext, useMemo, useState } from 'react';
import { SessionState, Role, MapSet } from '../types';

type Ctx = {
  state: SessionState;
  setSession: (sessionId: string, participantId: string, role: Role) => void;
  setTrial: (trialIndex: number, durationSec: number) => void;
  setMapSet: (mapSet: MapSet) => void;
  clear: () => void;
};

const SessionCtx = createContext<Ctx | null>(null);

const DEFAULTS: SessionState = {
  sessionId: null,
  participantId: null,
  role: null,
  trialIndex: 1,
  durationSec: 180,
  mapSet: 1,
  trialTotal: 8,
  warmupCount: 2,
};

const KEY = 'session';

function readSession(): SessionState {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) || {};
    return { ...DEFAULTS, ...parsed };
  } catch {
    return DEFAULTS;
  }
}

function writeSession(s: SessionState) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // ignore
  }
}

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<SessionState>(() => readSession());

  const value = useMemo<Ctx>(() => ({
    state,
    setSession: (sessionId, participantId, role) => {
      setState(prev => {
        const next: SessionState = { ...prev, sessionId, participantId, role, trialIndex: 1 };
        writeSession(next);
        return next;
      });
    },
    setTrial: (trialIndex, durationSec) => {
      setState(prev => {
        const next: SessionState = { ...prev, trialIndex, durationSec };
        writeSession(next);
        return next;
      });
    },
    setMapSet: (mapSet) => {
      setState(prev => {
        const next: SessionState = { ...prev, mapSet };
        writeSession(next);
        return next;
      });
    },
    clear: () => {
      setState(() => {
        writeSession(DEFAULTS);
        return DEFAULTS;
      });
    }
  }), [state]);

  return <SessionCtx.Provider value={value}>{children}</SessionCtx.Provider>;
}

export function useSession() {
  const ctx = useContext(SessionCtx);
  if (!ctx) throw new Error('useSession must be used within SessionProvider');
  return ctx;
}
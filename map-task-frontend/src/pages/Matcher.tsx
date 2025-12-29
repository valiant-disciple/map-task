import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import MapViewer from '../components/MapViewer';
import Toolbar from '../components/Toolbar';
import TLXForm from '../components/TLXForm';
import PSMMForm from '../components/PSMMForm';
import { useSession } from '../hooks/useSession';
import { useEventLog } from '../hooks/useEventLog';
import { joinSession, signalFormSubmitted, signalEvt, signalTrialEnd, signalSyncRequest } from '../services/realtime';
import type { EventRecord } from '../types';

import { getMapSrc } from '../utils/mapAssets';

function rid(len = 8) { const c = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({ length: len }, () => c[Math.floor(Math.random() * c.length)]).join(''); }
function mapNumber(mapSet: 1 | 2, trialIndex: number) { return (mapSet === 1 ? 0 : 8) + (trialIndex - 1); }


export default function Matcher() {
  const loc = useLocation();
  const { state, setTrial, setSession, setMapSet } = useSession();
  const { addRaw } = useEventLog();

  const [showTLX, setShowTLX] = useState(false);
  const [showPSMM, setShowPSMM] = useState(false);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });

  // Timer state
  const [startAt, setStartAt] = useState<number | null>(null);
  const [stoppedRemainSec, setStoppedRemainSec] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());

  const channelRef = useRef<ReturnType<typeof joinSession> | null>(null);
  const endedRef = useRef(false);
  const activeTrialRef = useRef<number>(state.trialIndex);

  useEffect(() => {
    const params = new URLSearchParams(loc.search);
    const sid = params.get('session');
    const setParam = params.get('set');

    if (sid) {
      const pid = state.participantId ?? rid();
      setSession(sid, pid, 'matcher');
    }
    if (setParam) {
      const num = Number(setParam);
      if (num === 1 || num === 2) setMapSet(num as 1 | 2);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.search]);

  const currentMapNum = mapNumber(state.mapSet, activeTrialRef.current);
  const isDataTrial = activeTrialRef.current > state.warmupCount;

  const log = (type: string, payload?: any, role?: 'director' | 'matcher') => {
    const rec: EventRecord = {
      t: Date.now(),
      type,
      role,
      payload: { ...(payload || {}), trialIndex: activeTrialRef.current, mapNumber: currentMapNum }
    };
    addRaw(rec);
    if (state.participantId) signalEvt(channelRef.current, rec, state.participantId);
  };

  function computeRemainSec(): number {
    if (!startAt) return state.durationSec;
    const endAt = startAt + state.durationSec * 1000;
    return Math.max(0, Math.ceil((endAt - now) / 1000));
  }

  const remainSec = stoppedRemainSec !== null ? stoppedRemainSec : computeRemainSec();

  const countdownSec = useMemo(() => {
    if (!startAt) return 0;
    return Math.max(0, Math.ceil((startAt - now) / 1000));
  }, [startAt, now]);

  // --- Handlers ---
  const handleStart = (payload: any) => {
    const ti = Number(payload?.trialIndex);
    if (Number.isFinite(ti)) {
      activeTrialRef.current = ti;
      setTrial(ti, state.durationSec);
    }
    if (payload?.startAt) {
      setStartAt(Number(payload.startAt));
      setStoppedRemainSec(null);
      endedRef.current = false;
    }
  };

  const handleTrialEnd = (payload: any) => {
    if (endedRef.current) return;
    endedRef.current = true;

    // Freeze timer at event timestamp
    const endTs = payload?.at || Date.now();
    let finalRemain = 0;
    if (startAt) {
      const endLimit = startAt + state.durationSec * 1000;
      finalRemain = Math.max(0, Math.ceil((endLimit - endTs) / 1000));
    } else {
      finalRemain = state.durationSec;
    }

    setStoppedRemainSec(finalRemain);

    if (isDataTrial) {
      setShowTLX(true);
    }
  };

  const handlePrepare = (payload: any) => {
    const ti = Number(payload?.trialIndex);
    if (ti) {
      // Reset for new trial
      activeTrialRef.current = ti;
      setTrial(ti, state.durationSec);

      setStartAt(null);
      setStoppedRemainSec(null);
      endedRef.current = false;
      setShowTLX(false);
      setShowPSMM(false);
    }
  };

  useEffect(() => {
    setTrial(activeTrialRef.current, state.durationSec);

    if (state.sessionId && state.participantId) {
      channelRef.current = joinSession(state.sessionId);

      channelRef.current?.on('broadcast', { event: 'start' }, ({ payload }) => handleStart(payload));
      channelRef.current?.on('broadcast', { event: 'trial_end' }, ({ payload }) => handleTrialEnd(payload));
      channelRef.current?.on('broadcast', { event: 'trial_prepare' }, ({ payload }) => handlePrepare(payload));

      channelRef.current?.on('broadcast', { event: 'sync_state' }, ({ payload }) => {
        // Sync response
        if (payload) {
          if (payload.trialIndex) {
            activeTrialRef.current = Number(payload.trialIndex);
            setTrial(activeTrialRef.current, state.durationSec);
          }
          if (payload.startAt) {
            setStartAt(Number(payload.startAt));
            setStoppedRemainSec(null);
            endedRef.current = false;
          }
          // Could also sync 'ended' state if we passed it
        }
      });

      channelRef.current?.on('broadcast', { event: 'evt' }, ({ payload }) => {
        if (payload?.from && payload.from !== state.participantId) {
          if (payload?.rec) addRaw(payload.rec as EventRecord);
        }
      });

      // Request full sync on channel join
      signalSyncRequest(channelRef.current);
    }
  }, [state.sessionId, state.participantId, state.mapSet]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  // --- Actions ---

  async function endTrialNow(cause: 'manual' | 'timeout' = 'manual') {
    if (endedRef.current) return;
    endedRef.current = true;
    const remainNow = computeRemainSec();
    setStoppedRemainSec(remainNow);
    log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause }, 'matcher');
    await signalTrialEnd(channelRef.current);
    if (isDataTrial) setShowTLX(true);
  }

  function onTLXSubmit(v: any) { log('tlx_submit', v, 'matcher'); setShowTLX(false); setShowPSMM(true); }
  async function onPSMMSubmit(rows: any[]) {
    log('psmm_submit', rows, 'matcher');
    setShowPSMM(false);
    await signalFormSubmitted(channelRef.current, 'matcher');
  }

  const onToggleMode = () => log('action_toggle_mode', { isErase: false }, 'matcher');

  return (
    <div>
      <Toolbar
        sessionId={state.sessionId || ''}
        role={`matcher (trial ${activeTrialRef.current}/${state.trialTotal ?? 8}${!isDataTrial ? ' warmup' : ''})`}
        remain={remainSec}
        countdownSec={countdownSec}
        isErase={false}
        onToggleMode={onToggleMode}
        onHere={undefined}
        onError={undefined}
        onEnd={() => endTrialNow('manual')}
        showHere={false}
        showError={false}
      />
      <div className="container" style={{ display: 'flex', gap: 20 }}>
        <div style={{ flex: 1 }}>
          <div className="row"><span>Pos: {cursorPos.x}, {cursorPos.y}</span></div>
          <MapViewer
            key={currentMapNum} // Reset canvas state on new map
            src={getMapSrc('matcher', currentMapNum)}
            isInteractive={true}
            onStroke={(stroke) => log('draw_stroke', stroke, 'matcher')}
            onCursorMove={(x, y) => setCursorPos({ x, y })}
          />
        </div>
      </div>
      <TLXForm open={showTLX} onClose={() => setShowTLX(false)} onSubmit={onTLXSubmit} />
      <PSMMForm open={showPSMM} onClose={() => setShowPSMM(false)} onSubmit={onPSMMSubmit} />
    </div>
  );
}

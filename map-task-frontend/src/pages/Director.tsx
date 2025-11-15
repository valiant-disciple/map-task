import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import MapViewer from '../components/MapViewer';
import Toolbar from '../components/Toolbar';
import TLXForm from '../components/TLXForm';
import PSMMForm from '../components/PSMMForm';
import { useSession } from '../hooks/useSession';
import { useEventLog } from '../hooks/useEventLog';
import { joinSession, signalStart, signalTrialEnd, signalFormSubmitted, signalEvt, signalTrialPrepare, signalSyncState } from '../services/realtime';
import type { SyncPhase, SyncState } from '../services/realtime';
import { downloadSessionZip } from '../utils/zip';
import type { EventRecord } from '../types';

function rid(len=8) { const c='ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({length:len},()=>c[Math.floor(Math.random()*c.length)]).join(''); }
function mapNumber(mapSet: 1|2, trialIndex: number) { return (mapSet === 1 ? 0 : 8) + (trialIndex - 1); }
function mapSrc(role: 'director'|'matcher', mapNum: number) {
  const suffix = role === 'director' ? 'g' : 'f';
  return new URL(`../maps/map${mapNum}${suffix}.gif`, import.meta.url).href;
}

export default function Director() {
  const loc = useLocation();
  const { state, setTrial, setSession, setMapSet } = useSession();
  const { events, addRaw } = useEventLog();

  const [showTLX, setShowTLX] = useState(false);
  const [showPSMM, setShowPSMM] = useState(false);
  const [formsDone, setFormsDone] = useState(false);
  const [peerDone, setPeerDone] = useState(false);

  const [startAt, setStartAt] = useState<number | null>(null);
  const [stoppedRemainSec, setStoppedRemainSec] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const channelRef = useRef<ReturnType<typeof joinSession> | null>(null);
  const endedRef = useRef(false);

  // Forward-only active trial (prevents regressions on late events)
  const activeTrialRef = useRef<number>(state.trialIndex);

  useEffect(() => {
    const params = new URLSearchParams(loc.search);
    const sid = params.get('session');
    const setParam = params.get('set');
    const urlStart = params.get('startAt');

    if (sid) {
      const pid = state.participantId ?? rid();
      setSession(sid, pid, 'director');
    }
    if (setParam) {
      const num = Number(setParam);
      if (num === 1 || num === 2) setMapSet(num as 1|2);
    }
    if (urlStart) setStartAt(Number(urlStart));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.search]);

  const currentMapNum = mapNumber(state.mapSet, activeTrialRef.current);
  const isDataTrial = activeTrialRef.current > state.warmupCount;

  const log = (type: string, payload?: any, role?: 'director'|'matcher') => {
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

  function phaseNow(): SyncPhase {
    if (showTLX || showPSMM) return 'forms';
    if (startAt) {
      if (now < startAt) return 'countdown';
      const endAt = startAt + state.durationSec * 1000;
      if (stoppedRemainSec === null && now < endAt) return 'running';
    }
    return 'idle';
  }

  function emitSync(phase: SyncPhase, overrides?: Partial<SyncState>) {
    const tIdx = overrides?.trialIndex ?? activeTrialRef.current;
    const sAt = overrides?.startAt ?? startAt ?? null;
    const sync: SyncState = {
      ts: Date.now(),
      trialIndex: tIdx,
      mapSet: state.mapSet as 1|2,
      durationSec: state.durationSec,
      mapNumber: mapNumber(state.mapSet as 1|2, tIdx),
      startAt: sAt,
      phase
    };
    signalSyncState(channelRef.current, sync);
  }

  useEffect(() => {
    setTrial(activeTrialRef.current, state.durationSec);

    if (state.sessionId && state.participantId) {
      log('session_meta', {
        sessionId: state.sessionId,
        participantId: state.participantId,
        role: 'director',
        trialIndex: activeTrialRef.current,
        durationSec: state.durationSec,
        mapSet: state.mapSet,
        trialTotal: state.trialTotal,
        warmupCount: state.warmupCount,
        maps: { director: `map${currentMapNum}g.gif`, matcher: `map${currentMapNum}f.gif` }
      }, 'director');

      channelRef.current = joinSession(state.sessionId);

      channelRef.current?.on('broadcast', { event: 'start' }, ({ payload }) => {
        const ti = Number(payload?.trialIndex);
        if (Number.isFinite(ti) && ti > activeTrialRef.current) {
          activeTrialRef.current = ti;
          setTrial(ti, state.durationSec);
        }
        if (payload?.startAt) setStartAt(Number(payload.startAt));
        emitSync('running');
      });

      channelRef.current?.on('broadcast', { event: 'trial_end' }, () => {
        if (endedRef.current) return;
        endedRef.current = true;
        const remainNow = computeRemainSec();
        setStoppedRemainSec(remainNow);
        log('trial_final_time', {
          remainSec: remainNow,
          elapsedSec: state.durationSec - remainNow,
          cause: 'broadcast'
        }, 'director');
        if (isDataTrial) {
          setShowTLX(true);
          emitSync('forms');
        } else {
          // warmup: wait for explicit Next Trial (no auto-advance)
        }
      });

      channelRef.current?.on('broadcast', { event: 'forms_submitted' }, ({ payload }) => {
        if (payload?.role === 'matcher') setPeerDone(true);
      });

      channelRef.current?.on('broadcast', { event: 'evt' }, ({ payload }) => {
        if (payload?.from && payload.from === state.participantId) return;
        if (payload?.rec) addRaw(payload.rec as EventRecord);
      });

      emitSync(phaseNow());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.sessionId, state.participantId, state.mapSet]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const countdownSec = useMemo(() => {
    if (!startAt) return 0;
    return Math.max(0, Math.ceil((startAt - now) / 1000));
  }, [startAt, now]);

  const remainSec = stoppedRemainSec !== null ? stoppedRemainSec : computeRemainSec();

  // Timeout path: enter forms (data) or wait (warmup). Never change trials automatically.
  useEffect(() => {
    if (startAt && countdownSec === 0 && remainSec === 0 && !showTLX && !showPSMM && stoppedRemainSec === null) {
      if (endedRef.current) return;
      endedRef.current = true;
      setStoppedRemainSec(0);
      log('trial_final_time', { remainSec: 0, elapsedSec: state.durationSec, cause: 'timeout' }, 'director');
      if (isDataTrial) {
        setShowTLX(true);
        emitSync('forms');
      } else {
        // warmup: wait for explicit Next Trial
      }
    }
  }, [startAt, countdownSec, remainSec, showTLX, showPSMM, stoppedRemainSec, isDataTrial]);

  const urlSid = new URLSearchParams(loc.search).get('session');
  if ((urlSid && (state.sessionId !== urlSid || state.role !== 'director')) || (!state.sessionId && urlSid)) {
    return <div className="container">Joining session…</div>;
  }
  if (!state.sessionId || state.role !== 'director') {
    return <div className="container">Invalid session/role. Go to Lobby.</div>;
  }

  async function startSync() {
    const sAt = Date.now() + 3000;
    setStartAt(sAt);
    await signalStart(channelRef.current, sAt, activeTrialRef.current, currentMapNum);
    emitSync('countdown', { startAt: sAt });
  }

  async function endTrialNow() {
    const alreadyEnded = endedRef.current;
    if (!alreadyEnded) {
      endedRef.current = true;
      const remainNow = computeRemainSec();
      setStoppedRemainSec(remainNow);
      log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause: 'manual' }, 'director');
      await signalTrialEnd(channelRef.current);
    }
    if (isDataTrial) {
      setShowTLX(true);
      emitSync('forms');
    }
  }

  function onTLXSubmit(v: any) { log('tlx_submit', v, 'director'); setShowTLX(false); setShowPSMM(true); }
  async function onPSMMSubmit(rows: any[]) { log('psmm_submit', rows, 'director'); setShowPSMM(false); setFormsDone(true); await signalFormSubmitted(channelRef.current, 'director'); }

  function nextTrial() {
    endedRef.current = false;
    setStartAt(null);
    setStoppedRemainSec(null);
    setShowTLX(false);
    setShowPSMM(false);
    setFormsDone(false);
    setPeerDone(false);

    if (activeTrialRef.current < state.trialTotal) {
      const nextIndex = activeTrialRef.current + 1;
      activeTrialRef.current = nextIndex;
      const nextMap = mapNumber(state.mapSet, nextIndex);
      setTrial(nextIndex, state.durationSec);
      signalTrialPrepare(channelRef.current, nextIndex, nextMap);
      emitSync('idle', { trialIndex: nextIndex, startAt: null });
    }
  }

  async function downloadZip() {
    const finalImageDataUrl = null;
    await downloadSessionZip({ sessionId: state.sessionId!, events, finalImageDataUrl });
  }

  const total = state.trialTotal ?? 8;
  const displayTrial = activeTrialRef.current;

  // NEW strict gating: Director may go next only when both sides submitted forms (for data trials)
  const canNextData = isDataTrial && formsDone && peerDone;

  return (
    <div>
      <Toolbar
        sessionId={state.sessionId}
        role={`director (trial ${displayTrial}/${total}${!isDataTrial ? ' warmup' : ''})`}
        remain={remainSec}
        countdownSec={countdownSec}
        isErase={false}
        onToggleMode={()=>{}}
        onHere={()=>{}}
        onError={()=>{}}
        onEnd={endTrialNow}
      />
      <div className="container">
        <div className="row"><button onClick={startSync}>Start (3s synced)</button></div>
        <MapViewer src={mapSrc('director', currentMapNum)} />
        <div className="row right" style={{ gap: 8 }}>
          {stoppedRemainSec !== null && displayTrial < total && (
            !isDataTrial
              ? <button onClick={nextTrial}>Next Trial</button>
              : <button disabled={!canNextData} onClick={nextTrial}>Next Trial</button>
          )}
          {displayTrial >= total && isDataTrial && (
            <button disabled={!canNextData} onClick={downloadZip}>Download ZIP</button>
          )}
        </div>
      </div>
      <TLXForm open={showTLX} onClose={()=>setShowTLX(false)} onSubmit={onTLXSubmit} />
      <PSMMForm open={showPSMM} onClose={()=>setShowPSMM(false)} onSubmit={onPSMMSubmit} />
    </div>
  );
}
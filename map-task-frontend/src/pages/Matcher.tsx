import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import CanvasBoard, { CanvasBoardHandle } from '../components/CanvasBoard';
import Toolbar from '../components/Toolbar';
import TLXForm from '../components/TLXForm';
import PSMMForm from '../components/PSMMForm';
import { useSession } from '../hooks/useSession';
import { useEventLog } from '../hooks/useEventLog';
import { joinSession, signalFormSubmitted, signalTrialEnd, signalEvt, realtimeEnabled, signalSyncRequest } from '../services/realtime';
import type { SyncState } from '../services/realtime';
import { downloadSessionZip } from '../utils/zip';
import type { EventRecord } from '../types';

function rid(len=8) { const c='ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({length:len},()=>c[Math.floor(Math.random()*c.length)]).join(''); }
function mapNumber(mapSet: 1|2, trialIndex: number) { return (mapSet === 1 ? 0 : 8) + (trialIndex - 1); }
function mapSrc(role: 'director'|'matcher', mapNum: number) {
  const suffix = role === 'director' ? 'g' : 'f';
  return new URL(`../maps/map${mapNum}${suffix}.gif`, import.meta.url).href;
}
const mapSetFromMapNumber = (n: number): 1|2 => (n <= 7 ? 1 : 2);

export default function Matcher() {
  const loc = useLocation();
  const { state, setTrial, setSession, setMapSet } = useSession();
  const { events, addRaw } = useEventLog();

  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const [showTLX, setShowTLX] = useState(false);
  const [showPSMM, setShowPSMM] = useState(false);
  const [formsDone, setFormsDone] = useState(false);
  const [peerDone, setPeerDone] = useState(false);

  const [startAt, setStartAt] = useState<number | null>(null);
  const [stoppedRemainSec, setStoppedRemainSec] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [mode, setMode] = useState<'draw'|'erase'>('draw');

  const boardRef = useRef<CanvasBoardHandle | null>(null);
  const channelRef = useRef<ReturnType<typeof joinSession> | null>(null);
  const endedRef = useRef(false);

  const [activeTrial, setActiveTrial] = useState<number>(state.trialIndex);
  const lastSyncTsRef = useRef(0);

  useEffect(() => {
    const params = new URLSearchParams(loc.search);
    const sid = params.get('session');
    const setParam = params.get('set');
    const urlStart = params.get('startAt');

    if (sid) {
      const pid = state.participantId ?? rid();
      setSession(sid, pid, 'matcher');
    }
    if (setParam) {
      const num = Number(setParam);
      if (num === 1 || num === 2) setMapSet(num as 1|2);
    }
    if (urlStart) setStartAt(Number(urlStart));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.search]);

  const currentMapNum = mapNumber(state.mapSet, activeTrial);

  const log = (type: string, payload?: any, role?: 'director'|'matcher') => {
    const rec: EventRecord = {
      t: Date.now(),
      type,
      role,
      payload: { ...(payload || {}), trialIndex: activeTrial, mapNumber: currentMapNum }
    };
    addRaw(rec);
    if (state.participantId) signalEvt(channelRef.current, rec, state.participantId);
  };

  function resetInteractionState() {
    endedRef.current = false;
    setStartAt(null);
    setStoppedRemainSec(null);
    setShowTLX(false);
    setShowPSMM(false);
    setFormsDone(false);
    setPeerDone(false);
    setMode('draw');
    setPos(null);
    lastPtrRef.current = 0;
  }

  useEffect(() => {
    setTrial(activeTrial, state.durationSec);

    if (state.sessionId && state.participantId) {
      log('session_meta', {
        sessionId: state.sessionId,
        participantId: state.participantId,
        role: 'matcher',
        trialIndex: activeTrial,
        durationSec: state.durationSec,
        mapSet: state.mapSet,
        trialTotal: state.trialTotal,
        warmupCount: state.warmupCount,
        maps: { matcher: `map${currentMapNum}f.gif` }
      }, 'matcher');

      channelRef.current = joinSession(state.sessionId);

      // Ask Director for latest state
      signalSyncRequest(channelRef.current);

      // Apply Director's authoritative sync (ignore older by ts)
      channelRef.current?.on('broadcast', { event: 'sync_state' }, ({ payload }) => {
        const s = payload as SyncState;
        if (!s || typeof s.ts !== 'number') return;
        if (s.ts <= lastSyncTsRef.current) return;
        lastSyncTsRef.current = s.ts;

        const nextMapSet = mapSetFromMapNumber(s.mapNumber);
        if (nextMapSet !== state.mapSet) setMapSet(nextMapSet);

        if (s.trialIndex > activeTrial) {
          resetInteractionState();
          setActiveTrial(s.trialIndex);
          setTrial(s.trialIndex, s.durationSec);
        }
        if (typeof s.startAt === 'number') setStartAt(s.startAt);
        else setStartAt(null);

        if (s.phase === 'forms' && s.trialIndex > state.warmupCount) setShowTLX(true);
      });

      // Ignore stale starts
      channelRef.current?.on('broadcast', { event: 'start' }, ({ payload }) => {
        const ti = Number(payload?.trialIndex);
        const mapNum = Number(payload?.mapNumber);
        if (Number.isFinite(ti) && ti >= activeTrial) {
          resetInteractionState();
          if (Number.isFinite(mapNum)) setMapSet(mapSetFromMapNumber(mapNum));
          setActiveTrial(ti);
          setTrial(ti, state.durationSec);
          if (payload?.startAt) setStartAt(Number(payload.startAt));
        }
      });

      channelRef.current?.on('broadcast', { event: 'trial_end' }, () => {
        if (endedRef.current) return;
        endedRef.current = true;
        const remainNow = computeRemainSec();
        setStoppedRemainSec(remainNow);

        const isDataTrial = activeTrial > state.warmupCount;
        const dataUrl = boardRef.current?.exportPNG();
        if (isDataTrial && dataUrl) log('final_image', { dataUrl }, 'matcher');

        log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause: 'broadcast' }, 'matcher');
        if (isDataTrial) setShowTLX(true);
      });

      channelRef.current?.on('broadcast', { event: 'forms_submitted' }, ({ payload }) => {
        if (payload?.role === 'director') setPeerDone(true);
      });

      channelRef.current?.on('broadcast', { event: 'evt' }, ({ payload }) => {
        if (payload?.from && payload.from === state.participantId) return;
        if (payload?.rec) addRaw(payload.rec as EventRecord);
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.sessionId, state.participantId, state.mapSet, activeTrial]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const countdownSec = useMemo(() => {
    if (!startAt) return 0;
    return Math.max(0, Math.ceil((startAt - now) / 1000));
  }, [startAt, now]);

  function computeRemainSec(): number {
    if (!startAt) return state.durationSec;
    const endAt = startAt + state.durationSec * 1000;
    return Math.max(0, Math.ceil((endAt - now) / 1000));
  }

  const remainSec = stoppedRemainSec !== null ? stoppedRemainSec : computeRemainSec();

  const lastPtrRef = useRef(0);
  function onPosUpdate(p: {x:number;y:number}) {
    setPos(p);
    const t = Date.now();
    if (t - lastPtrRef.current > 50) {
      const isDataTrial = activeTrial > state.warmupCount;
      if (isDataTrial) log('pointer', { x: p.x, y: p.y }, 'matcher');
      lastPtrRef.current = t;
    }
  }

  function openForms() { setShowTLX(true); }
  function onTLXSubmit(v: any) { log('tlx_submit', v, 'matcher'); setShowTLX(false); setShowPSMM(true); }
  async function onPSMMSubmit(rows: any[]) { log('psmm_submit', rows, 'matcher'); setShowPSMM(false); setFormsDone(true); await signalFormSubmitted(channelRef.current, 'matcher'); }

  function toggleMode() {
    const next = mode === 'draw' ? 'erase' : 'draw';
    setMode(next);
    const isDataTrial = activeTrial > state.warmupCount;
    if (isDataTrial) log('mode_change', { mode: next }, 'matcher');
  }

  async function endTrialNow() {
    if (endedRef.current) return;
    endedRef.current = true;
    const remainNow = computeRemainSec();
    setStoppedRemainSec(remainNow);

    const isDataTrial = activeTrial > state.warmupCount;
    const dataUrl = boardRef.current?.exportPNG();
    if (isDataTrial && dataUrl) log('final_image', { dataUrl }, 'matcher');
    log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause: 'manual' }, 'matcher');
    await signalTrialEnd(channelRef.current);
    if (isDataTrial) setShowTLX(true);
  }

  useEffect(() => {
    const isDataTrial = activeTrial > state.warmupCount;
    if (startAt && countdownSec === 0 && remainSec === 0 && !showTLX && !showPSMM && stoppedRemainSec === null) {
      if (endedRef.current) return;
      endedRef.current = true;
      const remainNow = computeRemainSec();
      setStoppedRemainSec(remainNow);
      const dataUrl = boardRef.current?.exportPNG();
      if (isDataTrial && dataUrl) log('final_image', { dataUrl }, 'matcher');
      log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause: 'timeout' }, 'matcher');
      signalTrialEnd(channelRef.current);
      if (isDataTrial) setShowTLX(true);
    }
  }, [startAt, countdownSec, remainSec, showTLX, showPSMM, stoppedRemainSec, activeTrial, state.warmupCount]);

  const urlSid = new URLSearchParams(loc.search).get('session');
  if ((urlSid && (state.sessionId !== urlSid || state.role !== 'matcher')) || (!state.sessionId && urlSid)) {
    return <div className="container">Joining session…</div>;
  }
  if (!state.sessionId || state.role !== 'matcher') {
    return <div className="container">Invalid session/role. Go to Lobby.</div>;
  }

  const total = state.trialTotal ?? 8;
  const isDataTrial = activeTrial > state.warmupCount;
  const isLastDataTrial = isDataTrial && activeTrial >= total;

  async function downloadZip() {
    const finalImageDataUrl = null;
    await downloadSessionZip({ sessionId: state.sessionId!, events, finalImageDataUrl });
  }

  const canDownload = formsDone && (peerDone || !realtimeEnabled);
  const boardKey = `${state.sessionId}-${activeTrial}-${currentMapNum}`;

  return (
    <div>
      <Toolbar
        sessionId={state.sessionId}
        role={`matcher (trial ${activeTrial}/${total}${!isDataTrial ? ' warmup' : ''})`}
        remain={remainSec}
        countdownSec={countdownSec}
        isErase={mode === 'erase'}
        onToggleMode={toggleMode}
        showHere={false}
        showError={false}
        onEnd={endTrialNow}
      />
      <div className="container">
        <div className="row"><span>Pos: {pos ? `x=${pos.x.toFixed(3)} y=${pos.y.toFixed(3)}` : '--'}</span></div>
        <CanvasBoard
          key={boardKey}
          ref={boardRef}
          imgSrc={mapSrc('matcher', currentMapNum)}
          mode={mode}
          onDrawEnd={(poly)=>{ if (isDataTrial) log('draw_end', { polyline: poly, mode }, 'matcher'); }}
          onPos={onPosUpdate}
        />
        <div className="row right" style={{ gap: 8 }}>
          {isLastDataTrial && <button disabled={!canDownload} onClick={downloadZip}>Download ZIP</button>}
        </div>
      </div>
      <TLXForm open={showTLX} onClose={()=>setShowTLX(false)} onSubmit={onTLXSubmit} />
      <PSMMForm open={showPSMM} onClose={()=>setShowPSMM(false)} onSubmit={onPSMMSubmit} />
    </div>
  );
}
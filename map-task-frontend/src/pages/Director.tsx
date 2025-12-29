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
import { audioRecorder } from '../services/audioRecorder';
import type { AudioRecordingResult } from '../services/audioRecorder';
import type { EventRecord } from '../types';

import { getMapSrc } from '../utils/mapAssets';

function rid(len = 8) { const c = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({ length: len }, () => c[Math.floor(Math.random() * c.length)]).join(''); }
function mapNumber(mapSet: 1 | 2, trialIndex: number) { return (mapSet === 1 ? 0 : 8) + (trialIndex - 1); }


export default function Director() {
  const loc = useLocation();
  const { state, setTrial, setSession, setMapSet } = useSession();
  const { events, addRaw } = useEventLog();

  const [showTLX, setShowTLX] = useState(false);
  const [showPSMM, setShowPSMM] = useState(false);
  const [formsDone, setFormsDone] = useState(false);
  const [peerDone, setPeerDone] = useState(false);

  // Timer state
  const [startAt, setStartAt] = useState<number | null>(null);
  const [stoppedRemainSec, setStoppedRemainSec] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedMicId, setSelectedMicId] = useState<string>('');

  const channelRef = useRef<ReturnType<typeof joinSession> | null>(null);
  const endedRef = useRef(false);
  const activeTrialRef = useRef<number>(state.trialIndex);
  const audioFilesRef = useRef<Map<number, { blob: Blob; filename: string }[]>>(new Map());

  // console.log('[Director] Render', { ts: Date.now(), activeTrial: activeTrialRef.current, showTLX, stoppedRemainSec });

  useEffect(() => {
    const params = new URLSearchParams(loc.search);
    const sid = params.get('session');
    const setParam = params.get('set');
    // console.log('[Director] Effect Mount/Update', { sid, setParam, now: Date.now() });

    if (sid) {
      const pid = state.participantId ?? rid();
      setSession(sid, pid, 'director');
    }
    if (setParam) {
      const num = Number(setParam);
      if (num === 1 || num === 2) setMapSet(num as 1 | 2);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loc.search]);

  useEffect(() => {
    audioRecorder.getDevices().then(devs => {
      setDevices(devs);
      if (devs.length > 0) setSelectedMicId(devs[0].deviceId);
    });
  }, []);

  const currentMapNum = mapNumber(state.mapSet, activeTrialRef.current);
  const isDataTrial = activeTrialRef.current > state.warmupCount;

  // Derive strokes from Matcher
  const matcherStrokes = useMemo(() => {
    return events
      .filter(e => e.type === 'draw_stroke' && e.role === 'matcher' && e.payload?.trialIndex === activeTrialRef.current)
      .map(e => e.payload);
  }, [events, activeTrialRef.current]);

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

    // Stop recording
    if (audioRecorder.isRecording()) {
      audioRecorder.stop().then(res => {
        // console.log('[Director] Stopped recording', res);
        const ti = activeTrialRef.current;
        const current = audioFilesRef.current.get(ti) || [];
        current.push({ blob: res.blob, filename: `director_T${ti}.webm` });
        audioFilesRef.current.set(ti, current);
      }).catch(err => console.error('Error stopping recording:', err));
    }

    const endTs = payload?.at || Date.now();
    let finalRemain = 0;
    if (startAt) {
      const endLimit = startAt + state.durationSec * 1000;
      finalRemain = Math.max(0, Math.ceil((endLimit - endTs) / 1000));
    } else {
      finalRemain = state.durationSec;
    }

    setStoppedRemainSec(finalRemain);
    log('trial_final_time', { remainSec: finalRemain, elapsedSec: state.durationSec - finalRemain, cause: 'broadcast' }, 'director');


    // Recompute isDataTrial to avoid stale closure issues
    const currentIsDataTrial = activeTrialRef.current > state.warmupCount;
    if (currentIsDataTrial) {
      setShowTLX(true);
    }
  };

  const handlePrepare = (payload: any) => {
    const ti = Number(payload?.trialIndex);
    if (ti) {
      resetForNewTrial(ti);
    }
  };

  // Respond to sync requests
  const emitSync = () => {
    const s: SyncState = {
      ts: Date.now(),
      trialIndex: activeTrialRef.current,
      mapSet: state.mapSet,
      durationSec: state.durationSec,
      mapNumber: currentMapNum,
      startAt: startAt,
      phase: 'running' // simplified
    };
    signalSyncState(channelRef.current, s);
  };

  useEffect(() => {
    setTrial(activeTrialRef.current, state.durationSec);

    if (state.sessionId && state.participantId) {
      channelRef.current = joinSession(state.sessionId);

      channelRef.current?.on('broadcast', { event: 'start' }, ({ payload }) => handleStart(payload));
      channelRef.current?.on('broadcast', { event: 'trial_end' }, ({ payload }) => {
        handleTrialEnd(payload);
      });
      channelRef.current?.on('broadcast', { event: 'trial_prepare' }, ({ payload }) => handlePrepare(payload));

      channelRef.current?.on('broadcast', { event: 'sync_request' }, () => emitSync());

      channelRef.current?.on('broadcast', { event: 'forms_submitted' }, ({ payload }) => {
        if (payload?.role === 'matcher') setPeerDone(true);
      });

      channelRef.current?.on('broadcast', { event: 'evt' }, ({ payload }) => {
        if (payload?.from && payload.from !== state.participantId) {
          if (payload?.rec) addRaw(payload.rec as EventRecord);
        }
      });
    }
  }, [state.sessionId, state.participantId, state.mapSet]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (startAt && countdownSec === 0 && remainSec === 0 && stoppedRemainSec === null) {
      if (!endedRef.current) {
        endTrialNow('timeout');
      }
    }
  }, [startAt, countdownSec, remainSec, stoppedRemainSec]);

  async function startSync() {
    const sAt = Date.now() + 3000;
    setStartAt(sAt);
    setStoppedRemainSec(null);
    endedRef.current = false;

    // Start recording
    try {
      await audioRecorder.start(selectedMicId);
    } catch (e) {
      console.error('Failed to start recording', e);
      alert('Microphone access failed. Audio will not be recorded.');
    }

    await signalStart(channelRef.current, sAt, activeTrialRef.current, currentMapNum);
    handleStart({ startAt: sAt, trialIndex: activeTrialRef.current, mapNumber: currentMapNum });
  }

  async function endTrialNow(cause: 'manual' | 'timeout' = 'manual') {
    if (endedRef.current) return;
    endedRef.current = true;

    // Stop recording
    if (audioRecorder.isRecording()) {
      audioRecorder.stop().then(res => {
        // console.log('[Director] Stopped recording (manual/timeout)', res);
        const ti = activeTrialRef.current;
        const current = audioFilesRef.current.get(ti) || [];
        current.push({ blob: res.blob, filename: `director_T${ti}.webm` });
        audioFilesRef.current.set(ti, current);
      }).catch(err => console.error('Error stopping recording:', err));
    }

    const remainNow = computeRemainSec();
    setStoppedRemainSec(remainNow);
    log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause }, 'director');
    await signalTrialEnd(channelRef.current);
    if (isDataTrial) setShowTLX(true);
  }

  function resetForNewTrial(nextIndex: number) {
    endedRef.current = false;
    setStartAt(null);
    setStoppedRemainSec(null);
    setShowTLX(false);
    setShowPSMM(false);
    setFormsDone(false);
    setPeerDone(false);
    activeTrialRef.current = nextIndex;
    setTrial(nextIndex, state.durationSec);
  }

  async function nextTrial() {
    if (activeTrialRef.current < (state.trialTotal ?? 8)) {
      const nextIndex = activeTrialRef.current + 1;
      const nextMap = mapNumber(state.mapSet, nextIndex);
      resetForNewTrial(nextIndex);
      await signalTrialPrepare(channelRef.current, nextIndex, nextMap);
    }
  }

  function onTLXSubmit(v: any) { log('tlx_submit', v, 'director'); setShowTLX(false); setShowPSMM(true); }
  async function onPSMMSubmit(rows: any[]) {
    log('psmm_submit', rows, 'director');
    setShowPSMM(false);
    setFormsDone(true);
    await signalFormSubmitted(channelRef.current, 'director');
  }

  async function downloadZip() {
    await downloadSessionZip({
      sessionId: state.sessionId!,
      events,
      finalImageDataUrl: null,
      audioFiles: audioFilesRef.current
    });
  }

  const total = state.trialTotal ?? 8;
  const canNextData = isDataTrial && formsDone && peerDone;

  return (
    <div>
      <Toolbar
        sessionId={state.sessionId || ''}
        role={`director (trial ${activeTrialRef.current}/${total}${!isDataTrial ? ' warmup' : ''})`}
        remain={remainSec}
        countdownSec={countdownSec}
        isErase={false}
        onToggleMode={() => { }}
        onHere={() => { }}
        onError={() => { }}
        onEnd={() => endTrialNow('manual')}
        showHere={true}
        showError={false}
      />
      <div className="container">
        <div className="row">
          <button onClick={startSync}>Start (3s synced)</button>
          <select
            value={selectedMicId}
            onChange={e => setSelectedMicId(e.target.value)}
            style={{ maxWidth: 200, marginLeft: 10 }}
          >
            {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || 'Mic ' + d.deviceId.slice(0, 4)}</option>)}
          </select>
        </div>
        <MapViewer
          src={getMapSrc('director', currentMapNum)}
          isInteractive={false}
        />
        <div className="row right" style={{ gap: 8 }}>
          {stoppedRemainSec !== null && activeTrialRef.current < total && (
            !isDataTrial
              ? <button onClick={nextTrial}>Next Trial</button>
              : <button disabled={!canNextData} onClick={nextTrial}>Next Trial</button>
          )}
          {activeTrialRef.current >= total && isDataTrial && (
            <button disabled={!canNextData} onClick={downloadZip}>Download ZIP</button>
          )}
        </div>
      </div>
      <TLXForm open={showTLX} onClose={() => setShowTLX(false)} onSubmit={onTLXSubmit} />
      <PSMMForm open={showPSMM} onClose={() => setShowPSMM(false)} onSubmit={onPSMMSubmit} />
    </div>
  );
}

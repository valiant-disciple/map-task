import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import MapViewer from '../components/MapViewer';
import Toolbar from '../components/Toolbar';
import TrialSuccessForm from '../components/TrialSuccessForm';
import type { TrialSuccessData } from '../components/TrialSuccessForm';
import TLXForm from '../components/TLXForm';
import PSMMForm from '../components/PSMMForm';
import HRWidget from '../components/HRWidget';
import { useSession } from '../hooks/useSession';
import { useEventLog } from '../hooks/useEventLog';
import { joinSession, signalStart, signalTrialEnd, signalFormSubmitted, signalTrialPrepare, signalSyncState, signalClockPong, measureClockOffset } from '../services/realtime';
import type { SyncState, ClockOffsetResult } from '../services/realtime';
import SyncFlash from '../components/SyncFlash';
import { downloadSessionZip } from '../utils/zip';
import { audioRecorder } from '../services/audioRecorder';
import { watchService, type HRReading } from '../services/watchService';
import MicCheckWidget from '../components/MicCheckWidget';
import type { EventRecord } from '../types';

import { getMapSrc } from '../utils/mapAssets';

function rid(len = 8) { const c = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({ length: len }, () => c[Math.floor(Math.random() * c.length)]).join(''); }
function mapNumberFallback(mapSet: 1 | 2, trialIndex: number) { return (mapSet === 1 ? 0 : 8) + (trialIndex - 1); }


export default function Director() {
  const loc = useLocation();
  const { state, setTrial, setSession, setMapSet } = useSession();
  const { events, addRaw } = useEventLog();

  const [showTrialSuccess, setShowTrialSuccess] = useState(false);
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
  const [micConfirmed, setMicConfirmed] = useState(false);

  const channelRef = useRef<ReturnType<typeof joinSession> | null>(null);
  const endedRef = useRef(false);
  const activeTrialRef = useRef<number>(state.trialIndex);
  const audioFilesRef = useRef<Map<number, { blob: Blob; filename: string }[]>>(new Map());
  // (audio now fetched via HTTP, no more WS chunk reassembly)

  // HR state
  const [baselineDone, setBaselineDone] = useState(false);
  const [baselineHR, setBaselineHR] = useState<number | null>(null);
  const hrDataRef = useRef<Map<number, HRReading[]>>(new Map());

  // Log demographics once on mount
  useEffect(() => {
    watchService.setBaseForRole('director');
  }, []);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('demographics');
      if (raw) {
        const data = JSON.parse(raw);
        addRaw({ t: data.submittedAt || Date.now(), type: 'demographics', role: 'director', payload: data });
        sessionStorage.removeItem('demographics');
      }
    } catch { /* ignore */ }
  }, []);

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

  const refreshDevices = async () => {
    const devs = await audioRecorder.getDevices();
    console.log('[Director] Devices found:', devs.length);
    setDevices(devs);
    if (devs.length > 0 && !selectedMicId) setSelectedMicId(devs[0].deviceId);
    return devs.length;
  };

  useEffect(() => {
    // navigator.mediaDevices is undefined on insecure origins (HTTP + non-localhost IP)
    if (!navigator.mediaDevices) {
      console.warn('[Director] navigator.mediaDevices unavailable — page must be served over HTTPS or localhost');
      return;
    }

    // Initial fetch - may fail if permission not granted yet
    refreshDevices().then(count => {
      // If no devices found, retry after a short delay (user may be granting permission)
      if (count === 0) {
        console.log('[Director] No devices on first try, scheduling retry...');
        const retryTimer = setTimeout(() => {
          refreshDevices();
        }, 2000); // Retry after 2 seconds
        return () => clearTimeout(retryTimer);
      }
    });

    // Listen for device changes (e.g., permission granted, device plugged in)
    const handleDeviceChange = () => {
      console.log('[Director] Device change detected, refreshing...');
      refreshDevices();
    };
    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, []);

  // Use shuffled map order if available, fallback to sequential
  const currentMapNum = state.mapOrder
    ? (state.mapOrder[activeTrialRef.current - 1] ?? mapNumberFallback(state.mapSet, activeTrialRef.current))
    : mapNumberFallback(state.mapSet, activeTrialRef.current);
  const isDataTrial = activeTrialRef.current > state.warmupCount;

  const log = (type: string, payload?: any, role?: 'director' | 'matcher') => {
    const rec: EventRecord = {
      t: Date.now(),
      type,
      role,
      payload: { ...(payload || {}), trialIndex: activeTrialRef.current, mapNumber: currentMapNum }
    };
    addRaw(rec);
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

    // Set HR phase to idle and save HR data (director only)
    watchService.setPhase('idle');
    const ti = activeTrialRef.current;
    const trialHR = watchService.getReadingsForPhase('trial');
    const existing = hrDataRef.current.get(ti) || [];
    hrDataRef.current.set(ti, [...existing, ...trialHR]);

    // Stop recording
    if (audioRecorder.isRecording()) {
      audioRecorder.stop().then(res => {
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
      setShowTrialSuccess(true); // Director: TrialSuccess → TLX → PSMM
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
      channelRef.current = joinSession(state.sessionId, 'director');

      channelRef.current?.on('broadcast', { event: 'start' }, ({ payload }) => handleStart(payload));
      channelRef.current?.on('broadcast', { event: 'trial_end' }, ({ payload }) => {
        handleTrialEnd(payload);
      });
      channelRef.current?.on('broadcast', { event: 'trial_prepare' }, ({ payload }) => handlePrepare(payload));

      channelRef.current?.on('broadcast', { event: 'sync_request' }, () => emitSync());

      channelRef.current?.on('broadcast', { event: 'forms_submitted' }, ({ payload }) => {
        if (payload?.role === 'matcher') setPeerDone(true);
      });

      // Respond to Matcher's clock pings (echo back pingId for disambiguation)
      channelRef.current?.on('broadcast', { event: 'clock_ping' }, ({ payload }) => {
        signalClockPong(channelRef.current, payload.t1, payload.pingId);
      });

      // Measure clock offset (Director initiates)
      measureClockOffset(channelRef.current, 5, 300).then((result) => {
        console.log(`[Sync] Clock offset to Matcher: ${result.offsetMs}ms (RTT: ${result.rttMs}ms, samples: ${result.samples})`);
        addRaw({
          t: Date.now(),
          type: 'clock_offset',
          role: 'director',
          payload: { offsetMs: result.offsetMs, rttMs: result.rttMs, samples: result.samples, peerRole: 'matcher' },
        });
      });

    }
  }, [state.sessionId, state.participantId, state.mapSet]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 50);
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
    if (!baselineDone) {
      alert('Please complete baseline HR measurement first.');
      return;
    }
    if (!micConfirmed) {
      alert('Please check your microphone first.');
      return;
    }

    const sAt = Date.now() + 5000;
    setStartAt(sAt);
    setStoppedRemainSec(null);
    endedRef.current = false;

    // Set HR phase to trial
    watchService.setPhase('trial');

    // Start recording
    try {
      await audioRecorder.start(selectedMicId);
    } catch (e) {
      console.error('Failed to start recording', e);
      alert('Microphone access failed. Audio will not be recorded.');
    }

    await signalStart(channelRef.current, sAt, activeTrialRef.current, currentMapNum, state.durationSec);
    handleStart({ startAt: sAt, trialIndex: activeTrialRef.current, mapNumber: currentMapNum, durationSec: state.durationSec });
  }

  async function endTrialNow(cause: 'manual' | 'timeout' = 'manual') {
    if (endedRef.current) return;
    endedRef.current = true;

    // Set HR phase to idle and save HR data (director only)
    watchService.setPhase('idle');
    const ti = activeTrialRef.current;
    const trialHR = watchService.getReadingsForPhase('trial');
    const existing = hrDataRef.current.get(ti) || [];
    hrDataRef.current.set(ti, [...existing, ...trialHR]);

    // Stop recording
    if (audioRecorder.isRecording()) {
      audioRecorder.stop().then(res => {
        const current = audioFilesRef.current.get(ti) || [];
        current.push({ blob: res.blob, filename: `director_T${ti}.webm` });
        audioFilesRef.current.set(ti, current);
      }).catch(err => console.error('Error stopping recording:', err));
    }

    const remainNow = computeRemainSec();
    setStoppedRemainSec(remainNow);
    log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause }, 'director');
    await signalTrialEnd(channelRef.current);
    if (isDataTrial) setShowTrialSuccess(true); // Director: TrialSuccess → TLX → PSMM
  }

  function resetForNewTrial(nextIndex: number) {
    endedRef.current = false;
    setStartAt(null);
    setStoppedRemainSec(null);
    setShowTrialSuccess(false);
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
      const nextMap = state.mapOrder
        ? (state.mapOrder[nextIndex - 1] ?? mapNumberFallback(state.mapSet, nextIndex))
        : mapNumberFallback(state.mapSet, nextIndex);
      resetForNewTrial(nextIndex);
      await signalTrialPrepare(channelRef.current, nextIndex, nextMap, state.durationSec);
    }
  }

  function onTrialSuccessSubmit(data: TrialSuccessData) {
    log('trial_success', data, 'director');
    setShowTrialSuccess(false);
    setShowTLX(true); // Next: TLX
  }
  function onTLXSubmit(v: any) { log('tlx_submit', v, 'director'); setShowTLX(false); setShowPSMM(true); }
  async function onPSMMSubmit(rows: any[]) {
    log('psmm_submit', rows, 'director');
    setShowPSMM(false);
    setFormsDone(true);
    await signalFormSubmitted(channelRef.current, 'director');
  }

  const onBaselineComplete = React.useCallback((avgBpm: number) => {
    setBaselineHR(avgBpm);
    setBaselineDone(true);
    log('baseline_hr', { avgBpm, role: 'director' }, 'director');
  }, []);

  async function downloadZip() {
    await downloadSessionZip({
      role: 'director',
      sessionId: state.sessionId!,
      events: events.filter(e => !e.role || e.role === 'director'),
      audioFiles: audioFilesRef.current,
      hrData: hrDataRef.current,
      baselineHR: baselineHR,
    });
  }

  const total = state.trialTotal ?? 8;
  const canNextData = isDataTrial && formsDone && peerDone;

  return (
    <div style={{ display: 'flex', gap: 16 }}>
      {/* Sidebar with HR Widget */}
      {/* Sidebar */}
      <div style={{ width: 220, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <HRWidget onBaselineComplete={onBaselineComplete} baselineDuration={20} showSimToggle={true} />
        <MicCheckWidget
          onConfirm={() => setMicConfirmed(true)}
          selectedMicId={selectedMicId}
          devices={devices}
          onSelectMic={setSelectedMicId}
          onRefreshDevices={refreshDevices}
        />
      </div>

      {/* Main Content */}
      <div style={{ flex: 1 }}>
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
          trialRunning={startAt !== null && stoppedRemainSec === null && countdownSec === 0}
          trialEnded={stoppedRemainSec !== null}
        />
        <div className="container">
          {/* Start button — only show before trial is running */}
          {startAt === null && stoppedRemainSec === null && (
          <div className="row">
            <button
              onClick={startSync}
              disabled={!baselineDone || !micConfirmed}
              style={{
                backgroundColor: (!baselineDone || !micConfirmed) ? '#ccc' : '#4CAF50',
                  color: '#fff',
                cursor: (!baselineDone || !micConfirmed) ? 'not-allowed' : 'pointer'
              }}
            >
              {baselineDone && micConfirmed ? 'Start (5s synced)' : !baselineDone ? 'Complete Baseline First' : 'Complete Mic Check First'}
            </button>
          </div>
          )}
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
            {activeTrialRef.current >= total && isDataTrial && stoppedRemainSec !== null && canNextData && (
              <button onClick={downloadZip}>
                Download ZIP
              </button>
            )}
          </div>
        </div>
      </div>
      <SyncFlash
        startAt={startAt}
        onFlash={(ts) => log('sync_flash', { flashTs: ts, trialIndex: activeTrialRef.current }, 'director')}
      />
      <TrialSuccessForm open={showTrialSuccess} onClose={() => setShowTrialSuccess(false)} onSubmit={onTrialSuccessSubmit} trialIndex={activeTrialRef.current} />
      <TLXForm open={showTLX} onClose={() => setShowTLX(false)} onSubmit={onTLXSubmit} />
      <PSMMForm open={showPSMM} onClose={() => setShowPSMM(false)} onSubmit={onPSMMSubmit} />
    </div >
  );
}

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import MapViewer from '../components/MapViewer';
import Toolbar from '../components/Toolbar';
import TLXForm from '../components/TLXForm';
import PSMMForm from '../components/PSMMForm';
import HRWidget from '../components/HRWidget';
import { useSession } from '../hooks/useSession';
import { useEventLog } from '../hooks/useEventLog';
import { joinSession, signalFormSubmitted, signalEvt, signalTrialEnd, signalSyncRequest, signalAudioChunk, signalHRData, signalBaselineComplete } from '../services/realtime';
import { audioRecorder } from '../services/audioRecorder';
import { watchService, type HRReading } from '../services/watchService';
import MicCheckWidget from '../components/MicCheckWidget';
import type { EventRecord } from '../types';

import { getMapSrc } from '../utils/mapAssets';

function rid(len = 8) { const c = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({ length: len }, () => c[Math.floor(Math.random() * c.length)]).join(''); }
function mapNumberFallback(mapSet: 1 | 2, trialIndex: number) { return (mapSet === 1 ? 0 : 8) + (trialIndex - 1); }


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

  // Audio state
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedMicId, setSelectedMicId] = useState<string>('');

  // HR state
  const [baselineDone, setBaselineDone] = useState(false);
  const [baselineHR, setBaselineHR] = useState<number | null>(null);
  const [waitingForBaseline, setWaitingForBaseline] = useState(false);
  const [micConfirmed, setMicConfirmed] = useState(false);

  // Load devices
  const refreshDevices = async () => {
    const devs = await audioRecorder.getDevices();
    console.log('[Matcher] Devices found:', devs.length);
    setDevices(devs);
    if (devs.length > 0 && !selectedMicId) setSelectedMicId(devs[0].deviceId);
    return devs.length;
  };

  useEffect(() => {
    // navigator.mediaDevices is undefined on insecure origins (HTTP + non-localhost IP)
    if (!navigator.mediaDevices) {
      console.warn('[Matcher] navigator.mediaDevices unavailable — page must be served over HTTPS or localhost');
      return;
    }

    // Initial fetch - may fail if permission not granted yet
    refreshDevices().then(count => {
      // If no devices found, retry after a short delay (user may be granting permission)
      if (count === 0) {
        console.log('[Matcher] No devices on first try, scheduling retry...');
        const retryTimer = setTimeout(() => {
          refreshDevices();
        }, 2000); // Retry after 2 seconds
        return () => clearTimeout(retryTimer);
      }
    });

    // Listen for device changes (e.g., permission granted, device plugged in)
    const handleDeviceChange = () => {
      console.log('[Matcher] Device change detected, refreshing...');
      refreshDevices();
    };
    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, []);

  // Log demographics once on mount
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('demographics');
      if (raw) {
        const data = JSON.parse(raw);
        addRaw({ t: data.submittedAt || Date.now(), type: 'demographics', role: 'matcher', payload: data });
        sessionStorage.removeItem('demographics');
      }
    } catch { /* ignore */ }
  }, []);

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

  const [isErase, setIsErase] = useState(false);

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

    // Start Recording (Matcher)
    if (payload?.startAt) {
      audioRecorder.start(selectedMicId).catch(err => console.error('[Matcher] Rec Start Error', err));
      // Set HR phase to trial
      watchService.setPhase('trial');
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

    // Set HR phase to idle and send HR data to Director
    watchService.setPhase('idle');
    const ti = activeTrialRef.current;
    const trialHR = watchService.getReadingsForPhase('trial');
    if (trialHR.length > 0) {
      // Send as CSV string
      const csvData = trialHR.map(r => `${r.t},${r.bpm},${r.phase}`).join('|');
      signalHRData(channelRef.current, {
        trialIndex: ti,
        role: 'matcher',
        data: csvData
      });
      console.log(`[Matcher] Sent HR data for trial ${ti}: ${trialHR.length} readings`);
    }

    // Stop Recording & Send to Director
    if (audioRecorder.isRecording()) {
      audioRecorder.stop().then(async (res) => {
        console.log('[Matcher] Audio Recorded, sending to Director...', res.blob.size);
        const reader = new FileReader();
        reader.readAsDataURL(res.blob);
        reader.onloadend = async () => {
          const base64data = (reader.result as string).split(',')[1];
          const CHUNK_SIZE = 100 * 1024; // 100KB
          const totalChunks = Math.ceil(base64data.length / CHUNK_SIZE);
          const trialIndex = activeTrialRef.current;

          for (let i = 0; i < totalChunks; i++) {
            const chunk = base64data.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
            await signalAudioChunk(channelRef.current, {
              trialIndex,
              chunkIndex: i,
              totalChunks,
              data: chunk,
              filename: `matcher_T${trialIndex}.webm`
            });
            // small delay to prevent overflowing websocket buffer
            await new Promise(r => setTimeout(r, 50));
          }
          console.log('[Matcher] Sent all audio chunks');
        };
      }).catch(err => console.error('[Matcher] Rec Stop Error', err));
    }

    // FIX: Recompute isDataTrial to avoid stale closure
    const currentIsDataTrial = activeTrialRef.current > state.warmupCount;
    if (currentIsDataTrial) {
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

      // Listen for baseline start signal from Director
      channelRef.current?.on('broadcast', { event: 'baseline_start' }, () => {
        console.log('[Matcher] Received baseline_start signal');
        setWaitingForBaseline(true);
        // Auto-start baseline measurement if connected
        // Widget will handle the actual measurement
      });
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

  const onToggleMode = () => {
    const next = !isErase;
    setIsErase(next);
    log('action_toggle_mode', { isErase: next }, 'matcher');
  };

  const onBaselineComplete = React.useCallback((avgBpm: number) => {
    setBaselineHR(avgBpm);
    setBaselineDone(true);
    setWaitingForBaseline(false);
    log('baseline_hr', { avgBpm, role: 'matcher' }, 'matcher');
    signalBaselineComplete(channelRef.current, 'matcher', avgBpm);
  }, []);

  return (
    <div style={{ display: 'flex', gap: 16 }}>
      {/* Sidebar with HR Widget */}
      <div style={{ width: 220, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <HRWidget
          onBaselineComplete={onBaselineComplete}
          baselineDuration={20}
          showSimToggle={true}
        />
        <MicCheckWidget
          onConfirm={() => setMicConfirmed(true)}
          selectedMicId={selectedMicId}
          devices={devices}
          onSelectMic={setSelectedMicId}
          onRefreshDevices={refreshDevices}
        />
        {waitingForBaseline && !baselineDone && (
          <div style={{ marginTop: 8, padding: 8, backgroundColor: '#fff3e0', borderRadius: 4, fontSize: 12 }}>
            ⏳ Director started baseline. Please measure yours.
          </div>
        )}
      </div>

      {/* Main Content */}
      <div style={{ flex: 1 }}>
        <Toolbar
          sessionId={state.sessionId || ''}
          role={`matcher (trial ${activeTrialRef.current}/${state.trialTotal ?? 8}${!isDataTrial ? ' warmup' : ''})`}
          remain={remainSec}
          countdownSec={countdownSec}
          isErase={isErase}
          onToggleMode={onToggleMode}
          onHere={undefined}
          onError={undefined}
          onEnd={() => endTrialNow('manual')}
          showHere={false}
          showError={false}
        />
        <div className="container" style={{ display: 'flex', gap: 20 }}>
          <div style={{ flex: 1 }}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <span>Pos: {cursorPos.x}, {cursorPos.y}</span>
            </div>
            <MapViewer
              key={currentMapNum} // Reset canvas state on new map
              src={getMapSrc('matcher', currentMapNum)}
              isInteractive={true}
              isErase={isErase}
              onStroke={(stroke) => log('draw_stroke', stroke, 'matcher')}
              onCursorMove={(x, y) => setCursorPos({ x, y })}
            />
          </div>
        </div>
      </div>
      <TLXForm open={showTLX} onClose={() => setShowTLX(false)} onSubmit={onTLXSubmit} />
      <PSMMForm open={showPSMM} onClose={() => setShowPSMM(false)} onSubmit={onPSMMSubmit} />
    </div>
  );
}

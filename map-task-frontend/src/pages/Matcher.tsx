import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import MapViewer from '../components/MapViewer';
import Toolbar from '../components/Toolbar';
import TLXForm from '../components/TLXForm';
import PSMMForm from '../components/PSMMForm';
import HRWidget from '../components/HRWidget';
import { useSession } from '../hooks/useSession';
import { useEventLog } from '../hooks/useEventLog';
import { joinSession, signalFormSubmitted, signalTrialEnd, signalSyncRequest, signalClockPong, measureClockOffset } from '../services/realtime';
import SyncFlash from '../components/SyncFlash';
import { downloadSessionZip } from '../utils/zip';
import { audioRecorder } from '../services/audioRecorder';
import { watchService, type HRReading } from '../services/watchService';
import MicCheckWidget from '../components/MicCheckWidget';
import type { EventRecord } from '../types';

import { getMapSrc } from '../utils/mapAssets';

function rid(len = 8) { const c = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({ length: len }, () => c[Math.floor(Math.random() * c.length)]).join(''); }
function mapNumberFallback(mapSet: 1 | 2, trialIndex: number) { return (mapSet === 1 ? 0 : 8) + (trialIndex - 1); }


export default function Matcher() {
  const loc = useLocation();
  const { state, setTrial, setSession, setMapSet, setDuration } = useSession();
  const { events, addRaw } = useEventLog();

  const [showTLX, setShowTLX] = useState(false);
  const [showPSMM, setShowPSMM] = useState(false);
  const [formsDone, setFormsDone] = useState(false);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });

  // Timer state
  const [startAt, setStartAt] = useState<number | null>(null);
  const [stoppedRemainSec, setStoppedRemainSec] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());

  const channelRef = useRef<ReturnType<typeof joinSession> | null>(null);
  const endedRef = useRef(false);
  const activeTrialRef = useRef<number>(state.trialIndex);

  // Map number synced from Director (overrides local mapOrder)
  const [mapNumberOverride, setMapNumberOverride] = useState<number | null>(null);

  // Audio state
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedMicId, setSelectedMicId] = useState<string>('');
  const audioFilesRef = useRef<Map<number, { blob: Blob; filename: string }[]>>(new Map());

  // HR state
  const [baselineDone, setBaselineDone] = useState(false);
  const [baselineHR, setBaselineHR] = useState<number | null>(null);
  const [micConfirmed, setMicConfirmed] = useState(false);
  const hrDataRef = useRef<Map<number, HRReading[]>>(new Map());

  // Clock offset: Director's clock - Matcher's clock (ms). Positive = Director ahead.
  const clockOffsetRef = useRef<number>(0);

  // Load devices
  const refreshDevices = async () => {
    const devs = await audioRecorder.getDevices();
    console.log('[Matcher] Devices found:', devs.length);
    setDevices(devs);
    if (devs.length > 0 && !selectedMicId) setSelectedMicId(devs[0].deviceId);
    return devs.length;
  };

  useEffect(() => {
    watchService.setBaseForRole('matcher');
  }, []);

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

  // Use Director's map number if synced, otherwise fall back to local order
  const currentMapNum = mapNumberOverride !== null
    ? mapNumberOverride
    : (state.mapOrder
        ? (state.mapOrder[activeTrialRef.current - 1] ?? mapNumberFallback(state.mapSet, activeTrialRef.current))
        : mapNumberFallback(state.mapSet, activeTrialRef.current));
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

  const [isErase, setIsErase] = useState(false);

  // --- Handlers ---
  const handleStart = (payload: any) => {
    const ti = Number(payload?.trialIndex);
    if (Number.isFinite(ti)) {
      activeTrialRef.current = ti;
    }
    // Sync map number from Director
    if (payload?.mapNumber !== undefined) {
      setMapNumberOverride(Number(payload.mapNumber));
      console.log(`[Matcher] Synced mapNumber=${payload.mapNumber} from Director`);
    }
    // Sync duration from Director
    if (payload?.durationSec) {
      setDuration(Number(payload.durationSec));
    }
    if (payload?.startAt) {
      // Adjust Director's timestamp to Matcher's local clock using measured offset
      // offsetMs = Director_clock - Matcher_clock, so local = remote - offset
      const adjusted = Number(payload.startAt) - clockOffsetRef.current;
      setStartAt(adjusted);
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

    stopAndStore();

    const currentIsDataTrial = activeTrialRef.current > state.warmupCount;
    if (currentIsDataTrial) {
      setShowTLX(true);
    }
  };

  const handlePrepare = (payload: any) => {
    const ti = Number(payload?.trialIndex);
    if (ti) {
      activeTrialRef.current = ti;
      if (payload?.durationSec) setDuration(Number(payload.durationSec));
      // Sync map number from Director for the upcoming trial
      if (payload?.mapNumber !== undefined) {
        setMapNumberOverride(Number(payload.mapNumber));
        console.log(`[Matcher] Prepare: synced mapNumber=${payload.mapNumber} from Director`);
      }
      setStartAt(null);
      setStoppedRemainSec(null);
      endedRef.current = false;
      setShowTLX(false);
      setShowPSMM(false);

      // Re-measure clock offset between trials to correct for drift
      (channelRef.current as any)?._remeasureOffset?.();
    }
  };

  useEffect(() => {
    setTrial(activeTrialRef.current, state.durationSec);

    if (state.sessionId && state.participantId) {
      channelRef.current = joinSession(state.sessionId, 'matcher');

      channelRef.current?.on('broadcast', { event: 'start' }, ({ payload }) => handleStart(payload));
      channelRef.current?.on('broadcast', { event: 'trial_end' }, ({ payload }) => handleTrialEnd(payload));
      channelRef.current?.on('broadcast', { event: 'trial_prepare' }, ({ payload }) => handlePrepare(payload));

      channelRef.current?.on('broadcast', { event: 'sync_state' }, ({ payload }) => {
        // Sync response
        if (payload) {
          if (payload.durationSec) {
            setDuration(Number(payload.durationSec));
          }
          if (payload.trialIndex) {
            activeTrialRef.current = Number(payload.trialIndex);
          }
          // Sync map number from Director
          if (payload.mapNumber !== undefined) {
            setMapNumberOverride(Number(payload.mapNumber));
            console.log(`[Matcher] Sync: synced mapNumber=${payload.mapNumber} from Director`);
          }
          if (payload.startAt) {
            const adjusted = Number(payload.startAt) - clockOffsetRef.current;
            setStartAt(adjusted);
            setStoppedRemainSec(null);
            endedRef.current = false;
            // Ensure HR phase is correct when syncing mid-trial
            watchService.setPhase('trial');
          }
        }
      });

      // Respond to Director's clock pings (echo back pingId for disambiguation)
      channelRef.current?.on('broadcast', { event: 'clock_ping' }, ({ payload }) => {
        signalClockPong(channelRef.current, payload.t1, payload.pingId);
      });

      // Measure clock offset to Director (for adjusting startAt).
      // Re-measures between trials to correct for drift.
      const doOffsetMeasurement = () => {
        measureClockOffset(channelRef.current!, 5, 300).then((result) => {
          if (result.samples > 0) {
            clockOffsetRef.current = result.offsetMs;
          }
          console.log(`[Sync] Clock offset to Director: ${result.offsetMs}ms (RTT: ${result.rttMs}ms, samples: ${result.samples})`);
          addRaw({
            t: Date.now(),
            type: 'clock_offset',
            role: 'matcher',
            payload: { offsetMs: result.offsetMs, rttMs: result.rttMs, samples: result.samples, peerRole: 'director' },
          });
        });
      };
      doOffsetMeasurement();
      // Store for re-measurement between trials
      (channelRef.current as any)._remeasureOffset = doOffsetMeasurement;

      // Request full sync on channel join
      signalSyncRequest(channelRef.current);
    }
  }, [state.sessionId, state.participantId, state.mapSet]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 50);
    return () => window.clearInterval(id);
  }, []);

  // Auto-end trial on timeout (independent of Director's trial_end broadcast)
  useEffect(() => {
    if (startAt && countdownSec === 0 && remainSec === 0 && stoppedRemainSec === null) {
      if (!endedRef.current) {
        endTrialNow('timeout');
      }
    }
  }, [startAt, countdownSec, remainSec, stoppedRemainSec]);

  // --- Stop recording, store HR & audio locally ---

  function stopAndStore() {
    watchService.setPhase('idle');
    const ti = activeTrialRef.current;

    // Store HR data locally
    const trialHR = watchService.getReadingsForPhase('trial');
    const existing = hrDataRef.current.get(ti) || [];
    hrDataRef.current.set(ti, [...existing, ...trialHR]);

    // Stop recording & store audio locally
    if (audioRecorder.isRecording()) {
      audioRecorder.stop().then(res => {
        const filename = `matcher_T${ti}.webm`;
        const current = audioFilesRef.current.get(ti) || [];
        current.push({ blob: res.blob, filename });
        audioFilesRef.current.set(ti, current);
      }).catch(err => console.error('[Matcher] Rec Stop Error', err));
    }
  }

  // --- Actions ---

  async function endTrialNow(cause: 'manual' | 'timeout' = 'manual') {
    if (endedRef.current) return;
    endedRef.current = true;
    const remainNow = computeRemainSec();
    setStoppedRemainSec(remainNow);
    log('trial_final_time', { remainSec: remainNow, elapsedSec: state.durationSec - remainNow, cause }, 'matcher');
    stopAndStore();
    await signalTrialEnd(channelRef.current);
    if (isDataTrial) setShowTLX(true);
  }

  function onTLXSubmit(v: any) { log('tlx_submit', v, 'matcher'); setShowTLX(false); setShowPSMM(true); }
  async function onPSMMSubmit(rows: any[]) {
    log('psmm_submit', rows, 'matcher');
    setShowPSMM(false);
    setFormsDone(true);
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
    log('baseline_hr', { avgBpm, role: 'matcher' }, 'matcher');
  }, []);

  const total = state.trialTotal ?? 8;
  const allTrialsDone = activeTrialRef.current >= total && isDataTrial && stoppedRemainSec !== null && formsDone;

  async function downloadZip() {
    await downloadSessionZip({
      role: 'matcher',
      sessionId: state.sessionId!,
      events: events.filter(e => !e.role || e.role === 'matcher'),
      audioFiles: audioFilesRef.current,
      hrData: hrDataRef.current,
      baselineHR: baselineHR,
    });
  }

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
          trialRunning={startAt !== null && stoppedRemainSec === null && countdownSec === 0}
          trialEnded={stoppedRemainSec !== null}
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
              onStroke={(stroke) => {
                const payload = {
                  ...stroke,
                  points: stroke.points,
                  polyline: stroke.points // ensure polyline populated with x/y points
                };
                log('draw_stroke', payload, 'matcher');
              }}
              onCursorMove={(x, y) => setCursorPos({ x, y })}
            />
            {allTrialsDone && (
              <div className="row right" style={{ marginTop: 8 }}>
                <button onClick={downloadZip}>Download ZIP</button>
              </div>
            )}
          </div>
        </div>
      </div>
      <SyncFlash
        startAt={startAt}
        onFlash={(ts) => log('sync_flash', { flashTs: ts, trialIndex: activeTrialRef.current }, 'matcher')}
      />
      <TLXForm open={showTLX} onClose={() => setShowTLX(false)} onSubmit={onTLXSubmit} />
      <PSMMForm open={showPSMM} onClose={() => setShowPSMM(false)} onSubmit={onPSMMSubmit} />
    </div>
  );
}

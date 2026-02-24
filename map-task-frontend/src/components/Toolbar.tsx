import React from 'react';
import { fmtTime } from '../utils/time';

export default function Toolbar({
  sessionId,
  role,
  remain,
  countdownSec,
  isErase,
  onToggleMode,
  onHere,
  onError,
  onEnd,
  showHere = true,
  showError = true,
  trialRunning = false,
  trialEnded = false,
}: {
  sessionId: string;
  role: string;
  remain: number;
  countdownSec?: number | null;
  isErase: boolean;
  onToggleMode: () => void;
  onHere?: () => void;
  onError?: () => void;
  onEnd: () => void;
  showHere?: boolean;
  showError?: boolean;
  trialRunning?: boolean;
  trialEnded?: boolean;
}) {
  const showCountdown = (countdownSec ?? 0) > 0 && (countdownSec ?? 0) <= 3;
  const display = showCountdown ? String(countdownSec) : fmtTime(remain);

  const endDisabled = !trialRunning || trialEnded;

  return (
    <div className="toolbar">
      <span>Session: {sessionId}</span>
      <span>Role: {role}</span>

      <div className={`timer ${showCountdown ? 'prestart' : ''}`}>
        {display}
      </div>

      <div className="spacer" />

      {showHere && onHere && <button onClick={onHere} disabled={!trialRunning || trialEnded}>Here</button>}

      <button onClick={onToggleMode} disabled={!trialRunning || trialEnded}>
        {isErase ? 'Continue' : 'Backtrack'}
      </button>

      {showError && onError && <button onClick={onError} disabled={!trialRunning || trialEnded}>Error</button>}

      <button
        onClick={onEnd}
        disabled={endDisabled}
        style={{
          backgroundColor: endDisabled ? '#ccc' : '#d33',
          color: endDisabled ? '#888' : '#fff',
          cursor: endDisabled ? 'not-allowed' : 'pointer',
        }}
      >
        End Trial
      </button>
    </div>
  );
}

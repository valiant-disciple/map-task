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
}) {
  const showCountdown = (countdownSec ?? 0) > 0 && (countdownSec ?? 0) <= 3;
  const display = showCountdown ? String(countdownSec) : fmtTime(remain);

  return (
    <div className="toolbar">
      <span>Session: {sessionId}</span>
      <span>Role: {role}</span>

      <div className={`timer ${showCountdown ? 'prestart' : ''}`}>
        {display}
      </div>

      <div className="spacer" />

      {showHere && onHere && <button onClick={onHere}>Here</button>}

      <button onClick={onToggleMode}>
        {isErase ? 'Continue' : 'Backtrack'}
      </button>

      {showError && onError && <button onClick={onError}>Error</button>}

      <button onClick={onEnd}>End Trial</button>
    </div>
  );
}
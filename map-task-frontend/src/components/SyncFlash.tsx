import React, { useEffect, useState } from 'react';

/**
 * Full-screen white flash at the exact moment a trial starts (startAt timestamp).
 * Creates a detectable pupil constriction event in eye tracker data for
 * post-hoc clock synchronization between devices.
 *
 * Flash duration: 150ms — long enough for both SmartEye (60Hz = 16ms frames)
 * and Aurora (120Hz = 8ms frames) to capture multiple frames.
 */
export default function SyncFlash({
  startAt,
  onFlash,
}: {
  startAt: number | null;
  onFlash?: (flashTimestamp: number) => void;
}) {
  const [flashing, setFlashing] = useState(false);
  const [fired, setFired] = useState(false);

  useEffect(() => {
    // Reset when startAt changes (new trial)
    setFired(false);
    setFlashing(false);
  }, [startAt]);

  useEffect(() => {
    if (!startAt || fired) return;

    const now = Date.now();
    const delay = startAt - now;

    if (delay < -500) {
      // Too late, startAt already passed by >500ms
      return;
    }

    const fireAt = Math.max(0, delay);

    const startTimer = setTimeout(() => {
      const actualTs = Date.now();
      setFlashing(true);
      setFired(true);
      onFlash?.(actualTs);

      // End flash after 150ms
      setTimeout(() => setFlashing(false), 150);
    }, fireAt);

    return () => clearTimeout(startTimer);
  }, [startAt, fired, onFlash]);

  if (!flashing) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: '#ffffff',
        zIndex: 99999,
        pointerEvents: 'none',
      }}
    />
  );
}

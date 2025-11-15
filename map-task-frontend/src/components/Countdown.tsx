import React, { useEffect, useState } from 'react';

export default function Countdown({ startAt, onDone }: { startAt: number | null; onDone?: () => void }) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(id);
  }, []);

  if (!startAt || Number.isNaN(startAt)) return null;

  const msLeft = Math.max(0, startAt - now);
  const sec = Math.ceil(msLeft / 1000);
  const show = sec > 0 && sec <= 3;

  useEffect(() => {
    if (msLeft === 0 && onDone) onDone();
  }, [msLeft, onDone]);

  if (!show) return null;

  return (
    <div className="modal-backdrop" style={{ background: 'rgba(0,0,0,.35)', zIndex: 9999 }}>
      <div className="modal" style={{ textAlign: 'center', fontSize: '64px', zIndex: 10000 }}>
        {sec}
      </div>
    </div>
  );
}
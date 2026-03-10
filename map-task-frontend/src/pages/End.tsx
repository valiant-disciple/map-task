import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DebriefForm from '../components/DebriefForm';
import type { DebriefData } from '../components/DebriefForm';
import { useEventLog } from '../hooks/useEventLog';
import { useSession } from '../hooks/useSession';
import { downloadSessionZip } from '../utils/zip';

export default function End() {
  const nav = useNavigate();
  const { events, add, clearLog } = useEventLog();
  const { state, clear } = useSession();
  const [showDebrief, setShowDebrief] = useState(true);
  const [debriefDone, setDebriefDone] = useState(false);

  function onDebriefSubmit(data: DebriefData) {
    add('debrief_submit', data, state.role ?? undefined);
    setShowDebrief(false);
    setDebriefDone(true);
  }

  async function onExport() {
    if (!state.sessionId) return;
    const role = state.role ?? 'director';
    await downloadSessionZip({ role, sessionId: state.sessionId, events: events.filter(e => !e.role || e.role === role) });
  }

  return (
    <div className="container">
      <h2>End of Session</h2>
      {debriefDone ? (
        <>
          <p style={{ color: '#4CAF50', fontWeight: 600, marginBottom: 16 }}>
            ✅ Debrief submitted. Thank you for participating!
          </p>
          <div className="row" style={{ gap: 8 }}>
        <button onClick={onExport}>Download ZIP</button>
            <button onClick={() => { clearLog(); clear(); nav('/'); }}>Back to Lobby</button>
      </div>
        </>
      ) : (
        <p style={{ color: '#666', fontSize: 13 }}>
          Please complete the post-experiment debrief to finish.
        </p>
      )}
      <DebriefForm open={showDebrief} onClose={() => setShowDebrief(false)} onSubmit={onDebriefSubmit} />
    </div>
  );
}

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PSMMForm from '../components/PSMMForm';
import { useEventLog } from '../hooks/useEventLog';
import { useSession } from '../hooks/useSession';
import { downloadSessionZip } from '../utils/zip';

export default function End() {
  const nav = useNavigate();
  const { events, add } = useEventLog();
  const { state, clear } = useSession();
  const [open, setOpen] = useState(true);

  async function onExport() {
    if (!state.sessionId) return;
    const finalImageDataUrl = events.slice().reverse().find(e => e.type === 'final_image')?.payload?.dataUrl ?? null;
    await downloadSessionZip({ sessionId: state.sessionId, events, finalImageDataUrl });
  }

  return (
    <div className="container">
      <h2>End of Session</h2>
      <div className="row">
        <button onClick={onExport}>Download ZIP</button>
        <button onClick={()=>{ clear(); nav('/'); }}>Back to Lobby</button>
      </div>
      <PSMMForm open={open} onClose={()=>setOpen(false)} onSubmit={(rows)=>{ add('psmm_submit', rows); setOpen(false); }} />
    </div>
  );
}
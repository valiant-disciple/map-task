import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSession } from '../hooks/useSession';
import type { Role, MapSet } from '../types';

function rid(len=6) { const c='ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({length:len},()=>c[Math.floor(Math.random()*c.length)]).join(''); }

export default function Lobby() {
  const nav = useNavigate();
  const { setSession, setMapSet } = useSession();
  const [sessionId, setSessionId] = useState(rid());
  const [displayName, setDisplayName] = useState('');
  const [role, setRole] = useState<Role>('director');
  const [mapSet, setMapSetLocal] = useState<MapSet>(1);

  function enter() {
    const participantId = rid(8);
    setSession(sessionId, participantId, role);
    setMapSet(mapSet);
    const q = `?session=${encodeURIComponent(sessionId)}&set=${mapSet}`;
    nav(role === 'director' ? `/director${q}` : `/matcher${q}`);
  }

  return (
    <div className="container">
      <h2>Lobby</h2>
      <div className="card">
        <label>Session ID</label><input value={sessionId} onChange={(e)=>setSessionId(e.target.value)} />
        <label>Display Name (optional)</label><input value={displayName} onChange={(e)=>setDisplayName(e.target.value)} />
        <label>Role</label>
        <select value={role} onChange={(e)=>setRole(e.target.value as Role)}>
          <option value="director">Director</option>
          <option value="matcher">Matcher</option>
        </select>
        <label>Map Set</label>
        <select value={mapSet} onChange={(e)=>setMapSetLocal(Number(e.target.value) as MapSet)}>
          <option value={1}>Set 1 (maps 0–7)</option>
          <option value={2}>Set 2 (maps 8–15)</option>
        </select>
        <button onClick={enter}>Enter</button>
      </div>
      <p className="hint">Place maps under src/maps: map0f.gif/map0g.gif … map15f.gif/map15g.gif.</p>
    </div>
  );
}
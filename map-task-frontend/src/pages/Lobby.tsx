import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSession } from '../hooks/useSession';
import DemographicsForm from '../components/DemographicsForm';
import type { Demographics } from '../components/DemographicsForm';
import type { Role, MapSet } from '../types';

function rid(len = 6) { const c = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; return Array.from({ length: len }, () => c[Math.floor(Math.random() * c.length)]).join(''); }

export default function Lobby() {
  const nav = useNavigate();
  const { setSession, setMapSet, setDuration, setMapOrder } = useSession();
  const [sessionId, setSessionId] = useState(rid());
  const [displayName, setDisplayName] = useState('');
  const [role, setRole] = useState<Role>('director');
  const [mapSet, setMapSetLocal] = useState<MapSet>(1);
  const DURATION_SEC = 30; // Fixed internal duration
  const [randomize, setRandomize] = useState(true); // counterbalance map order

  // Two-step: demographics first, then session config
  const [demographics, setDemographics] = useState<Demographics | null>(null);

  function handleDemographics(data: Demographics) {
    setDemographics(data);
  }

  function enter() {
    const participantId = rid(8);
    setSession(sessionId, participantId, role);
    setMapSet(mapSet);
    setDuration(DURATION_SEC);

    // Generate map order: 8 maps per set, first `warmupCount` are warmups
    // Set 1 → maps 0..7, Set 2 → maps 8..15
    const warmupCount = 2;
    const totalMaps = 8;
    const base = mapSet === 1 ? 0 : 8;
    const warmupMaps = Array.from({ length: warmupCount }, (_, i) => base + i);
    const dataMaps = Array.from({ length: totalMaps - warmupCount }, (_, i) => base + warmupCount + i);

    // Shuffle data maps if counterbalancing is enabled (warmup maps stay fixed)
    if (randomize) {
      for (let i = dataMaps.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [dataMaps[i], dataMaps[j]] = [dataMaps[j], dataMaps[i]];
      }
    }

    const order = [...warmupMaps, ...dataMaps];
    setMapOrder(order);

    // Store demographics in sessionStorage so Director/Matcher can log it as their first event
    if (demographics) {
      sessionStorage.setItem('demographics', JSON.stringify({
        ...demographics,
        displayName,
        participantId,
        sessionId,
        role,
        submittedAt: Date.now(),
      }));
    }

    const q = `?session=${encodeURIComponent(sessionId)}&set=${mapSet}`;
    nav(role === 'director' ? `/director${q}` : `/matcher${q}`);
  }

  // Step 1: Demographics
  if (!demographics) {
    return (
      <div className="container" style={{ maxWidth: 600 }}>
        <h2>Welcome — Map Task Experiment</h2>
        <DemographicsForm onSubmit={handleDemographics} />
      </div>
    );
  }

  // Step 2: Session configuration
  return (
    <div className="container">
      <h2>Session Setup</h2>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 12 }}>
        Demographics recorded ✅ — now configure the session and enter.
      </p>
      <div className="card">
        <label>Session ID</label>
        <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} />
        <label>Display Name (optional)</label>
        <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
        <label>Role</label>
        <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
          <option value="director">Director</option>
          <option value="matcher">Matcher</option>
        </select>
        <label>Map Set</label>
        <select value={mapSet} onChange={(e) => setMapSetLocal(Number(e.target.value) as MapSet)}>
          <option value={1}>Set 1 (maps 0–7)</option>
          <option value={2}>Set 2 (maps 8–15)</option>
        </select>
        <p style={{ fontSize: 12, color: '#888', margin: '8px 0 0' }}>Trial duration: {DURATION_SEC}s</p>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, cursor: 'pointer' }}>
          <input type="checkbox" checked={randomize} onChange={(e) => setRandomize(e.target.checked)} />
          Randomize map order (counterbalancing)
        </label>
        <button onClick={enter} style={{ marginTop: 8 }}>Enter</button>
      </div>
    </div>
  );
}

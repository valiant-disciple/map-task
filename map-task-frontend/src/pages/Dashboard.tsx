import React, { useState } from 'react';

type TrialResult = {
  trialIndex: number;
  mapNumber: number;
  mapAccuracy?: { iou: number; precision: number; recall: number; f1: number };
  transcripts?: Record<string, any>;
  hrMetrics?: Record<string, any>;
};

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<TrialResult[]>([]);

  const onUpload = async () => {
    if (!file) {
      setError('Please choose a ZIP file');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const resp = await fetch('/api/process/process-zip', { method: 'POST', body: fd });
      if (!resp.ok) throw new Error(await resp.text());
      const json = await resp.json();
      setResults(json.trials || []);
    } catch (e: any) {
      setError(e?.message || 'Failed to process');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ paddingTop: 20, paddingBottom: 40 }}>
      <h2>Session Processor</h2>
      <p>Upload the session ZIP (from Director) to run ASR, HR mdrqa, and map accuracy.</p>
      <div className="card" style={{ marginBottom: 16 }}>
        <input
          type="file"
          accept=".zip"
          onChange={e => setFile(e.target.files?.[0] || null)}
          style={{ marginBottom: 10 }}
        />
        <button onClick={onUpload} disabled={loading} style={{ minWidth: 160 }}>
          {loading ? 'Processing…' : 'Process ZIP'}
        </button>
        {error && <div style={{ color: '#c00', marginTop: 8 }}>{error}</div>}
      </div>

      {results.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {results.map(r => (
            <div key={r.trialIndex} className="card">
              <div style={{ fontWeight: 700, marginBottom: 6 }}>
                Trial {r.trialIndex} (map {r.mapNumber})
              </div>
              {r.mapAccuracy && (
                <div style={{ fontSize: 13, marginBottom: 6 }}>
                  Map accuracy — IoU: {r.mapAccuracy.iou.toFixed(3)}, F1:{' '}
                  {r.mapAccuracy.f1.toFixed(3)} (P: {r.mapAccuracy.precision.toFixed(3)}, R:{' '}
                  {r.mapAccuracy.recall.toFixed(3)})
                </div>
              )}
              {r.hrMetrics && (
                <div style={{ fontSize: 13, marginBottom: 6 }}>
                  <div>HR mdrqa:</div>
                  <pre style={{ background: '#f7f7f7', padding: 8, overflowX: 'auto' }}>
                    {JSON.stringify(r.hrMetrics, null, 2)}
                  </pre>
                </div>
              )}
              {r.transcripts && (
                <div style={{ fontSize: 13 }}>
                  <div>Transcripts:</div>
                  <pre style={{ background: '#f7f7f7', padding: 8, overflowX: 'auto' }}>
                    {JSON.stringify(r.transcripts, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

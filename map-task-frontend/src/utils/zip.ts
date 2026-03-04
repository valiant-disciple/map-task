import JSZip from 'jszip';

function dataUrlToBlob(dataUrl: string): Blob {
  const [meta, b64] = dataUrl.split(',');
  const mime = /data:(.*);base64/.exec(meta)?.[1] || 'application/octet-stream';
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

// HR Reading type (matches watchService.ts)
interface HRReading {
  t: number;
  bpm: number;
  phase: 'baseline' | 'trial' | 'idle';
}

// Fallback: synthesize a final image from strokes if no final_image event exists.
type StrokeLike = { polyline?: { x: number; y: number }[]; points?: { x: number; y: number }[]; role?: string; mode?: string };

function dist2(a: { x: number; y: number }, b: { x: number; y: number }) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function pathLength(pts: { x: number; y: number }[]): number {
  let len = 0;
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i].x - pts[i - 1].x;
    const dy = pts[i].y - pts[i - 1].y;
    len += Math.hypot(dx, dy);
  }
  return len;
}

function cleanStrokes(raw: StrokeLike[], role: 'matcher' | 'director' | 'any' = 'matcher') {
  if (!Array.isArray(raw)) return [];
  return raw
    .map(s => {
      const pts = s.polyline?.length ? s.polyline : (s.points ?? []);
      const normPts = Array.isArray(pts) ? pts.filter(p => Number.isFinite(p?.x) && Number.isFinite(p?.y)).map(p => ({ x: +p.x, y: +p.y })) : [];
      const mode = s.mode || 'draw';
      return { ...s, mode, points: normPts, polyline: normPts };
    })
    .filter(s => {
      if (role !== 'any' && s.role && s.role !== role) return false;
      if (s.mode !== 'draw' && s.mode !== 'erase') return false;
      if (!s.points || s.points.length < 2) return false;
      // Require meaningful movement to avoid cursor blips
      const endDistOk = dist2(s.points[0], s.points[s.points.length - 1]) > 1;
      const lenOk = pathLength(s.points) > 6; // ~>= 6px total path
      return endDistOk && lenOk;
    });
}

function strokesToDataUrl(strokes: StrokeLike[], size = 1024): string | null {
  const cleaned = cleanStrokes(strokes, 'matcher');
  if (!cleaned.length) return null;
  if (!strokes || strokes.length === 0) return null;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, size, size);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const s of cleaned) {
    const mode = s.mode || 'draw';
    const pts = s.points || s.polyline || [];
    const isErase = mode === 'erase';
    ctx.globalCompositeOperation = isErase ? 'destination-out' : 'source-over';
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = isErase ? 20 : 3;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
    ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';
  return canvas.toDataURL('image/png');
}

function hrReadingsToCSV(readings: HRReading[]): string {
  const header = 'timestamp_unix_ms,timestamp_iso,bpm,phase';
  const rows = readings.map(r => {
    const iso = new Date(r.t).toISOString();
    return `${r.t},${iso},${r.bpm},${r.phase}`;
  });
  return [header, ...rows].join('\n');
}

export async function downloadSessionZip(options: {
  sessionId: string;
  events: any[];
  finalImageDataUrl?: string | null;
  audioFiles?: Map<number, { blob: Blob; filename: string }[]>; // trialIndex -> list of audio files
  hrData?: Map<number, { director: HRReading[]; matcher: HRReading[] }>; // trialIndex -> HR readings
  baselineHR?: { director: number | null; matcher: number | null }; // Baseline averages
}) {
  const { sessionId, events, audioFiles, hrData, baselineHR } = options;

  const metas = events.filter(e => e.type === 'session_meta').map(e => e.payload || {});
  const metaDirector = metas.find((m: any) => m.role === 'director') || {};
  const metaMatcher = metas.find((m: any) => m.role === 'matcher') || {};

  const mapSet = metaDirector.mapSet ?? metaMatcher.mapSet ?? 1;
  const trialTotal = metaDirector.trialTotal ?? metaMatcher.trialTotal ?? 8;
  const warmupCount = metaDirector.warmupCount ?? metaMatcher.warmupCount ?? 2;

  const trialMap = new Map<number, any[]>();
  for (const e of events) {
    const ti = e?.payload?.trialIndex ?? 1;
    if (!trialMap.has(ti)) trialMap.set(ti, []);
    trialMap.get(ti)!.push(e);
  }

  // Export all trials (including warmups) so strokes.json never comes out empty
  const dataTrialIndices = Array.from(trialMap.keys())
    .filter(ti => ti >= 1 && ti <= trialTotal)
    .sort((a, b) => a - b);

  const zip = new JSZip();

  const trialSummaries: any[] = [];
  for (const ti of dataTrialIndices) {
    const tevents = trialMap.get(ti) || [];
    const mapNumber = tevents.find((e: any) => typeof e?.payload?.mapNumber === 'number')?.payload?.mapNumber
      ?? ((mapSet === 1 ? 0 : 8) + (ti - 1));

    const tlxDirector = tevents.filter(e => e.type === 'tlx_submit' && e.role === 'director').map(e => e.payload);
    const tlxMatcher = tevents.filter(e => e.type === 'tlx_submit' && e.role === 'matcher').map(e => e.payload);

    const psmmDirector = tevents.filter(e => e.type === 'psmm_submit' && e.role === 'director').flatMap((e: any) => Array.isArray(e.payload) ? e.payload : [e.payload]);
    const psmmMatcher = tevents.filter(e => e.type === 'psmm_submit' && e.role === 'matcher').flatMap((e: any) => Array.isArray(e.payload) ? e.payload : [e.payload]);

    const modeTimeline = tevents.filter(e => e.type === 'mode_change').map((e: any) => ({ t: e.t, role: e.role, mode: e.payload?.mode || 'draw' }));
    const rawStrokes = tevents
      .filter(e => (e.type === 'draw_stroke' || e.type === 'draw_end') && e.role === 'matcher')
      .map((e: any) => {
        const pts = e.payload?.polyline?.length ? e.payload.polyline : (e.payload?.points ?? []);
        return { t: e.t, role: e.role, mode: e.payload?.mode || 'draw', polyline: pts };
      });
    const strokes = cleanStrokes(rawStrokes, 'matcher');
    const cursor = tevents.filter(e => e.type === 'pointer' && e.payload && typeof e.payload.x === 'number').map((e: any) => ({ t: e.t, role: e.role, x: e.payload.x, y: e.payload.y }));
    const final = tevents.slice().reverse().find((e: any) => e.type === 'final_image')?.payload?.dataUrl
      ?? strokesToDataUrl(strokes, 1024);

    const finalTimes = tevents.filter(e => e.type === 'trial_final_time').map((e: any) => ({
      t: e.t,
      role: e.role,
      remainSec: e.payload?.remainSec ?? null,
      elapsedSec: e.payload?.elapsedSec ?? null,
      cause: e.payload?.cause ?? null
    }));

    const dir = zip.folder(`trials/T${String(ti).padStart(2, '0')}`)!;
    dir.file('events.json', JSON.stringify(tevents, null, 2));
    dir.file('strokes.json', JSON.stringify(strokes, null, 2));
    dir.file('cursor.json', JSON.stringify(cursor, null, 2));
    dir.file('tlx_director.json', JSON.stringify(tlxDirector, null, 2));
    dir.file('tlx_matcher.json', JSON.stringify(tlxMatcher, null, 2));
    dir.file('psmm_director.json', JSON.stringify(psmmDirector, null, 2));
    dir.file('psmm_matcher.json', JSON.stringify(psmmMatcher, null, 2));
    if (final) dir.file('final_image.png', dataUrlToBlob(final), { binary: true });

    // Save audio files for this trial
    if (audioFiles && audioFiles.has(ti)) {
      const audios = audioFiles.get(ti)!;
      const audioDir = dir.folder('audio')!;
      audios.forEach(a => {
        audioDir.file(a.filename, a.blob);
      });
    }

    // Save HR CSV files for this trial
    if (hrData && hrData.has(ti)) {
      const trialHR = hrData.get(ti)!;
      const hrDir = dir.folder('hr')!;
      if (trialHR.director.length > 0) {
        hrDir.file('hr_director.csv', hrReadingsToCSV(trialHR.director));
      }
      if (trialHR.matcher.length > 0) {
        hrDir.file('hr_matcher.csv', hrReadingsToCSV(trialHR.matcher));
      }
    }

    trialSummaries.push({
      trialIndex: ti,
      mapNumber,
      maps: {
        director: `map${mapNumber}g.gif`,
        matcher: `map${mapNumber}f.gif`
      },
      tlx: { director: tlxDirector.length, matcher: tlxMatcher.length },
      psmm: { director: psmmDirector.length, matcher: psmmMatcher.length },
      finalTimes
    });
  }

  const sessionJson = {
    session: { id: metaDirector.sessionId || metaMatcher.sessionId || sessionId, createdAt: events[0]?.t || null },
    participants: [
      { role: 'director', participantId: metaDirector.participantId || null },
      { role: 'matcher', participantId: metaMatcher.participantId || null }
    ],
    config: {
      mapSet,
      trialTotal,
      warmupCount,
      durationSec: metaDirector.durationSec || metaMatcher.durationSec
    },
    trials: trialSummaries
  };

  zip.file('session/session.json', JSON.stringify(sessionJson, null, 2));
  zip.file('session/events.json', JSON.stringify(events, null, 2));

  const blob = await zip.generateAsync({ type: 'blob' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `map_task_session_${sessionId}.zip`; a.click();
  URL.revokeObjectURL(url);
}
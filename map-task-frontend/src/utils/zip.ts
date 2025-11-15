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

export async function downloadSessionZip(options: {
  sessionId: string;
  events: any[];
  finalImageDataUrl?: string | null;
}) {
  const { sessionId, events } = options;

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

  const dataTrialIndices = Array.from(trialMap.keys()).filter(ti => ti > warmupCount && ti <= trialTotal).sort((a,b)=>a-b);

  const zip = new JSZip();

  const trialSummaries: any[] = [];
  for (const ti of dataTrialIndices) {
    const tevents = trialMap.get(ti) || [];
    const mapNumber = tevents.find((e:any)=>typeof e?.payload?.mapNumber === 'number')?.payload?.mapNumber
      ?? ((mapSet === 1 ? 0 : 8) + (ti - 1));

    const tlxDirector = tevents.filter(e => e.type === 'tlx_submit' && e.role === 'director').map(e => e.payload);
    const tlxMatcher  = tevents.filter(e => e.type === 'tlx_submit' && e.role === 'matcher').map(e => e.payload);

    const psmmDirector = tevents.filter(e => e.type === 'psmm_submit' && e.role === 'director').flatMap((e: any) => Array.isArray(e.payload) ? e.payload : [e.payload]);
    const psmmMatcher  = tevents.filter(e => e.type === 'psmm_submit' && e.role === 'matcher').flatMap((e: any) => Array.isArray(e.payload) ? e.payload : [e.payload]);

    const modeTimeline = tevents.filter(e => e.type === 'mode_change').map((e: any) => ({ t: e.t, role: e.role, mode: e.payload?.mode || 'draw' }));
    const strokes = tevents.filter(e => e.type === 'draw_end').map((e: any) => ({ t: e.t, role: e.role, mode: e.payload?.mode || 'draw', polyline: e.payload?.polyline || [] }));
    const cursor = tevents.filter(e => e.type === 'pointer' && e.payload && typeof e.payload.x === 'number').map((e: any) => ({ t: e.t, role: e.role, x: e.payload.x, y: e.payload.y }));
    const final = tevents.slice().reverse().find((e: any) => e.type === 'final_image')?.payload?.dataUrl ?? null;

    const finalTimes = tevents.filter(e => e.type === 'trial_final_time').map((e: any) => ({
      t: e.t,
      role: e.role,
      remainSec: e.payload?.remainSec ?? null,
      elapsedSec: e.payload?.elapsedSec ?? null,
      cause: e.payload?.cause ?? null
    }));

    const dir = zip.folder(`trials/T${String(ti).padStart(2,'0')}`)!;
    dir.file('events.json', JSON.stringify(tevents, null, 2));
    dir.file('strokes.json', JSON.stringify(strokes, null, 2));
    dir.file('cursor.json', JSON.stringify(cursor, null, 2));
    dir.file('tlx_director.json', JSON.stringify(tlxDirector, null, 2));
    dir.file('tlx_matcher.json', JSON.stringify(tlxMatcher, null, 2));
    dir.file('psmm_director.json', JSON.stringify(psmmDirector, null, 2));
    dir.file('psmm_matcher.json', JSON.stringify(psmmMatcher, null, 2));
    if (final) dir.file('final_image.png', dataUrlToBlob(final), { binary: true });

    trialSummaries.push({
      trialIndex: ti,
      mapNumber,
      maps: {
        director: `map${mapNumber}g.gif`,
        matcher:  `map${mapNumber}f.gif`
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
      { role: 'matcher',  participantId: metaMatcher.participantId  || null }
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
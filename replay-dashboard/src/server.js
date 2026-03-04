import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import JSZip from 'jszip';
import sharp from 'sharp';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import os from 'os';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

ffmpeg.setFfmpegPath(ffmpegPath || '');

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 200 * 1024 * 1024 } });

// ---------- Map accuracy utils ----------
function binaryMetrics(gt, pred) {
  const n = gt.length;
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < n; i++) {
    const g = gt[i] > 0;
    const p = pred[i] > 0;
    if (p && g) tp++;
    else if (p && !g) fp++;
    else if (!p && g) fn++;
  }
  const denomIou = tp + fp + fn;
  const iou = denomIou > 0 ? tp / denomIou : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const dice = f1;
  return { iou, precision, recall, f1, dice };
}

function ssim(gt, pred) {
  const n = gt.length;
  let muX = 0, muY = 0;
  for (let i = 0; i < n; i++) { muX += gt[i]; muY += pred[i]; }
  muX /= n; muY /= n;
  let varX = 0, varY = 0, cov = 0;
  for (let i = 0; i < n; i++) {
    const dx = gt[i] - muX;
    const dy = pred[i] - muY;
    varX += dx * dx;
    varY += dy * dy;
    cov += dx * dy;
  }
  varX /= n; varY /= n; cov /= n;
  const C1 = 6.5025, C2 = 58.5225;
  return ((2 * muX * muY + C1) * (2 * cov + C2)) / ((muX * muX + muY * muY + C1) * (varX + varY + C2));
}

function hausdorff(gt, pred, width, height) {
  const fgCoords = (buf) => {
    const coords = [];
    for (let i = 0; i < buf.length; i++) {
      if (buf[i] > 0) {
        const y = Math.floor(i / width);
        const x = i - y * width;
        coords.push([x, y]);
      }
    }
    return coords;
  };
  const A = fgCoords(gt);
  const B = fgCoords(pred);
  if (!A.length || !B.length) return 0;
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const directed = (P, Q) => {
    let maxMin = 0;
    for (const p of P) {
      let min = Infinity;
      for (const q of Q) {
        const d = dist(p, q);
        if (d < min) min = d;
        if (min === 0) break;
      }
      if (min > maxMin) maxMin = min;
    }
    return maxMin;
  };
  return Math.max(directed(A, B), directed(B, A));
}

async function imageToMask(buffer, width, height, threshold = 180) {
  const buf = await sharp(buffer)
    .resize(width, height)
    .greyscale()
    .threshold(threshold)
    .toColourspace('b-w')
    .raw()
    .toBuffer();
  const out = Buffer.alloc(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = 255 - buf[i];
  return out;
}

function svgFromStrokes(strokes, width, height) {
  const paths = strokes
    .filter(s => s.polyline && s.polyline.length > 1)
    .map(s => {
      const pts = s.polyline.map(p => `${p.x},${p.y}`).join(' ');
      return `<polyline points="${pts}" fill="none" stroke="black" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />`;
    })
    .join('\n');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="100%" height="100%" fill="white"/>
    ${paths}
  </svg>`;
}

async function strokesToMask(strokes, width, height) {
  if (!strokes.length) return Buffer.alloc(width * height, 0);
  const svg = svgFromStrokes(strokes, width, height);
  const buf = await sharp(Buffer.from(svg)).raw().toBuffer();
  const mask = Buffer.alloc(width * height);
  for (let i = 0, j = 0; i < buf.length; i += 3, j++) {
    const v = buf[i];
    mask[j] = v > 10 ? 255 : 0;
  }
  return mask;
}

async function computeMapAccuracy({ groundTruth, matcherFinal, strokes = [], threshold = 180, gtStrokes = null, gtDims = null }) {
  let width = 800, height = 800;
  if (groundTruth) {
    const meta = await sharp(groundTruth).metadata();
    width = meta.width || width;
    height = meta.height || height;
  } else if (gtDims && gtDims.width && gtDims.height) {
    width = gtDims.width;
    height = gtDims.height;
  }

  let gtMask;
  if (gtStrokes && gtStrokes.length) {
    gtMask = await strokesToMask(gtStrokes, width, height);
  } else if (groundTruth) {
    gtMask = await imageToMask(groundTruth, width, height, threshold);
  } else {
    return { width, height, iou: 0, precision: 0, recall: 0, f1: 0, dice: 0, ssim: 0, hausdorff: 0 };
  }

  let predMask;
  if (matcherFinal) predMask = await imageToMask(matcherFinal, width, height, threshold);
  else predMask = await strokesToMask(strokes, width, height);
  const metrics = binaryMetrics(gtMask, predMask);
  const ssimScore = ssim(gtMask, predMask);
  const hd = hausdorff(gtMask, predMask, width, height);
  return { width, height, ...metrics, ssim: ssimScore, hausdorff: hd };
}

// ---------- Audio helpers ----------
async function toWav(buffer) {
  const tmpIn = path.join(os.tmpdir(), `audio-in-${Date.now()}.bin`);
  const tmpOut = path.join(os.tmpdir(), `audio-out-${Date.now()}.wav`);
  await fs.promises.writeFile(tmpIn, buffer);
  await new Promise((resolve, reject) => {
    ffmpeg(tmpIn).outputOptions(['-ar 16000', '-ac 1', '-f wav']).save(tmpOut).on('end', resolve).on('error', reject);
  });
  const wav = await fs.promises.readFile(tmpOut);
  await fs.promises.unlink(tmpIn).catch(() => {});
  await fs.promises.unlink(tmpOut).catch(() => {});
  return wav;
}

async function transcribeSmallestAI(wav, apiKey) {
  const endpoint = 'https://api.smallest.ai/waves/v1/pulse/get_text';
  const params = new URLSearchParams({ language: 'en', word_timestamps: 'true' }).toString();
  const resp = await fetch(`${endpoint}?${params}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'audio/wav' },
    body: wav,
    signal: AbortSignal.timeout(120_000)
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`ASR failed ${resp.status}: ${text}`);
  }
  return resp.json();
}

// ---------- HR MDRQA via Python ----------
async function runMdrqa(series, params, scriptPath) {
  return await new Promise(resolve => {
    const py = spawn('python3', [scriptPath]);
    let out = '';
    let err = '';
    py.stdout.on('data', d => out += d.toString());
    py.stderr.on('data', d => err += d.toString());
    py.on('close', () => {
      if (err) { resolve({ error: err.trim(), params }); return; }
      try { resolve({ ...JSON.parse(out), params }); }
      catch (e) { resolve({ error: e?.message || 'mdrqa parse failed', params }); }
    });
    py.stdin.write(JSON.stringify({ series, embedding: params.embedding, delay: params.delay, radius: params.radius }));
    py.stdin.end();
  });
}

// ---------- Helpers ----------
async function parseCsvHr(buf) {
  const text = buf.toString('utf8').trim();
  const lines = text.split('\n').slice(1);
  const rows = [];
  for (const line of lines) {
    if (!line) continue;
    const parts = line.split(',');
    const t = Number(parts[0]);
    const bpm = Number(parts[2]);
    const phase = parts[3] || '';
    if (Number.isFinite(t) && Number.isFinite(bpm)) rows.push({ t, bpm, phase });
  }
  return rows;
}

async function loadGroundTruth(mapNumber) {
  const root = path.resolve(process.cwd(), '..');
  const p1 = path.join(root, `map${mapNumber}g.gif`);
  const p2 = path.join(root, 'map-task-frontend', `map${mapNumber}g.gif`);
  const candidate = fs.existsSync(p1) ? p1 : (fs.existsSync(p2) ? p2 : p1);
  return fs.promises.readFile(candidate);
}

function loadGtStrokes(mapNumber) {
  const root = path.resolve(process.cwd(), '..');
  const p = path.join(root, `gt_${mapNumber}.json`);
  if (!fs.existsSync(p)) return null;
  try {
    const json = JSON.parse(fs.readFileSync(p, 'utf8'));
    const strokes = Array.isArray(json.strokes) ? json.strokes : [];
    const image = json.image || {};
    return { strokes, dims: { width: image.width, height: image.height } };
  } catch { return null; }
}

function toIST(ts) {
  const d = new Date(ts);
  return d.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' });
}
function normalizeTimes(trialEvents) {
  const times = trialEvents.map(e => e.t).filter(Boolean);
  const t0 = times.length ? Math.min(...times) : null;
  const tEnd = times.length ? Math.max(...times) : null;
  return { t0, tEnd, t0Ist: t0 ? toIST(t0) : null, tEndIst: tEnd ? toIST(tEnd) : null };
}
function remapEventsToRel(trialEvents, t0) {
  if (!t0) return trialEvents;
  return trialEvents.map(e => ({ ...e, t_rel: e.t ? e.t - t0 : null }));
}
function resampleHr(rows, t0, stepMs = 250) {
  if (!rows || rows.length === 0 || !t0) return { raw: rows, resampled: [] };
  const sorted = rows.slice().sort((a, b) => a.t - b.t);
  const start = sorted[0].t;
  const end = sorted[sorted.length - 1].t;
  const resampled = [];
  for (let t = start; t <= end; t += stepMs) {
    let i = 0;
    while (i < sorted.length - 1 && sorted[i + 1].t < t) i++;
    const a = sorted[i];
    const b = sorted[Math.min(i + 1, sorted.length - 1)];
    let bpm = a.bpm;
    if (b.t !== a.t && t >= a.t && t <= b.t) {
      const ratio = (t - a.t) / (b.t - a.t);
      bpm = a.bpm + ratio * (b.bpm - a.bpm);
    }
    resampled.push({ t, t_rel: t - t0, bpm, phase: a.phase });
  }
  const raw = sorted.map(r => ({ ...r, t_rel: r.t - t0 }));
  return { raw, resampled };
}

// ---------- Express app ----------
const app = express();
app.use(express.static(path.join(process.cwd(), 'public')));

app.post('/api/process-zip', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'file required' });
    const apiKey = process.env.SMALLEST_API_KEY;
    if (!apiKey) return res.status(500).json({ error: 'SMALLEST_API_KEY missing' });

    const zip = await JSZip.loadAsync(req.file.buffer);
    const eventsEntry = zip.file('session/events.json');
    const events = eventsEntry ? JSON.parse(await eventsEntry.async('string')) : [];

    const trialBundles = new Map();
    for (const ev of events) {
      const ti = ev?.payload?.trialIndex ?? 1;
      if (!trialBundles.has(ti)) {
        trialBundles.set(ti, { mapNumber: ev?.payload?.mapNumber ?? 0, finalImage: null, strokes: [], hr: { director: [], matcher: [] }, audio: [] });
      }
      const tb = trialBundles.get(ti);
      if (!tb.mapNumber && ev?.payload?.mapNumber) tb.mapNumber = ev.payload.mapNumber;
      if ((ev.type === 'draw_stroke' || ev.type === 'draw_end') && ev.payload?.polyline) tb.strokes.push({ polyline: ev.payload.polyline });
    }

    const entries = Object.values(zip.files);
    for (const entry of entries) {
      const name = entry.name;
      if (!name.startsWith('trials/')) continue;
      const match = name.match(/trials\/T(\d+)\//);
      if (!match) continue;
      const ti = Number(match[1]);
      if (!trialBundles.has(ti)) continue;
      const tb = trialBundles.get(ti);

      if (name.endsWith('final_image.png')) {
        tb.finalImage = Buffer.from(await entry.async('arraybuffer'));
      }
      if (name.endsWith('hr/hr_director.csv')) {
        tb.hr.director = await parseCsvHr(Buffer.from(await entry.async('arraybuffer')));
      }
      if (name.endsWith('hr/hr_matcher.csv')) {
        tb.hr.matcher = await parseCsvHr(Buffer.from(await entry.async('arraybuffer')));
      }
      if (name.includes('/audio/') && !entry.dir) {
        const parts = name.split('/');
        const filename = parts[parts.length - 1];
        const buffer = Buffer.from(await entry.async('arraybuffer'));
        const role = filename.toLowerCase().includes('matcher') ? 'matcher' : filename.toLowerCase().includes('director') ? 'director' : 'unknown';
        tb.audio.push({ role, filename, buffer });
      }
    }

    const results = [];
    const outZip = new JSZip();
    const summaryRows = [['trialIndex','mapNumber','iou','f1','dice','precision','recall','ssim','hausdorff']];
    const mdrqaScript = path.join(process.cwd(), 'scripts', 'mdrqa.py');
    const paramSets = [
      { embedding: 1, delay: 1, radius: 0.05 },
      { embedding: 1, delay: 1, radius: 0.1 },
      { embedding: 1, delay: 1, radius: 0.2 },
      { embedding: 2, delay: 1, radius: 0.1 },
    ];

    for (const [ti, tb] of trialBundles.entries()) {
      if (!tb.mapNumber) continue;
      const gtStrokes = loadGtStrokes(tb.mapNumber);
      const gtImg = gtStrokes ? null : await loadGroundTruth(tb.mapNumber);
      const accuracy = await computeMapAccuracy({
        groundTruth: gtImg,
        matcherFinal: tb.finalImage || null,
        strokes: tb.strokes,
        gtStrokes: gtStrokes?.strokes || null,
        gtDims: gtStrokes?.dims || null
      });

      const trialEvents = events.filter(e => (e?.payload?.trialIndex ?? 1) === ti);
      const timeMeta = normalizeTimes(trialEvents);

      // ASR
      const transcripts = {};
      for (const aud of tb.audio) {
        try {
          const wav = await toWav(aud.buffer);
          const json = await transcribeSmallestAI(wav, apiKey);
          transcripts[aud.role || aud.filename] = json;
        } catch (e) {
          transcripts[aud.role || aud.filename] = { error: e?.message || 'asr failed' };
        }
      }

      // HR mdrqa
      const hrMetrics = {};
      if (tb.hr.director.length) {
        hrMetrics.director = [];
        for (const p of paramSets) hrMetrics.director.push(await runMdrqa(tb.hr.director, p, mdrqaScript));
      }
      if (tb.hr.matcher.length) {
        hrMetrics.matcher = [];
        for (const p of paramSets) hrMetrics.matcher.push(await runMdrqa(tb.hr.matcher, p, mdrqaScript));
      }

      // HR resampled & normalized to t_rel
      const hrSeries = {};
      if (tb.hr.director.length) hrSeries.director = resampleHr(tb.hr.director, timeMeta.t0);
      if (tb.hr.matcher.length) hrSeries.matcher = resampleHr(tb.hr.matcher, timeMeta.t0);

      const eventsRel = remapEventsToRel(trialEvents, timeMeta.t0);

      results.push({
        trialIndex: ti,
        mapNumber: tb.mapNumber,
        mapAccuracy: accuracy,
        transcripts,
        hrMetrics,
        hrSeries,
        events: eventsRel,
        time: { start_ts: timeMeta.t0, end_ts: timeMeta.tEnd, start_ist: timeMeta.t0Ist, end_ist: timeMeta.tEndIst }
      });

      // Build CSVs per trial
      const trialDir = outZip.folder(`trials/T${String(ti).padStart(2,'0')}`);
      if (!trialDir) continue;
      const acc = accuracy || {};
      summaryRows.push([ti, tb.mapNumber, acc.iou ?? '', acc.f1 ?? '', acc.dice ?? '', acc.precision ?? '', acc.recall ?? '', acc.ssim ?? '', acc.hausdorff ?? '']);
      trialDir.file('map_metrics.csv',
        ['metric,value',
         `iou,${acc.iou ?? ''}`,
         `f1,${acc.f1 ?? ''}`,
         `dice,${acc.dice ?? ''}`,
         `precision,${acc.precision ?? ''}`,
         `recall,${acc.recall ?? ''}`,
         `ssim,${acc.ssim ?? ''}`,
         `hausdorff,${acc.hausdorff ?? ''}`].join('\n'));

      // HR CSVs
      const hrDir = trialDir.folder('hr');
      const hrRoles = ['director','matcher'];
      hrRoles.forEach(role => {
        const series = hrSeries[role];
        if (!hrDir || !series) return;
        if (series.raw?.length) hrDir.file(`${role}_raw.csv`, ['t_ms,t_rel_ms,bpm,phase', ...series.raw.map(r => `${r.t},${r.t_rel},${r.bpm},${r.phase||''}`)].join('\n'));
        if (series.resampled?.length) hrDir.file(`${role}_resampled.csv`, ['t_ms,t_rel_ms,bpm,phase', ...series.resampled.map(r => `${r.t},${r.t_rel},${r.bpm},${r.phase||''}`)].join('\n'));
      });

      // MDRQA metrics
      const mdDir = trialDir.folder('mdrqa');
      Object.entries(hrMetrics).forEach(([role, arr]) => { if (mdDir) mdDir.file(`${role}.json`, JSON.stringify(arr, null, 2)); });

      // Transcripts
      const trDir = trialDir.folder('transcripts');
      Object.entries(transcripts).forEach(([role, t]) => {
        if (!trDir) return;
        const text = typeof t === 'object' && t?.transcription ? t.transcription : JSON.stringify(t);
        trDir.file(`${role}.txt`, text);
        trDir.file(`${role}.json`, JSON.stringify(t, null, 2));
      });
    }

    outZip.file('summary_map_metrics.csv', summaryRows.map(r => r.join(',')).join('\n'));
    const zipBase64 = await outZip.generateAsync({ type: 'base64' });
    res.json({ trials: results, zipBase64 });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e?.message || 'processing failed' });
  }
});

const port = process.env.PORT || 4100;
app.listen(port, () => {
  console.log(`Replay dashboard server running on http://localhost:${port}`);
});

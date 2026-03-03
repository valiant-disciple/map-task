import { Router } from 'express';
import multer from 'multer';
import JSZip from 'jszip';
import { computeMapAccuracy } from '../utils/mapAccuracy.js';
import { computeMdrqa } from '../utils/mdrqa.js';
import { toWav, transcribeSmallestAI } from '../utils/asr.js';
import sharp from 'sharp';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 200 * 1024 * 1024 } });

const router = Router();

type TrialBundle = {
  mapNumber: number;
  finalImage?: Buffer | null;
  strokes: { polyline: { x: number; y: number }[] }[];
  hr: { director: number[]; matcher: number[] };
  audio: { role: string; filename: string; buffer: Buffer }[];
};

async function parseCsvHr(buf: Buffer): Promise<number[]> {
  const text = buf.toString('utf8').trim();
  const lines = text.split('\n').slice(1); // skip header
  const vals: number[] = [];
  for (const line of lines) {
    const parts = line.split(',');
    const bpm = Number(parts[2]);
    if (Number.isFinite(bpm)) vals.push(bpm);
  }
  return vals;
}

async function loadGroundTruth(mapNumber: number): Promise<Buffer> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const root = path.resolve(__dirname, '..', '..');
  const mapPath = path.join(root, `map${mapNumber}g.gif`);
  const altPath = path.join(root, 'map-task-frontend', `map${mapNumber}g.gif`);
  const candidate = fs.existsSync(mapPath) ? mapPath : altPath;
  return fs.promises.readFile(candidate);
}

router.post('/process-zip', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'file required' });
    const apiKey = process.env.SMALLEST_API_KEY;
    if (!apiKey) return res.status(500).json({ error: 'SMALLEST_API_KEY missing' });

    const zip = await JSZip.loadAsync(req.file.buffer);

    // Load events to get map numbers per trial
    const eventsEntry = zip.file('session/events.json');
    const events = eventsEntry ? JSON.parse(await eventsEntry.async('string')) : [];

    // Group by trial
    const trialBundles = new Map<number, TrialBundle>();
    for (const ev of events) {
      const ti = ev?.payload?.trialIndex ?? 1;
      if (!trialBundles.has(ti)) {
        trialBundles.set(ti, { mapNumber: ev?.payload?.mapNumber ?? 0, finalImage: null, strokes: [], hr: { director: [], matcher: [] }, audio: [] });
      }
      const tb = trialBundles.get(ti)!;
      if (!tb.mapNumber && ev?.payload?.mapNumber) tb.mapNumber = ev.payload.mapNumber;
      if (ev.type === 'draw_end' && ev.payload?.polyline) tb.strokes.push({ polyline: ev.payload.polyline });
    }

    // Scan zip entries for trial assets
    const entries = Object.values(zip.files);
    for (const entry of entries) {
      const name = entry.name;
      if (!name.startsWith('trials/')) continue;
      const match = name.match(/trials\/T(\d+)\//);
      if (!match) continue;
      const ti = Number(match[1]);
      if (!trialBundles.has(ti)) continue;
      const tb = trialBundles.get(ti)!;

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

    const results: any[] = [];

    for (const [ti, tb] of trialBundles.entries()) {
      if (!tb.mapNumber) continue;
      const gt = await loadGroundTruth(tb.mapNumber);
      const accuracy = await computeMapAccuracy({
        groundTruth: gt,
        matcherFinal: tb.finalImage || null,
        strokes: tb.strokes
      });

      // ASR
      const transcripts: Record<string, any> = {};
      for (const aud of tb.audio) {
        try {
          const wav = await toWav(aud.buffer);
          const json = await transcribeSmallestAI(wav, apiKey);
          transcripts[aud.role || aud.filename] = json;
        } catch (e: any) {
          transcripts[aud.role || aud.filename] = { error: e?.message || 'asr failed' };
        }
      }

      // HR mdrqa
      const hrMetrics: Record<string, any> = {};
      const eps = 1; // threshold in bpm
      if (tb.hr.director.length) hrMetrics.director = computeMdrqa(tb.hr.director, eps);
      if (tb.hr.matcher.length) hrMetrics.matcher = computeMdrqa(tb.hr.matcher, eps);

      results.push({
        trialIndex: ti,
        mapNumber: tb.mapNumber,
        mapAccuracy: accuracy,
        transcripts,
        hrMetrics
      });
    }

    res.json({ trials: results });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: e?.message || 'processing failed' });
  }
});

export const processRouter = router;

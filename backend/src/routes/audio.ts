import { Router } from 'express';

const router = Router();

// In-memory store: sessionId -> trialIndex -> { role, filename, buffer }
const audioStore = new Map<string, Map<number, { role: string; filename: string; buffer: Buffer }[]>>();

// POST /api/audio/:sessionId/:trialIndex
// Body: raw binary (audio/webm), headers: x-role, x-filename
router.post('/:sessionId/:trialIndex', (req, res) => {
  const { sessionId, trialIndex: tiStr } = req.params;
  const ti = Number(tiStr);
  const role = req.headers['x-role'] as string || 'unknown';
  const filename = req.headers['x-filename'] as string || `${role}_T${ti}.webm`;

  const chunks: Buffer[] = [];
  req.on('data', (chunk: Buffer) => chunks.push(chunk));
  req.on('end', () => {
    const buffer = Buffer.concat(chunks);

    if (!audioStore.has(sessionId)) audioStore.set(sessionId, new Map());
    const session = audioStore.get(sessionId)!;
    if (!session.has(ti)) session.set(ti, []);
    const trialFiles = session.get(ti)!;

    // Avoid duplicates
    const existing = trialFiles.findIndex(f => f.filename === filename);
    if (existing >= 0) trialFiles[existing] = { role, filename, buffer };
    else trialFiles.push({ role, filename, buffer });

    console.log(`[Audio] Stored ${filename} for session ${sessionId} trial ${ti} (${buffer.length} bytes)`);
    res.json({ ok: true, size: buffer.length });
  });
});

// GET /api/audio/:sessionId/:trialIndex
// Returns list of available audio files for this trial
router.get('/:sessionId/:trialIndex', (req, res) => {
  const { sessionId, trialIndex: tiStr } = req.params;
  const ti = Number(tiStr);
  const session = audioStore.get(sessionId);
  if (!session || !session.has(ti)) return res.json({ files: [] });
  const files = session.get(ti)!.map(f => ({ role: f.role, filename: f.filename, size: f.buffer.length }));
  res.json({ files });
});

// GET /api/audio/:sessionId/:trialIndex/:filename
// Download specific audio file
router.get('/:sessionId/:trialIndex/:filename', (req, res) => {
  const { sessionId, trialIndex: tiStr, filename } = req.params;
  const ti = Number(tiStr);
  const session = audioStore.get(sessionId);
  if (!session || !session.has(ti)) return res.status(404).json({ error: 'NotFound' });
  const file = session.get(ti)!.find(f => f.filename === filename);
  if (!file) return res.status(404).json({ error: 'NotFound' });

  res.setHeader('Content-Type', 'audio/webm');
  res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
  res.send(file.buffer);
});

// GET /api/audio/:sessionId - list all trials with audio
router.get('/:sessionId', (req, res) => {
  const { sessionId } = req.params;
  const session = audioStore.get(sessionId);
  if (!session) return res.json({ trials: {} });
  const trials: Record<number, string[]> = {};
  session.forEach((files, ti) => {
    trials[ti] = files.map(f => f.filename);
  });
  res.json({ trials });
});

export const audioRouter = router;

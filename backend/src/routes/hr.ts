import { Router } from 'express';
import { z } from 'zod';
import { HrPayloadSchema, type HrRecord } from '../types/hr.js';
import { store } from '../services/store.js';
import { auth } from '../middleware/auth.js';
import { SseHub } from '../services/sse.js';

const router = Router();
const sse = new SseHub<HrRecord>();

router.use(auth);

// POST /api/hr
router.post('/', (req, res, next) => {
  try {
    const parsed = HrPayloadSchema.parse(req.body);
    const record: HrRecord = {
      deviceId: parsed.deviceId,
      ts: parsed.ts ?? Date.now(),
      hr: parsed.hr,
      ibi: parsed.ibi ?? [],
      meta: parsed.meta
    };
    store.add(record);
    sse.broadcast('hr', record);
    return res.json({ ok: true });
  } catch (e) {
    return next(e);
  }
});

// GET /api/hr/latest?deviceId=...
router.get('/latest', (req, res) => {
  const deviceId = typeof req.query.deviceId === 'string' ? req.query.deviceId : undefined;
  const latest = store.latest(deviceId);
  if (!latest) return res.status(404).json({ error: 'NotFound' });
  return res.json(latest);
});

// GET /api/hr/history?deviceId=...&limit=100
router.get('/history', (req, res, next) => {
  try {
    const qp = z.object({
      deviceId: z.string().optional(),
      limit: z.coerce.number().int().positive().max(1000).optional()
    }).parse(req.query);

    const rows = store.history(qp.deviceId, qp.limit ?? 100);
    return res.json(rows);
  } catch (e) {
    return next(e);
  }
});

// GET /api/hr/stream (SSE)
router.get('/stream', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  (res as any).flushHeaders?.();

  const id = sse.addClient(res);
  res.write(`event: hello\ndata: ${JSON.stringify({ ok: true })}\n\n`);

  req.on('close', () => sse.removeClient(id));
});

export const hrRouter = router;



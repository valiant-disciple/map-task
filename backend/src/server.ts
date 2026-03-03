import express from 'express';
import cors from 'cors';
import { config } from './config.js';
import { errorHandler } from './middleware/error.js';
import { healthRouter } from './routes/health.js';
import { hrRouter } from './routes/hr.js';
import { audioRouter } from './routes/audio.js';
import { processRouter } from './routes/process.js';

export function createServer() {
  const app = express();

  app.disable('x-powered-by');
  app.use(cors({ origin: config.corsOrigin }));
  app.use(express.json({ limit: '256kb' }));

  app.use('/healthz', healthRouter);
  app.use('/api/hr', hrRouter);
  app.use('/api/audio', audioRouter);
  app.use('/api/process', processRouter);

  app.use(errorHandler);
  return app;
}



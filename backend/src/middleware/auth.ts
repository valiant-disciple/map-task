import type { Request, Response, NextFunction } from 'express';
import { config } from '../config.js';

export function auth(req: Request, res: Response, next: NextFunction) {
  if (!config.authToken) return next();
  const authHeader = req.header('authorization') || '';
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
  if (token && token === config.authToken) return next();
  return res.status(401).json({ error: 'Unauthorized' });
}



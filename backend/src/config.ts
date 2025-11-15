import * as dotenv from 'dotenv';

dotenv.config();

function getEnv(key: string, fallback?: string) {
  const v = process.env[key];
  if (v === undefined || v === '') return fallback;
  return v;
}

export const config = {
  port: Number(getEnv('PORT', '3000')),
  authToken: getEnv('AUTH_TOKEN'),
  corsOrigin: getEnv('CORS_ORIGIN', '*')
};



import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const allowed = [
  'localhost',
  '127.0.0.1',
  'distemperately-nonduplicative-elise.ngrok-free.dev',
];

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: allowed,
  },
  preview: {
    host: true,
    allowedHosts: allowed,
  },
});
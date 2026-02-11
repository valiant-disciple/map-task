import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';

const allowed = [
  'localhost',
  '127.0.0.1',
  'distemperately-nonduplicative-elise.ngrok-free.dev',
];

export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: true,
    https: true,
    allowedHosts: allowed,
  },
  preview: {
    host: true,
    allowedHosts: allowed,
  },
});
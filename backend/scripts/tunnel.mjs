import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import localtunnel from 'localtunnel';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const port = Number(process.env.PORT || 3000);
const outPath = path.join(__dirname, '..', 'TUNNEL_URL.txt');

(async () => {
  const tunnel = await localtunnel({ port });
  fs.writeFileSync(outPath, tunnel.url, 'utf8');
  console.log(`[tunnel] ${tunnel.url}`);
  process.on('SIGINT', async () => {
    await tunnel.close();
    process.exit(0);
  });
})();



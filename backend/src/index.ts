import { config } from './config.js';
import { createServer } from './server.js';

const app = createServer();

app.listen(config.port, () => {
  // eslint-disable-next-line no-console
  console.log(`API listening on http://0.0.0.0:${config.port}`);
});



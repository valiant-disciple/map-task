import { config } from './config.js';
import { createServer } from './server.js';

const app = createServer();

import { setupWebSocket, broadcastCommand } from './websocket.js';

app.post('/api/control', (req, res) => {
  const { command } = req.body;
  if (command === 'start' || command === 'stop') {
    broadcastCommand(command);
    res.json({ success: true, command });
  } else {
    res.status(400).json({ error: 'Invalid command. Use "start" or "stop".' });
  }
});

const server = app.listen(config.port, () => {
  // eslint-disable-next-line no-console
  console.log(`API listening on http://0.0.0.0:${config.port}`);
});

setupWebSocket(server);



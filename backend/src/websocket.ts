
import { WebSocketServer, WebSocket } from 'ws';
import { Server } from 'http';

// Store connected clients
const clients: Set<WebSocket> = new Set();

export function setupWebSocket(server: Server) {
    const wss = new WebSocketServer({ server });

    wss.on('connection', (ws) => {
        console.log('Client connected');
        clients.add(ws);

        ws.on('message', (message) => {
            // Log to server console
            try {
                const msgString = message.toString();
                const data = JSON.parse(msgString);
                console.log('Received:', data);

                // Broadcast to all other clients (e.g. wscat, frontend)
                clients.forEach(client => {
                    if (client !== ws && client.readyState === WebSocket.OPEN) {
                        client.send(msgString);
                    }
                });
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        });

        ws.on('close', () => {
            console.log('Client disconnected');
            clients.delete(ws);
        });

        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
        });
    });

    console.log('WebSocket server initialized');
}

export function broadcastCommand(command: 'start' | 'stop') {
    console.log(`Broadcasting command: ${command}`);
    const message = JSON.stringify({ type: 'command', action: command });

    clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

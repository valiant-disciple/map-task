
import { WebSocketServer, WebSocket } from 'ws';
import { Server } from 'http';

// Interface for client messages
interface ClientMessage {
    type: 'join' | 'broadcast' | 'command';
    session?: string;
    event?: string;
    payload?: any;
    action?: string;
}

// Store clients by session
const sessions: Map<string, Set<WebSocket>> = new Map();

export function setupWebSocket(server: Server) {
    const wss = new WebSocketServer({ server });

    wss.on('connection', (ws) => {
        let currentSession: string | null = null;
        console.log('Client connected');

        ws.on('message', (message) => {
            try {
                const msgString = message.toString();
                const data = JSON.parse(msgString) as ClientMessage;

                if (data.type === 'join') {
                    if (data.session) {
                        currentSession = data.session;
                        if (!sessions.has(currentSession)) {
                            sessions.set(currentSession, new Set());
                        }
                        sessions.get(currentSession)?.add(ws);
                        console.log(`Client joined session: ${currentSession}`);
                        ws.send(JSON.stringify({ event: 'system', payload: { status: 'joined', session: currentSession } }));
                    }
                } else if (data.type === 'broadcast') {
                    if (currentSession && sessions.has(currentSession)) {
                        const room = sessions.get(currentSession);
                        room?.forEach(client => {
                            if (client !== ws && client.readyState === WebSocket.OPEN) {
                                // Forward the exact payload expected by frontend
                                client.send(JSON.stringify({
                                    event: data.event,
                                    payload: data.payload
                                }));
                            }
                        });
                    }
                }
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        });

        ws.on('close', () => {
            if (currentSession && sessions.has(currentSession)) {
                sessions.get(currentSession)?.delete(ws);
                if (sessions.get(currentSession)?.size === 0) {
                    sessions.delete(currentSession);
                }
            }
            console.log('Client disconnected');
        });

        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
        });
    });

    console.log('WebSocket server initialized (Room Support)');
}

// Deprecated global broadcast, but kept for watch app compatibility if needed
export function broadcastCommand(command: 'start' | 'stop') {
    // This functionality might need to be targeted to specific sessions in future
    // For now, no-op or global broadcast if you really want
    console.log(`Broadcast command (system-wide): ${command}`);
}

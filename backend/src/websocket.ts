
import { WebSocketServer, WebSocket } from 'ws';
import { Server } from 'http';
import { store } from './services/store.js';
import type { HrRecord } from './types/hr.js';

// Interface for client messages
interface ClientMessage {
    type: string;
    session?: string;
    event?: string;
    payload?: any;
    action?: string;
    deviceId?: string;
    ts?: number;
    values?: number[];
    accuracy?: number;
}

// Sensor types that should be relayed from watch → frontend
const SENSOR_TYPES = new Set(['HEART_RATE', 'ACCELEROMETER', 'GYROSCOPE']);

// Store clients by session
const sessions: Map<string, Set<WebSocket>> = new Map();

// Store ALL connected clients (for global HR relay)
const allClients: Set<WebSocket> = new Set();

export function setupWebSocket(server: Server) {
    const wss = new WebSocketServer({ server });

    wss.on('connection', (ws) => {
        let currentSession: string | null = null;
        allClients.add(ws);
        console.log('Client connected');

        ws.on('message', (message) => {
            try {
                const msgString = message.toString();
                const data = JSON.parse(msgString) as ClientMessage;

                if (data.type === 'ping') {
                    // Keepalive — respond with pong to keep Render from sleeping
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ event: 'pong' }));
                    }
                } else if (data.type === 'join') {
                    // ── Session join (Director / Matcher) ──
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
                    // ── Session broadcast (Director ↔ Matcher sync) ──
                    if (currentSession && sessions.has(currentSession)) {
                        const room = sessions.get(currentSession);
                        room?.forEach(client => {
                            if (client !== ws && client.readyState === WebSocket.OPEN) {
                                client.send(JSON.stringify({
                                    event: data.event,
                                    payload: data.payload
                                }));
                            }
                        });
                    }
                } else if (SENSOR_TYPES.has(data.type)) {
                    // ── Sensor data from watch → relay to ALL other clients ──
                    allClients.forEach(client => {
                        if (client !== ws && client.readyState === WebSocket.OPEN) {
                            client.send(msgString);
                        }
                    });

                    // Also persist HR data to the REST store so /api/hr/latest works
                    if (data.type === 'HEART_RATE' && Array.isArray(data.values) && data.values.length > 0) {
                        const record: HrRecord = {
                            deviceId: data.deviceId || 'watch',
                            ts: data.ts || Date.now(),
                            hr: Math.round(data.values[0]),
                            ibi: [],
                        };
                        store.add(record);
                    }
                }
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        });

        ws.on('close', () => {
            allClients.delete(ws);
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

    console.log('WebSocket server initialized (Room + HR Relay Support)');
}

// Deprecated global broadcast, but kept for watch app compatibility if needed
export function broadcastCommand(command: 'start' | 'stop') {
    // This functionality might need to be targeted to specific sessions in future
    // For now, no-op or global broadcast if you really want
    console.log(`Broadcast command (system-wide): ${command}`);
}

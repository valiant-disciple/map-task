// Watch Service - Manages WebSocket connection to watch backend for HR data

export interface HRReading {
    t: number; // Unix timestamp in ms
    bpm: number;
    phase: 'baseline' | 'trial' | 'idle';
}

type HRCallback = (reading: HRReading) => void;
type StatusCallback = (status: 'connected' | 'disconnected' | 'connecting') => void;

class WatchService {
    private ws: WebSocket | null = null;
    private url: string;
    private hrCallbacks: HRCallback[] = [];
    private statusCallbacks: StatusCallback[] = [];
    private status: 'connected' | 'disconnected' | 'connecting' = 'disconnected';
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private currentPhase: 'baseline' | 'trial' | 'idle' = 'idle';

    // Stored HR readings
    public readings: HRReading[] = [];

    constructor(url?: string) {
        // Priority: 1. Constructor arg, 2. Env var, 3. Deployed Backend (User provided), 4. Localhost
        this.url = url || import.meta.env.VITE_WATCH_SERVER_URL || 'wss://watch-hr-backend.onrender.com';
    }

    setUrl(url: string) {
        this.url = url;
    }

    setPhase(phase: 'baseline' | 'trial' | 'idle') {
        this.currentPhase = phase;
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        this.setStatus('connecting');
        console.log(`[WatchService] Connecting to ${this.url}...`);

        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('[WatchService] Connected');
                this.setStatus('connected');
                if (this.reconnectTimer) {
                    clearTimeout(this.reconnectTimer);
                    this.reconnectTimer = null;
                }
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    // Expected format from watch: { type: 'sensor', sensor: 'hr', bpm: number, timestamp: number }
                    // or { event: 'sensor', payload: { sensor: 'hr', bpm: number } }
                    let bpm: number | undefined;
                    let timestamp: number = Date.now();

                    if (data.type === 'sensor' && data.sensor === 'hr') {
                        bpm = data.bpm;
                        timestamp = data.timestamp || Date.now();
                    } else if (data.event === 'sensor' && data.payload?.sensor === 'hr') {
                        bpm = data.payload.bpm;
                        timestamp = data.payload.timestamp || Date.now();
                    } else if (data.type === 'HEART_RATE' && Array.isArray(data.values)) {
                        // Galaxy Watch format
                        bpm = data.values[0];
                        timestamp = data.ts || Date.now();
                    } else if (data.bpm !== undefined) {
                        // Direct bpm format
                        bpm = data.bpm;
                        timestamp = data.timestamp || Date.now();
                    }

                    if (bpm !== undefined && typeof bpm === 'number') {
                        const reading: HRReading = { t: timestamp, bpm, phase: this.currentPhase };
                        this.readings.push(reading);
                        this.hrCallbacks.forEach(cb => cb(reading));
                    }
                } catch (e) {
                    console.error('[WatchService] Parse error:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('[WatchService] Disconnected');
                this.setStatus('disconnected');
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('[WatchService] Error:', error);
                this.setStatus('disconnected');
            };
        } catch (e) {
            console.error('[WatchService] Connection error:', e);
            this.setStatus('disconnected');
            this.scheduleReconnect();
        }
    }

    private scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, 3000);
    }

    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.setStatus('disconnected');
    }

    private setStatus(status: 'connected' | 'disconnected' | 'connecting') {
        this.status = status;
        this.statusCallbacks.forEach(cb => cb(status));
    }

    getStatus() {
        return this.status;
    }

    onHR(callback: HRCallback) {
        this.hrCallbacks.push(callback);
        return () => {
            this.hrCallbacks = this.hrCallbacks.filter(cb => cb !== callback);
        };
    }

    onStatusChange(callback: StatusCallback) {
        this.statusCallbacks.push(callback);
        return () => {
            this.statusCallbacks = this.statusCallbacks.filter(cb => cb !== callback);
        };
    }

    clearReadings() {
        this.readings = [];
    }

    getReadings() {
        return [...this.readings];
    }

    // Get readings for a specific trial or baseline
    getReadingsForPhase(phase: 'baseline' | 'trial' | 'idle') {
        return this.readings.filter(r => r.phase === phase);
    }

    // Calculate average HR for baseline
    getBaselineAverage(): number | null {
        const baselineReadings = this.getReadingsForPhase('baseline');
        if (baselineReadings.length === 0) return null;
        const sum = baselineReadings.reduce((acc, r) => acc + r.bpm, 0);
        return Math.round(sum / baselineReadings.length);
    }

    // Simulation mode for testing without real watch
    private simulationInterval: ReturnType<typeof setInterval> | null = null;

    startSimulation() {
        if (this.simulationInterval) return;
        console.log('[WatchService] Starting HR simulation');
        this.setStatus('connected');

        this.simulationInterval = setInterval(() => {
            // Generate random HR between 60-100 with some variance
            const baseBpm = 72;
            const variance = Math.floor(Math.random() * 20) - 10;
            const bpm = baseBpm + variance;

            const reading: HRReading = { t: Date.now(), bpm, phase: this.currentPhase };
            this.readings.push(reading);
            this.hrCallbacks.forEach(cb => cb(reading));
        }, 1000);
    }

    stopSimulation() {
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
            this.simulationInterval = null;
            this.setStatus('disconnected');
            console.log('[WatchService] Stopped HR simulation');
        }
    }

    isSimulating() {
        return this.simulationInterval !== null;
    }
}

// Export a singleton instance
export const watchService = new WatchService();

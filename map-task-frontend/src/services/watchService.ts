// Watch Service - Manages WebSocket connection to watch backend for HR data
// Falls back to REST polling if WebSocket messages aren't arriving

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
    private restUrl: string; // Base URL for REST API fallback
    private hrCallbacks: HRCallback[] = [];
    private statusCallbacks: StatusCallback[] = [];
    private status: 'connected' | 'disconnected' | 'connecting' = 'disconnected';
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private currentPhase: 'baseline' | 'trial' | 'idle' = 'idle';

    // REST polling fallback
    private pollTimer: ReturnType<typeof setInterval> | null = null;
    private wsMessageReceived = false;
    private pollFallbackStarted = false;
    private lastPollHr: number | null = null;
    private lastPollTs: number = 0;

    // Stored HR readings
    public readings: HRReading[] = [];
    public msgCount: number = 0;
    public pollCount: number = 0;
    public source: 'ws' | 'poll' | 'sim' | 'none' = 'none';

    constructor(url?: string) {
        const base = url
            || import.meta.env.VITE_WATCH_SERVER_URL
            || 'wss://watch-hr-backend.onrender.com';
        this.url = base;
        // Derive REST URL from WS URL (wss:// → https://)
        this.restUrl = base.replace('wss://', 'https://').replace('ws://', 'http://');
    }

    setUrl(url: string) {
        this.url = url;
        this.restUrl = url.replace('wss://', 'https://').replace('ws://', 'http://');
    }

    setPhase(phase: 'baseline' | 'trial' | 'idle') {
        this.currentPhase = phase;
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        this.setStatus('connecting');
        this.wsMessageReceived = false;
        console.log(`[WatchService] Connecting WS to ${this.url}...`);

        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('[WatchService] WS Connected');
                this.setStatus('connected');
                if (this.reconnectTimer) {
                    clearTimeout(this.reconnectTimer);
                    this.reconnectTimer = null;
                }

                // Start polling fallback after 5s if no WS messages arrive
                setTimeout(() => {
                    if (!this.wsMessageReceived && !this.pollFallbackStarted) {
                        console.log('[WatchService] No WS messages after 5s — starting REST poll fallback');
                        this.startPolling();
                    }
                }, 5000);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.msgCount++;
                    this.wsMessageReceived = true;

                    // If WS messages are flowing, stop polling
                    if (this.pollFallbackStarted) {
                        console.log('[WatchService] WS messages resumed — stopping poll');
                        this.stopPolling();
                    }
                    this.source = 'ws';

                    let bpm: number | undefined;
                    let timestamp: number = Date.now();

                    if (data.type === 'sensor' && data.sensor === 'hr') {
                        bpm = data.bpm;
                        timestamp = data.timestamp || Date.now();
                    } else if (data.event === 'sensor' && data.payload?.sensor === 'hr') {
                        bpm = data.payload.bpm;
                        timestamp = data.payload.timestamp || Date.now();
                    } else if (data.type === 'HEART_RATE' && Array.isArray(data.values)) {
                        bpm = Math.round(data.values[0]);
                        timestamp = data.ts || Date.now();
                    } else if (data.bpm !== undefined) {
                        bpm = data.bpm;
                        timestamp = data.timestamp || Date.now();
                    }

                    if (bpm !== undefined && typeof bpm === 'number') {
                        this.pushReading(timestamp, bpm);
                    }
                } catch (e) {
                    console.error('[WatchService] Parse error:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('[WatchService] WS Disconnected');
                this.setStatus('disconnected');
                this.scheduleReconnect();
                // Keep polling if it's running
                if (!this.pollFallbackStarted) {
                    this.startPolling();
                }
            };

            this.ws.onerror = (error) => {
                console.error('[WatchService] WS Error:', error);
                this.setStatus('disconnected');
            };
        } catch (e) {
            console.error('[WatchService] Connection error:', e);
            this.setStatus('disconnected');
            this.scheduleReconnect();
            this.startPolling();
        }
    }

    private pushReading(timestamp: number, bpm: number) {
        const reading: HRReading = { t: timestamp, bpm, phase: this.currentPhase };
        this.readings.push(reading);
        this.hrCallbacks.forEach(cb => cb(reading));
        console.log(`[WatchService] HR: ${bpm} bpm (src: ${this.source}, msgs: ${this.msgCount}, polls: ${this.pollCount})`);
    }

    // ── REST polling fallback ──
    private startPolling() {
        if (this.pollFallbackStarted) return;
        this.pollFallbackStarted = true;
        this.source = 'poll';
        console.log(`[WatchService] Starting REST poll: ${this.restUrl}/api/hr/latest`);

        // Also set status to connected since we can get data via polling
        this.setStatus('connected');

        this.pollTimer = setInterval(async () => {
            try {
                const resp = await fetch(`${this.restUrl}/api/hr/latest`);
                if (!resp.ok) return;
                const data = await resp.json();
                // data: { deviceId, ts, hr, ibi }
                if (data.hr !== undefined && typeof data.hr === 'number') {
                    // Only emit if the reading is new (different timestamp)
                    if (data.ts !== this.lastPollTs) {
                        this.lastPollTs = data.ts;
                        this.lastPollHr = data.hr;
                        this.pollCount++;
                        this.pushReading(data.ts, data.hr);
                    }
                }
            } catch {
                // Silently ignore fetch errors
            }
        }, 2000);
    }

    private stopPolling() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
            this.pollTimer = null;
        }
        this.pollFallbackStarted = false;
    }

    private scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, 3000);
    }

    disconnect() {
        this.stopPolling();
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

    getUrl() {
        return this.url;
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

    getReadingsForPhase(phase: 'baseline' | 'trial' | 'idle') {
        return this.readings.filter(r => r.phase === phase);
    }

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
        this.source = 'sim';

        this.simulationInterval = setInterval(() => {
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
            this.source = 'none';
            console.log('[WatchService] Stopped HR simulation');
        }
    }

    isSimulating() {
        return this.simulationInterval !== null;
    }
}

// Export a singleton instance
export const watchService = new WatchService();

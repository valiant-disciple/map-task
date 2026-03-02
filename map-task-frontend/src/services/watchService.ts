// Watch Service — polls Render backend for latest HR

export interface HRReading {
    t: number;
    bpm: number;
    phase: 'baseline' | 'trial' | 'idle';
}

type HRCallback = (reading: HRReading) => void;
type StatusCallback = (status: 'connected' | 'disconnected' | 'connecting') => void;

// Resolve watch backend base from env and force HTTP(S) for REST polling.
const WATCH_BASE = (import.meta.env.VITE_WATCH_SERVER_URL || import.meta.env.VITE_WS_URL || 'http://localhost:3000')
  .replace('ws://', 'http://')
  .replace('wss://', 'https://');

class WatchService {
    private hrCallbacks: HRCallback[] = [];
    private statusCallbacks: StatusCallback[] = [];
    private status: 'connected' | 'disconnected' | 'connecting' = 'disconnected';
    private currentPhase: 'baseline' | 'trial' | 'idle' = 'idle';
    private pollTimer: ReturnType<typeof setInterval> | null = null;
    private lastTs: number = 0;
    public readings: HRReading[] = [];

    // Simulation
    private simulationInterval: ReturnType<typeof setInterval> | null = null;

    setPhase(phase: 'baseline' | 'trial' | 'idle') {
        this.currentPhase = phase;
    }

    connect() {
        if (this.pollTimer) return;
        this.setStatus('connecting');

        this.pollTimer = setInterval(async () => {
            try {
                const resp = await fetch(`${WATCH_BASE}/api/hr/latest`);
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.hr && data.ts !== this.lastTs) {
                    this.lastTs = data.ts;
                this.setStatus('connected');
                    const reading: HRReading = { t: data.ts, bpm: data.hr, phase: this.currentPhase };
                        this.readings.push(reading);
                        this.hrCallbacks.forEach(cb => cb(reading));
                    }
            } catch {
                // network hiccup, just retry next interval
            }
        }, 1500);
    }

    disconnect() {
        if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
        this.stopSimulation();
        this.setStatus('disconnected');
    }

    private setStatus(s: 'connected' | 'disconnected' | 'connecting') {
        if (this.status === s) return;
        this.status = s;
        this.statusCallbacks.forEach(cb => cb(s));
    }

    getStatus() { return this.status; }

    onHR(cb: HRCallback) {
        this.hrCallbacks.push(cb);
        return () => { this.hrCallbacks = this.hrCallbacks.filter(c => c !== cb); };
    }

    onStatusChange(cb: StatusCallback) {
        this.statusCallbacks.push(cb);
        return () => { this.statusCallbacks = this.statusCallbacks.filter(c => c !== cb); };
    }

    clearReadings() { this.readings = []; }
    getReadings() { return [...this.readings]; }
    getReadingsForPhase(phase: 'baseline' | 'trial' | 'idle') { return this.readings.filter(r => r.phase === phase); }

    getBaselineAverage(): number | null {
        const bl = this.getReadingsForPhase('baseline');
        if (bl.length === 0) return null;
        return Math.round(bl.reduce((s, r) => s + r.bpm, 0) / bl.length);
    }

    // Simulation for testing without watch
    startSimulation() {
        if (this.simulationInterval) return;
        this.setStatus('connected');
        this.simulationInterval = setInterval(() => {
            const bpm = 72 + Math.floor(Math.random() * 20) - 10;
            const reading: HRReading = { t: Date.now(), bpm, phase: this.currentPhase };
            this.readings.push(reading);
            this.hrCallbacks.forEach(cb => cb(reading));
        }, 1000);
    }

    stopSimulation() {
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
            this.simulationInterval = null;
        }
    }

    isSimulating() { return this.simulationInterval !== null; }
}

export const watchService = new WatchService();

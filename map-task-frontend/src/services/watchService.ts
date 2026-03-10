// Watch Service — polls Render backend for latest HR

export interface HRReading {
    t: number;
    bpm: number;
    phase: 'baseline' | 'trial' | 'idle';
}

type HRCallback = (reading: HRReading) => void;
type StatusCallback = (status: 'connected' | 'disconnected' | 'connecting') => void;

// Resolve watch backend base from env and force HTTP(S) for REST polling.
function resolveBase(role?: 'director' | 'matcher') {
    let base: string | undefined;
    if (role === 'director') base = import.meta.env.VITE_WATCH_SERVER_URL_DIRECTOR || import.meta.env.VITE_WATCH_SERVER_URL;
    else if (role === 'matcher') base = import.meta.env.VITE_WATCH_SERVER_URL_MATCHER || import.meta.env.VITE_WATCH_SERVER_URL;
    else base = import.meta.env.VITE_WATCH_SERVER_URL;

    if (!base) {
        throw new Error(`Watch backend URL missing for role ${role || 'unknown'}. Set VITE_WATCH_SERVER_URL or VITE_WATCH_SERVER_URL_${role?.toUpperCase?.() || 'DIRECTOR/MATCHER'}.`);
    }
    return base.replace('ws://', 'http://').replace('wss://', 'https://');
}

function resolveDeviceId(role?: 'director' | 'matcher') {
    if (role === 'director') return import.meta.env.VITE_WATCH_DEVICE_ID_DIRECTOR || import.meta.env.VITE_WATCH_DEVICE_ID;
    if (role === 'matcher') return import.meta.env.VITE_WATCH_DEVICE_ID_MATCHER || import.meta.env.VITE_WATCH_DEVICE_ID;
    return import.meta.env.VITE_WATCH_DEVICE_ID;
}

class WatchService {
    private hrCallbacks: HRCallback[] = [];
    private statusCallbacks: StatusCallback[] = [];
    private status: 'connected' | 'disconnected' | 'connecting' = 'disconnected';
    private currentPhase: 'baseline' | 'trial' | 'idle' = 'idle';
    private pollTimer: ReturnType<typeof setInterval> | null = null;
    private lastTs: number = 0;
    private consecutiveFailures: number = 0;
    private base: string = (() => {
        try { return resolveBase(); } catch { return ''; }
    })();
    private deviceId: string | undefined = resolveDeviceId();
    public readings: HRReading[] = [];

    // Simulation
    private simulationInterval: ReturnType<typeof setInterval> | null = null;

    setPhase(phase: 'baseline' | 'trial' | 'idle') {
        this.currentPhase = phase;
    }

    connect() {
        if (this.pollTimer) return;
        if (!this.base) {
            console.warn('[WatchService] No backend URL configured — HR will not be available');
            this.setStatus('disconnected');
            return;
        }
        this.setStatus('connecting');
        this.consecutiveFailures = 0;

        this.pollTimer = setInterval(async () => {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 8000);
            try {
                const qp = this.deviceId ? `?deviceId=${encodeURIComponent(this.deviceId)}` : '';
                const resp = await fetch(`${this.base}/api/hr/latest${qp}`, { signal: controller.signal });
                clearTimeout(timeout);
                if (!resp.ok) {
                    this.consecutiveFailures++;
                    console.warn(`[WatchService] HTTP ${resp.status} from ${this.base} (fail #${this.consecutiveFailures})`);
                    if (this.consecutiveFailures >= 5) this.setStatus('disconnected');
                    return;
                }
                const data = await resp.json();
                if (data.hr && data.ts !== this.lastTs) {
                    this.lastTs = data.ts;
                    this.consecutiveFailures = 0;
                    this.setStatus('connected');
                    const reading: HRReading = { t: data.ts, bpm: data.hr, phase: this.currentPhase };
                    this.readings.push(reading);
                    this.hrCallbacks.forEach(cb => cb(reading));
                }
            } catch (err: any) {
                clearTimeout(timeout);
                this.consecutiveFailures++;
                const reason = err?.name === 'AbortError' ? 'timeout' : (err?.message || 'unknown');
                console.warn(`[WatchService] Poll failed: ${reason} (fail #${this.consecutiveFailures})`);
                if (this.consecutiveFailures >= 5) this.setStatus('disconnected');
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

    setBase(url: string) {
        if (this.base === url) return;
        this.base = url;
        this.disconnect();
        this.lastTs = 0;
        this.connect();
    }

    setBaseForRole(role: 'director' | 'matcher') {
        this.setBase(resolveBase(role));
        this.deviceId = resolveDeviceId(role);
    }
}

export const watchService = new WatchService();

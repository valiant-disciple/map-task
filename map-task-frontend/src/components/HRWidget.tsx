import { useEffect, useState } from 'react';
import { watchService } from '../services/watchService';

interface HRWidgetProps {
    onBaselineComplete?: (avgBpm: number) => void;
    baselineDuration?: number;
    showSimToggle?: boolean;
}

export default function HRWidget({ onBaselineComplete, baselineDuration = 20, showSimToggle = true }: HRWidgetProps) {
    const [status, setStatus] = useState(watchService.getStatus());
    const [currentBpm, setCurrentBpm] = useState<number | null>(null);
    const [baselinePhase, setBaselinePhase] = useState<'idle' | 'measuring' | 'complete'>('idle');
    const [baselineRemain, setBaselineRemain] = useState(baselineDuration);
    const [baselineAvg, setBaselineAvg] = useState<number | null>(null);
    const [isSimulating, setIsSimulating] = useState(false);

    useEffect(() => {
        const unsubStatus = watchService.onStatusChange(setStatus);
        const unsubHR = watchService.onHR((reading) => setCurrentBpm(reading.bpm));
        watchService.connect();
        return () => { unsubStatus(); unsubHR(); };
    }, []);

    useEffect(() => {
        if (baselinePhase !== 'measuring') return;
        watchService.setPhase('baseline');
        const end = Date.now() + baselineDuration * 1000;

        const id = setInterval(() => {
            const left = Math.max(0, Math.ceil((end - Date.now()) / 1000));
            setBaselineRemain(left);
            if (left <= 0) {
                clearInterval(id);
                watchService.setPhase('idle');
                const avg = watchService.getBaselineAverage();
                setBaselineAvg(avg);
                setBaselinePhase('complete');
                if (onBaselineComplete) onBaselineComplete(avg ?? 0);
            }
        }, 200);

        return () => clearInterval(id);
    }, [baselinePhase, baselineDuration, onBaselineComplete]);

    const startBaseline = () => {
        if (status !== 'connected') {
            alert('Watch not connected. Connect or start simulation first.');
            return;
        }
        watchService.clearReadings();
        setBaselineRemain(baselineDuration);
        setBaselineAvg(null);
        setBaselinePhase('measuring');
    };

    const toggleSim = () => {
        if (watchService.isSimulating()) { watchService.stopSimulation(); setIsSimulating(false); }
        else { watchService.startSimulation(); setIsSimulating(true); }
    };

    const color = status === 'connected' ? '#4CAF50' : status === 'connecting' ? '#FFC107' : '#F44336';

    return (
        <div style={{ border: '1px solid #ccc', borderRadius: 8, padding: 12, backgroundColor: '#fafafa', minWidth: 180, fontSize: 14 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>❤️ Heart Rate</div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: color, display: 'inline-block' }} />
                <span style={{ textTransform: 'capitalize' }}>{status}</span>
                {isSimulating && <span style={{ color: '#888', fontSize: 11 }}>(Sim)</span>}
            </div>

            <div style={{ fontSize: 24, fontWeight: 700, color: '#333', marginBottom: 8 }}>
                {currentBpm !== null ? `${currentBpm} bpm` : '-- bpm'}
            </div>

            {baselinePhase === 'idle' && (
                <button onClick={startBaseline} style={{ width: '100%', padding: '6px 10px', cursor: 'pointer' }}>
                    Measure Baseline ({baselineDuration}s)
                </button>
            )}
            {baselinePhase === 'measuring' && (
                <div style={{ color: '#1976D2', fontWeight: 500 }}>Baseline: {baselineRemain}s left...</div>
            )}
            {baselinePhase === 'complete' && baselineAvg !== null && (
                <div style={{ color: '#388E3C', fontWeight: 500 }}>Baseline: {baselineAvg} bpm ✓</div>
            )}

            {showSimToggle && (
                <button onClick={toggleSim} style={{ marginTop: 8, width: '100%', padding: '4px 8px', fontSize: 12, cursor: 'pointer', backgroundColor: isSimulating ? '#ffcdd2' : '#e3f2fd' }}>
                    {isSimulating ? 'Stop Simulation' : 'Simulate HR'}
                </button>
            )}
        </div>
    );
}

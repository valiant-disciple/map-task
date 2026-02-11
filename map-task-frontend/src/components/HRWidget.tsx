import React, { useEffect, useState } from 'react';
import { watchService, type HRReading } from '../services/watchService';

interface HRWidgetProps {
    onBaselineComplete?: (avgBpm: number) => void;
    baselineDuration?: number; // seconds, default 20
    showSimToggle?: boolean;
}

export default function HRWidget({ onBaselineComplete, baselineDuration = 20, showSimToggle = true }: HRWidgetProps) {
    const [status, setStatus] = useState(watchService.getStatus());
    const [currentBpm, setCurrentBpm] = useState<number | null>(null);
    const [baselinePhase, setBaselinePhase] = useState<'idle' | 'measuring' | 'complete'>('idle');
    const [baselineRemain, setBaselineRemain] = useState(baselineDuration);
    const [baselineAvg, setBaselineAvg] = useState<number | null>(null);
    const [isSimulating, setIsSimulating] = useState(watchService.isSimulating());
    const [msgCount, setMsgCount] = useState(0);

    useEffect(() => {
        const unsubStatus = watchService.onStatusChange(setStatus);
        const unsubHR = watchService.onHR((reading) => {
            setCurrentBpm(reading.bpm);
            setMsgCount(watchService.msgCount + watchService.pollCount);
        });

        // Try to connect on mount
        watchService.connect();

        return () => {
            unsubStatus();
            unsubHR();
        };
    }, []);

    useEffect(() => {
        if (baselinePhase !== 'measuring') return;

        watchService.setPhase('baseline');
        const startTime = Date.now();
        const endTime = startTime + baselineDuration * 1000;

        // Force initial update
        setBaselineRemain(baselineDuration);

        const intervalId = setInterval(() => {
            const now = Date.now();
            const remaining = Math.max(0, Math.ceil((endTime - now) / 1000));

            // Only update if changed to avoid excessive renders
            setBaselineRemain(prev => {
                if (prev !== remaining) return remaining;
                return prev;
            });

            if (remaining <= 0) {
                clearInterval(intervalId);
                watchService.setPhase('idle');
                const avg = watchService.getBaselineAverage();
                console.log('[HRWidget] Baseline complete. Avg:', avg);
                setBaselineAvg(avg);
                setBaselinePhase('complete');
                if (avg !== null && onBaselineComplete) {
                    onBaselineComplete(avg);
                } else if (avg === null) {
                    // Fallback if no data
                    console.warn('[HRWidget] No HR data collected during baseline');
                    setBaselineAvg(0);
                    if (onBaselineComplete) onBaselineComplete(0);
                }
            }
        }, 100); // 100ms for smoother checks

        return () => clearInterval(intervalId);
    }, [baselinePhase, baselineDuration, onBaselineComplete]);

    const startBaseline = () => {
        if (status !== 'connected') {
            alert('Watch not connected. Please connect or start simulation first.');
            return;
        }
        // Clear previous baseline readings
        watchService.clearReadings();
        setBaselineRemain(baselineDuration);
        setBaselineAvg(null);
        setBaselinePhase('measuring');
    };

    const toggleSimulation = () => {
        if (watchService.isSimulating()) {
            watchService.stopSimulation();
            setIsSimulating(false);
        } else {
            watchService.startSimulation();
            setIsSimulating(true);
        }
    };

    const statusColor = status === 'connected' ? '#4CAF50' : status === 'connecting' ? '#FFC107' : '#F44336';

    return (
        <div style={{
            border: '1px solid #ccc',
            borderRadius: 8,
            padding: 12,
            backgroundColor: '#fafafa',
            minWidth: 180,
            fontSize: 14
        }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>❤️ Heart Rate</div>

            {/* Connection Status */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                <span style={{
                    width: 10,
                    height: 10,
                    borderRadius: '50%',
                    backgroundColor: statusColor,
                    display: 'inline-block'
                }} />
                <span style={{ textTransform: 'capitalize' }}>{status}</span>
                {isSimulating && <span style={{ color: '#888', fontSize: 11 }}>(Sim)</span>}
            </div>

            {/* Current BPM */}
            <div style={{ fontSize: 24, fontWeight: 700, color: '#333', marginBottom: 8 }}>
                {currentBpm !== null ? `${currentBpm} bpm` : '-- bpm'}
            </div>

            {/* Baseline Section */}
            {baselinePhase === 'idle' && (
                <button
                    onClick={startBaseline}
                    style={{ width: '100%', padding: '6px 10px', cursor: 'pointer' }}
                >
                    Measure Baseline ({baselineDuration}s)
                </button>
            )}

            {baselinePhase === 'measuring' && (
                <div style={{ color: '#1976D2', fontWeight: 500 }}>
                    Baseline: {baselineRemain}s left...
                </div>
            )}

            {baselinePhase === 'complete' && baselineAvg !== null && (
                <div style={{ color: '#388E3C', fontWeight: 500 }}>
                    Baseline: {baselineAvg} bpm ✓
                </div>
            )}

            {/* Simulation Toggle */}
            {showSimToggle && (
                <button
                    onClick={toggleSimulation}
                    style={{
                        marginTop: 8,
                        width: '100%',
                        padding: '4px 8px',
                        fontSize: 12,
                        cursor: 'pointer',
                        backgroundColor: isSimulating ? '#ffcdd2' : '#e3f2fd'
                    }}
                >
                    {isSimulating ? 'Stop Simulation' : 'Simulate HR'}
                </button>
            )}

            {/* Debug info */}
            <div style={{ marginTop: 8, fontSize: 10, color: '#999', wordBreak: 'break-all' }}>
                <div>Src: {watchService.source} | Msgs: {msgCount} | Polls: {watchService.pollCount}</div>
            </div>
        </div>
    );
}

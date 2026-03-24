import React, { useState, useEffect, useRef, useCallback } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

interface MicCheckWidgetProps {
    onConfirm: () => void;
    selectedMicId: string;
    devices: MediaDeviceInfo[];
    onSelectMic: (deviceId: string) => void;
    onRefreshDevices?: () => void;
}

export default function MicCheckWidget({ onConfirm, selectedMicId, devices, onSelectMic, onRefreshDevices }: MicCheckWidgetProps) {
    const [isConfirmed, setIsConfirmed] = useState(false);
    const [volume, setVolume] = useState(0);
    const [isListening, setIsListening] = useState(false);

    // Volume Meter Refs
    const audioContextRef = useRef<AudioContext | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    // react-speech-recognition hook
    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition,
        isMicrophoneAvailable
    } = useSpeechRecognition();

    // Sync listening state
    useEffect(() => {
        setIsListening(listening);
    }, [listening]);

    const startVolumeMeter = useCallback(async () => {
        try {
            if (selectedMicId) {
                if (!navigator.mediaDevices) {
                    console.warn('[MicCheck] mediaDevices unavailable (insecure context)');
                    return;
                }
                // Use ideal (not exact) so it falls back to any mic if the
                // selected device is unavailable (e.g., stale deviceId on lab PCs)
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: { deviceId: { ideal: selectedMicId } }
                });
                mediaStreamRef.current = stream;

                const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
                audioContextRef.current = audioCtx;
                const analyser = audioCtx.createAnalyser();
                analyser.fftSize = 256;
                analyserRef.current = analyser;

                const source = audioCtx.createMediaStreamSource(stream);
                source.connect(analyser);

                const updateVolume = () => {
                    if (!mediaStreamRef.current) return;
                    const dataArray = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(dataArray);
                    const avg = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
                    setVolume(avg);
                    animationFrameRef.current = requestAnimationFrame(updateVolume);
                };
                updateVolume();
            }
        } catch (e) {
            console.error('[MicCheck] Failed to start volume meter', e);
        }
    }, [selectedMicId]);

    const stopVolumeMeter = useCallback(() => {
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(t => t.stop());
            mediaStreamRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        setVolume(0);
    }, []);

    const handleStartListening = async () => {
        resetTranscript();
        await startVolumeMeter();
        SpeechRecognition.startListening({ continuous: true, language: 'en-US' });
    };

    const handleStopListening = () => {
        SpeechRecognition.stopListening();
        stopVolumeMeter();
    };

    const handleConfirm = () => {
        handleStopListening();
        setIsConfirmed(true);
        onConfirm();
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopVolumeMeter();
            SpeechRecognition.stopListening();
        };
    }, [stopVolumeMeter]);

    if (!navigator.mediaDevices) {
        return (
            <div style={{ padding: 12, border: '1px solid #e53935', borderRadius: 8, backgroundColor: '#ffebee', marginTop: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>🎙️ Mic Check</div>
                <div style={{ fontSize: 12, color: '#c62828' }}>
                    ⚠️ Microphone access requires <b>localhost</b>.
                    <br />Open <a href="http://localhost:5173" style={{ color: '#1565c0' }}>http://localhost:5173</a> instead.
                </div>
                <button
                    onClick={() => { onConfirm(); setIsConfirmed(true); }}
                    style={{ marginTop: 8, padding: '6px 12px', backgroundColor: '#e3f2fd', border: '1px solid #bbdefb', borderRadius: 4, cursor: 'pointer' }}
                >
                    Skip Mic Check
                </button>
            </div>
        );
    }

    if (!browserSupportsSpeechRecognition) {
        return (
            <div style={{ padding: 12, border: '1px solid #ccc', borderRadius: 8, backgroundColor: '#f9f9f9', marginTop: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>🎙️ Mic Check</div>
                <div style={{ fontSize: 12, color: '#888' }}>
                    Speech recognition not supported in this browser.
                    <br />Use Chrome for best experience.
                </div>
                <button
                    onClick={() => { onConfirm(); setIsConfirmed(true); }}
                    style={{ marginTop: 8, padding: '6px 12px', backgroundColor: '#e3f2fd', border: '1px solid #bbdefb', borderRadius: 4, cursor: 'pointer' }}
                >
                    Skip Mic Check
                </button>
            </div>
        );
    }

    if (isConfirmed) {
        return (
            <div style={{
                padding: 8,
                border: '1px solid #4CAF50',
                borderRadius: 8,
                backgroundColor: '#E8F5E9',
                fontSize: 14,
                color: '#2E7D32',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginTop: 8
            }}>
                <span>✅ Mic Checked</span>
                <button
                    onClick={() => setIsConfirmed(false)}
                    style={{ padding: '2px 6px', fontSize: 11, background: 'none', border: '1px solid #2E7D32', borderRadius: 4, cursor: 'pointer' }}
                >
                    Redo
                </button>
            </div>
        );
    }

    return (
        <div style={{
            padding: 12,
            border: '1px solid #ccc',
            borderRadius: 8,
            backgroundColor: '#f9f9f9',
            marginTop: 8
        }}>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>🎙️ Mic Check</div>

            {/* Microphone Selection */}
            <div style={{ marginBottom: 10, display: 'flex', gap: 8 }}>
                <select
                    value={selectedMicId}
                    onChange={(e) => onSelectMic(e.target.value)}
                    style={{
                        flex: 1,
                        padding: '4px',
                        fontSize: 12,
                        borderRadius: 4,
                        border: '1px solid #ccc'
                    }}
                >
                    {devices.length === 0 && <option value="">No microphones found...</option>}
                    {devices.map(d => (
                        <option key={d.deviceId} value={d.deviceId}>
                            {d.label || 'Mic ' + d.deviceId.slice(0, 4)}
                        </option>
                    ))}
                </select>
                <button
                    onClick={onRefreshDevices}
                    title="Refresh Device List"
                    style={{
                        padding: '2px 8px',
                        cursor: 'pointer',
                        fontSize: 14,
                        background: '#fff',
                        border: '1px solid #ccc',
                        borderRadius: 4
                    }}
                >
                    🔄
                </button>
            </div>

            {/* Volume Meter Bar */}
            <div style={{
                height: 6,
                width: '100%',
                backgroundColor: '#ddd',
                borderRadius: 3,
                marginBottom: 10,
                overflow: 'hidden'
            }}>
                <div style={{
                    height: '100%',
                    width: `${Math.min(100, volume * 3)}%`,
                    backgroundColor: volume > 10 ? '#4CAF50' : '#888',
                    transition: 'width 0.1s ease-out'
                }} />
            </div>

            {/* Control Buttons */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <button
                    onClick={isListening ? handleStopListening : handleStartListening}
                    style={{
                        flex: 1,
                        padding: '6px 12px',
                        backgroundColor: isListening ? '#ffebee' : '#e3f2fd',
                        color: isListening ? '#c62828' : '#1565c0',
                        border: `1px solid ${isListening ? '#ef9a9a' : '#bbdefb'}`,
                        borderRadius: 4,
                        cursor: 'pointer',
                        fontWeight: 500
                    }}
                >
                    {isListening ? '🛑 Stop Checking' : '🎤 Check Signal'}
                </button>

                {(volume > 5 || transcript.length > 0) && (
                    <button
                        onClick={handleConfirm}
                        style={{
                            flex: 1,
                            padding: '6px 12px',
                            backgroundColor: '#e8f5e9',
                            color: '#2e7d32',
                            border: '1px solid #a5d6a7',
                            borderRadius: 4,
                            cursor: 'pointer',
                            fontWeight: 600
                        }}
                    >
                        ✅ Looks Good
                    </button>
                )}
            </div>

            {/* Transcription Display */}
            <div style={{
                minHeight: 40,
                maxHeight: 80,
                overflowY: 'auto',
                padding: 8,
                backgroundColor: '#fff',
                border: '1px solid #eee',
                borderRadius: 4,
                fontSize: 13,
                fontStyle: transcript ? 'normal' : 'italic',
                color: transcript ? '#333' : '#aaa'
            }}>
                {transcript || (isListening ? "Listening... speak now" : "Click 'Check Signal' and speak")}
            </div>

            {/* Mic availability warning */}
            {!isMicrophoneAvailable && isListening && (
                <div style={{ marginTop: 8, fontSize: 11, color: '#c62828' }}>
                    ⚠️ Microphone access denied. Check browser permissions.
                </div>
            )}
        </div>
    );
}

export interface AudioRecordingResult {
    blob: Blob;
    deviceId: string;
    deviceLabel: string;
    startTime: number;
    durationMs: number;
    mimeType: string;
}

export class AudioRecorder {
    private mediaRecorder: MediaRecorder | null = null;
    private chunks: Blob[] = [];
    private startTime: number = 0;
    private stream: MediaStream | null = null;

    private deviceId: string = 'default';
    private deviceLabel: string = 'Default Input';

    async getDevices(): Promise<MediaDeviceInfo[]> {
        try {
            if (!navigator.mediaDevices) {
                console.warn('[AudioRecorder] mediaDevices unavailable — use localhost or HTTPS');
                return [];
            }
            // Check if we already have permission (labels are populated)
            let allDevices = await navigator.mediaDevices.enumerateDevices();
            const hasLabels = allDevices.some(d => d.kind === 'audioinput' && d.label);
            if (!hasLabels) {
                // Need to request permission first — acquire and release
                const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                tempStream.getTracks().forEach(t => t.stop());
                // Brief delay to let driver fully release the device
                await new Promise(r => setTimeout(r, 100));
                allDevices = await navigator.mediaDevices.enumerateDevices();
            }
            const inputs = allDevices.filter(d => d.kind === 'audioinput');
            console.log(`[AudioRecorder] Found ${inputs.length} audio inputs`);
            return inputs;
        } catch (err) {
            console.error('Error listing audio devices:', err);
            return [];
        }
    }

    async start(deviceId?: string): Promise<void> {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            console.warn('AudioRecorder already recording');
            return;
        }

        if (!navigator.mediaDevices) {
            throw new Error('mediaDevices unavailable — page must be served over HTTPS or localhost');
        }

        // Try exact deviceId first, fall back to any mic if it fails
        // (handles stale deviceIds after device reconnect / driver restart)
        let stream: MediaStream | null = null;
        if (deviceId) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: { deviceId: { exact: deviceId } }
                });
            } catch (exactErr) {
                console.warn(`[AudioRecorder] Exact device ${deviceId} failed, falling back to preferred`, exactErr);
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        audio: { deviceId: { ideal: deviceId } }
                    });
                } catch (idealErr) {
                    console.warn('[AudioRecorder] Preferred device failed, falling back to default', idealErr);
                    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                }
            }
        } else {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }

        this.stream = stream;

        const track = this.stream.getAudioTracks()[0];
        this.deviceId = track.getSettings().deviceId || deviceId || 'default';
        this.deviceLabel = track.label || 'Unknown Device';

        // Prefer standard webm/opus, then plain webm, then whatever is available
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : MediaRecorder.isTypeSupported('audio/webm')
                ? 'audio/webm'
                : '';

        this.mediaRecorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
        this.chunks = [];

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.chunks.push(e.data);
        };

        this.mediaRecorder.start();
        this.startTime = Date.now();
        console.log(`[AudioRecorder] Started on ${this.deviceLabel} (${this.deviceId}), mime: ${this.mediaRecorder.mimeType}`);
    }

    async stop(): Promise<AudioRecordingResult> {
        return new Promise((resolve, reject) => {
            if (!this.mediaRecorder) {
                return reject(new Error('No active recorder'));
            }

            this.mediaRecorder.onstop = () => {
                const durationMs = Date.now() - this.startTime;
                const blob = new Blob(this.chunks, { type: this.mediaRecorder?.mimeType || 'audio/webm' });

                // Stop all tracks
                this.stream?.getTracks().forEach(t => t.stop());
                this.stream = null;
                this.mediaRecorder = null;

                resolve({
                    blob,
                    deviceId: this.deviceId,
                    deviceLabel: this.deviceLabel,
                    startTime: this.startTime,
                    durationMs,
                    mimeType: blob.type
                });
            };

            this.mediaRecorder.stop();
        });
    }

    isRecording(): boolean {
        return this.mediaRecorder?.state === 'recording';
    }
}

export const audioRecorder = new AudioRecorder();

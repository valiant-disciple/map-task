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
            // enhance permission prompt if not granted
            await navigator.mediaDevices.getUserMedia({ audio: true });
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(d => d.kind === 'audioinput');
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

        try {
            const constraints: MediaStreamConstraints = {
                audio: deviceId ? { deviceId: { exact: deviceId } } : true
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);

            const track = this.stream.getAudioTracks()[0];
            this.deviceId = deviceId || track.getSettings().deviceId || 'default';
            this.deviceLabel = track.label || 'Unknown Device';

            // Prefer standard webm/opus
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
            this.chunks = [];

            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.chunks.push(e.data);
            };

            this.mediaRecorder.start();
            this.startTime = Date.now();
            console.log(`[AudioRecorder] Started on ${this.deviceLabel} (${this.deviceId})`);

        } catch (err) {
            console.error('Failed to start recording:', err);
            throw err;
        }
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

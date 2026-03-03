import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import { Readable } from 'stream';
import { promises as fs } from 'fs';
import os from 'os';
import path from 'path';

ffmpeg.setFfmpegPath(ffmpegPath || '');

export async function toWav(buffer: Buffer): Promise<Buffer> {
  // Convert arbitrary audio to wav using ffmpeg in-memory
  const tmpIn = path.join(os.tmpdir(), `audio-in-${Date.now()}.bin`);
  const tmpOut = path.join(os.tmpdir(), `audio-out-${Date.now()}.wav`);
  await fs.writeFile(tmpIn, buffer);
  await new Promise<void>((resolve, reject) => {
    ffmpeg(tmpIn)
      .outputOptions(['-ar 16000', '-ac 1', '-f wav'])
      .save(tmpOut)
      .on('end', () => resolve())
      .on('error', reject);
  });
  const wav = await fs.readFile(tmpOut);
  await fs.unlink(tmpIn).catch(() => {});
  await fs.unlink(tmpOut).catch(() => {});
  return wav;
}

export async function transcribeSmallestAI(wav: Buffer, apiKey: string) {
  const endpoint = 'https://api.smallest.ai/waves/v1/pulse/get_text';
  const params = new URLSearchParams({ language: 'en', word_timestamps: 'true' }).toString();
  const resp = await fetch(`${endpoint}?${params}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'audio/wav'
    },
    body: wav,
    signal: AbortSignal.timeout(120_000)
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`ASR failed ${resp.status}: ${text}`);
  }
  const json = await resp.json();
  return json;
}

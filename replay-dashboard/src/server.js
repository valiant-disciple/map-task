import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import os from 'os';
import { execFile } from 'child_process';

const PYTHON_BIN = process.env.PYTHON || process.env.PYTHON_BIN || 'python3';

const runCmd = (cmd, args = []) => new Promise((resolve, reject) => {
  execFile(cmd, args, { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
    if (err) return reject(Object.assign(err, { stdout, stderr }));
    resolve({ stdout, stderr });
  });
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.static(path.join(__dirname, '..', 'public')));
app.use('/Ground%20Truth%20Maps', express.static(path.join(__dirname, '..', 'Ground Truth Maps')));

// Accept raw ZIP upload and run the Python postprocess script (full RQA, ASR if key set)
app.post('/api/process-zip', express.raw({ type: 'application/octet-stream', limit: '800mb' }), async (req, res) => {
  if (!req.body || !req.body.length) {
    res.status(400).send('No file');
    return;
  }
  const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'replay-'));
  const inZip = path.join(tmpDir, 'session.zip');
  const outDir = path.join(tmpDir, 'out');
  await fs.promises.writeFile(inZip, req.body);
  try {
    // Run Python processor
    const ppArgs = [
      path.join(__dirname, '..', 'scripts', 'postprocess.py'),
      '--zip', inZip,
      '--gt-dir', path.join(__dirname, '..', 'Ground Truth Maps'),
      '--out', outDir,
    ];
    const asrKey = process.env.SMALLEST_AI_KEY;
    if (asrKey) ppArgs.push('--smallest-key', asrKey);
    await runCmd(PYTHON_BIN, ppArgs);
    // Zip the output dir
    const outZip = path.join(tmpDir, 'output.zip');
    const zipScript = `
import zipfile, pathlib, sys
root = pathlib.Path(sys.argv[1])
out = sys.argv[2]
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(root))
`;
    await runCmd(PYTHON_BIN, ['-c', zipScript, outDir, outZip]);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', 'attachment; filename="postprocess_output.zip"');
    fs.createReadStream(outZip).pipe(res);
  } catch (e) {
    console.error('process-zip failed', e);
    res.status(500).send(e?.message || 'Processing failed');
  }
});

const port = process.env.PORT || 4100;
app.listen(port, () => {
  console.log(`Replay dashboard server running at http://localhost:${port}`);
});

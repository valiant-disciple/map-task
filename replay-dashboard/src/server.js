import 'dotenv/config';
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
app.use('/maps', express.static(path.join(__dirname, '..', '..', 'map-task-frontend', 'src', 'assets', 'maps')));

let lastSessionZipPath = null;

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
  lastSessionZipPath = inZip;
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
    console.error('process-zip failed', e?.message || e);
    if (e?.stderr) console.error('STDERR:', e.stderr);
    if (e?.stdout) console.error('STDOUT:', e.stdout);
    const detail = [e?.message, e?.stderr, e?.stdout].filter(Boolean).join('\n');
    res.status(500).send(detail || 'Processing failed');
  }
});

app.post('/api/process-eye', express.raw({ type: 'application/octet-stream', limit: '500mb' }), async (req, res) => {
  const format = req.query.format;
  const role = req.query.role;
  if (!format || !role) return res.status(400).send('Missing format or role query param');
  if (!lastSessionZipPath) return res.status(400).send('Upload session ZIP first');
  if (!req.body || !req.body.length) return res.status(400).send('No file');

  const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'eye-'));
  const ext = format === 'smarteye' ? '.log' : '.csv';
  const eyePath = path.join(tmpDir, 'eye' + ext);
  const outCsv = path.join(tmpDir, 'eye_processed.csv');
  await fs.promises.writeFile(eyePath, req.body);

  try {
    await runCmd(PYTHON_BIN, [
      path.join(__dirname, '..', 'scripts', 'preprocess_eye.py'),
      '--eye-file', eyePath,
      '--format', format,
      '--role', role,
      '--zip', lastSessionZipPath,
      '--out', outCsv
    ]);
    const csvContent = await fs.promises.readFile(outCsv, 'utf-8');
    res.setHeader('Content-Type', 'text/csv');
    res.send(csvContent);
  } catch (e) {
    console.error('process-eye failed', e?.message || e);
    if (e?.stderr) console.error('STDERR:', e.stderr);
    const detail = [e?.message, e?.stderr, e?.stdout].filter(Boolean).join('\n');
    res.status(500).send(detail || 'Eye processing failed');
  }
});

const port = process.env.PORT || 4100;
app.listen(port, () => {
  console.log(`Replay dashboard server running at http://localhost:${port}`);
});

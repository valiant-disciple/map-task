import sharp from 'sharp';
import { createHash } from 'crypto';

type Stroke = { polyline: { x: number; y: number }[] };

async function imageToMask(buffer: Buffer, width: number, height: number, threshold = 180): Promise<Buffer> {
  return sharp(buffer)
    .resize(width, height)
    .greyscale()
    .threshold(threshold)
    .toColourspace('b-w')
    .raw()
    .toBuffer();
}

function svgFromStrokes(strokes: Stroke[], width: number, height: number): string {
  const paths = strokes
    .filter(s => s.polyline && s.polyline.length > 1)
    .map(s => {
      const pts = s.polyline.map(p => `${p.x},${p.y}`).join(' ');
      return `<polyline points="${pts}" fill="none" stroke="black" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />`;
    })
    .join('\n');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="100%" height="100%" fill="white"/>
    ${paths}
  </svg>`;
}

async function strokesToMask(strokes: Stroke[], width: number, height: number): Promise<Buffer> {
  if (!strokes.length) {
    return Buffer.alloc(width * height, 0);
  }
  const svg = svgFromStrokes(strokes, width, height);
  return sharp(Buffer.from(svg)).raw().toBuffer({ resolveWithObject: false }).then(buf => {
    // buf is RGB; threshold to binary
    const mask = Buffer.alloc(width * height);
    for (let i = 0, j = 0; i < buf.length; i += 3, j++) {
      const v = buf[i]; // red channel
      mask[j] = v > 10 ? 255 : 0;
    }
    return mask;
  });
}

function binaryMetrics(gt: Buffer, pred: Buffer): { iou: number; precision: number; recall: number; f1: number; dice: number } {
  const n = gt.length;
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < n; i++) {
    const g = gt[i] > 0;
    const p = pred[i] > 0;
    if (p && g) tp++;
    else if (p && !g) fp++;
    else if (!p && g) fn++;
  }
  const denomIou = tp + fp + fn;
  const iou = denomIou > 0 ? tp / denomIou : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const dice = f1;
  return { iou, precision, recall, f1, dice };
}

function ssim(gt: Buffer, pred: Buffer): number {
  // Simple SSIM for binary masks (0/255)
  const n = gt.length;
  let muX = 0, muY = 0;
  for (let i = 0; i < n; i++) { muX += gt[i]; muY += pred[i]; }
  muX /= n; muY /= n;
  let varX = 0, varY = 0, cov = 0;
  for (let i = 0; i < n; i++) {
    const dx = gt[i] - muX;
    const dy = pred[i] - muY;
    varX += dx * dx;
    varY += dy * dy;
    cov += dx * dy;
  }
  varX /= n; varY /= n; cov /= n;
  const C1 = 6.5025, C2 = 58.5225;
  return ((2 * muX * muY + C1) * (2 * cov + C2)) / ((muX * muX + muY * muY + C1) * (varX + varY + C2));
}

function hausdorff(gt: Buffer, pred: Buffer, width: number, height: number): number {
  // Compute symmetric Hausdorff on foreground pixels; fallback to 0 if empty.
  const fgCoords = (buf: Buffer) => {
    const coords: [number, number][] = [];
    for (let i = 0; i < buf.length; i++) {
      if (buf[i] > 0) {
        const y = Math.floor(i / width);
        const x = i - y * width;
        coords.push([x, y]);
      }
    }
    return coords;
  };
  const A = fgCoords(gt);
  const B = fgCoords(pred);
  if (!A.length || !B.length) return 0;
  const dist = (p: [number, number], q: [number, number]) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const directed = (P: [number, number][], Q: [number, number][]) => {
    let maxMin = 0;
    for (const p of P) {
      let min = Infinity;
      for (const q of Q) {
        const d = dist(p, q);
        if (d < min) min = d;
        if (min === 0) break;
      }
      if (min > maxMin) maxMin = min;
    }
    return maxMin;
  };
  return Math.max(directed(A, B), directed(B, A));
}

export async function computeMapAccuracy(options: {
  groundTruth: Buffer;
  matcherFinal?: Buffer | null;
  strokes?: Stroke[];
  threshold?: number;
}) {
  const { groundTruth, matcherFinal, strokes = [], threshold = 180 } = options;
  const meta = await sharp(groundTruth).metadata();
  const width = meta.width || 800;
  const height = meta.height || 800;

  const gtMask = await imageToMask(groundTruth, width, height, threshold);

  let predMask: Buffer;
  if (matcherFinal) {
    predMask = await imageToMask(matcherFinal, width, height, threshold);
  } else {
    predMask = await strokesToMask(strokes, width, height);
  }

  const metrics = binaryMetrics(gtMask, predMask);
  const ssimScore = ssim(gtMask, predMask);
  const hd = hausdorff(gtMask, predMask, width, height);

  return {
    width,
    height,
    ...metrics,
    ssim: ssimScore,
    hausdorff: hd,
  };
}

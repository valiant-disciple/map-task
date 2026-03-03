import sharp from 'sharp';

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

function binaryMetrics(gt: Buffer, pred: Buffer): { iou: number; precision: number; recall: number; f1: number } {
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
  return { iou, precision, recall, f1 };
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
  return {
    width,
    height,
    ...metrics,
  };
}

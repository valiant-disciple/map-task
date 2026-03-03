// Minimal MDRQA-style metrics for HR time series.
// This is a simplified implementation to avoid external Python deps.
// Metrics: recurrence rate (RR), determinism (DET), laminarity (LAM), average line length (L),
// longest diagonal (Lmax), longest vertical (Vmax).

type Series = number[];

type RqaMetrics = {
  rr: number;
  det: number;
  lam: number;
  l: number;
  lmax: number;
  vmax: number;
};

function buildRecurrenceMatrix(series: Series, epsilon: number): number[][] {
  const n = series.length;
  const m: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      m[i][j] = Math.abs(series[i] - series[j]) <= epsilon ? 1 : 0;
    }
  }
  return m;
}

function lineLengthsDiag(m: number[][]): number[] {
  const n = m.length;
  const lens: number[] = [];
  for (let k = -(n - 1); k <= n - 1; k++) {
    let run = 0;
    for (let i = 0; i < n; i++) {
      const j = i - k;
      if (j < 0 || j >= n) continue;
      if (m[i][j]) {
        run++;
      } else if (run > 0) {
        lens.push(run);
        run = 0;
      }
    }
    if (run > 0) lens.push(run);
  }
  return lens;
}

function lineLengthsVert(m: number[][]): number[] {
  const n = m.length;
  const lens: number[] = [];
  for (let col = 0; col < n; col++) {
    let run = 0;
    for (let row = 0; row < n; row++) {
      if (m[row][col]) {
        run++;
      } else if (run > 0) {
        lens.push(run);
        run = 0;
      }
    }
    if (run > 0) lens.push(run);
  }
  return lens;
}

export function computeMdrqa(hr: Series, epsilon: number = 1): RqaMetrics | null {
  if (!hr || hr.length < 4) return null;
  const m = buildRecurrenceMatrix(hr, epsilon);
  const n = hr.length;
  const totalPoints = n * n;
  const ones = m.flat().reduce((s, v) => s + v, 0);
  const rr = ones / totalPoints;

  const diagLens = lineLengthsDiag(m).filter(l => l >= 2);
  const vertLens = lineLengthsVert(m).filter(l => l >= 2);

  const diagPoints = diagLens.reduce((s, l) => s + l, 0);
  const det = ones > 0 ? diagPoints / ones : 0;
  const lmax = diagLens.length ? Math.max(...diagLens) : 0;
  const l = diagLens.length ? diagPoints / diagLens.length : 0;

  const vertPoints = vertLens.reduce((s, l) => s + l, 0);
  const lam = ones > 0 ? vertPoints / ones : 0;
  const vmax = vertLens.length ? Math.max(...vertLens) : 0;

  return { rr, det, lam, l, lmax, vmax };
}

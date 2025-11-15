import React, { useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react';

type Pt = { x:number; y:number };
export type CanvasBoardHandle = { exportPNG: () => string | null };

type Props = {
  imgSrc: string;
  mode: 'draw' | 'erase';
  onDrawEnd: (polyline: Pt[]) => void;
  onPos?: (pt: Pt) => void;
};

const CanvasBoard = forwardRef<CanvasBoardHandle, Props>(({ imgSrc, mode, onDrawEnd, onPos }, ref) => {
  const cvRef = useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan] = useState({ x: 0, y: 0 });
  const [current, setCurrent] = useState<Pt[]>([]);
  const [strokes, setStrokes] = useState<Pt[][]>([]);
  const [erasing, setErasing] = useState(false);
  const W = 1024, H = 768;
  const ERASE_PX = 12;

  useImperativeHandle(ref, () => ({
    exportPNG: () => {
      const cv = cvRef.current;
      if (!cv) return null;
      redraw();
      try { return cv.toDataURL('image/png'); } catch { return null; }
    }
  }), [cvRef, img, zoom, pan, strokes, current]);

  useEffect(() => {
    const im = new Image();
    im.onload = () => setImg(im);
    im.src = imgSrc;
  }, [imgSrc]);

  useEffect(() => { redraw(); }, [img, zoom, pan, strokes, current]);

  function canvasToWorld(p: Pt): Pt {
    if (!img) return { x: 0, y: 0 };
    return { x: (p.x - pan.x) / (img.width * zoom), y: (p.y - pan.y) / (img.height * zoom) };
  }
  function worldToCanvas(p: Pt): Pt {
    if (!img) return { x: 0, y: 0 };
    return { x: p.x * img.width * zoom + pan.x, y: p.y * img.height * zoom + pan.y };
  }

  function redraw() {
    const cv = cvRef.current; if (!cv) return;
    const ctx = cv.getContext('2d')!;
    ctx.clearRect(0, 0, W, H);
    if (img) ctx.drawImage(img, pan.x, pan.y, img.width * zoom, img.height * zoom);

    const draw = (line: Pt[]) => {
      if (!line.length) return;
      ctx.lineWidth = 3; ctx.strokeStyle = '#d33'; ctx.beginPath();
      line.forEach((p, i) => {
        const c = worldToCanvas(p);
        if (i === 0) ctx.moveTo(c.x, c.y); else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
    };

    strokes.forEach(draw);
    if (mode === 'draw') draw(current);
  }

  function onMouseMove(e: React.MouseEvent) {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const cvsPt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    const pt = canvasToWorld(cvsPt);
    onPos && onPos(pt);

    if (mode === 'draw') {
      if (e.buttons === 1) setCurrent((prev) => [...prev, pt]);
    } else if (mode === 'erase') {
      if (erasing) eraseAtCanvasPoint(cvsPt);
    }
  }

  function onMouseDown(e: React.MouseEvent) {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const cvsPt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    const pt = canvasToWorld(cvsPt);

    if (mode === 'draw') {
      setCurrent([pt]);
    } else if (mode === 'erase') {
      setErasing(true);
      eraseAtCanvasPoint(cvsPt);
    }
  }

  function onMouseUp() {
    if (mode === 'draw') {
      if (current.length) {
        setStrokes((prev) => [...prev, current]);
        onDrawEnd(current);
      }
      setCurrent([]);
    } else if (mode === 'erase') {
      setErasing(false);
    }
  }

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    setZoom((z) => Math.max(0.5, Math.min(4, z * (e.deltaY < 0 ? 1.1 : 0.9))));
  }

  function eraseAtCanvasPoint(cvsPt: Pt) {
    const newStrokes: Pt[][] = [];
    for (const line of strokes) {
      let segment: Pt[] = [];
      for (let i = 0; i < line.length; i++) {
        const p = line[i];
        const pc = worldToCanvas(p);
        const dx = pc.x - cvsPt.x;
        const dy = pc.y - cvsPt.y;
        const dist = Math.hypot(dx, dy);
        const keep = dist > ERASE_PX;

        if (keep) {
          segment.push(p);
        } else {
          if (segment.length >= 2) newStrokes.push(segment);
          segment = [];
        }
      }
      if (segment.length >= 2) newStrokes.push(segment);
    }
    setStrokes(newStrokes);
  }

  return (
    <div className="panel">
      <canvas
        ref={cvRef}
        width={W}
        height={H}
        onMouseMove={onMouseMove}
        onMouseDown={onMouseDown}
        onMouseUp={onMouseUp}
        onWheel={onWheel}
      />
    </div>
  );
});

export default CanvasBoard;
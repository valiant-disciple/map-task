import React, { useEffect, useRef, useState } from 'react';

// Define types locally if not importing from types.ts to minimize dependencies
interface Point { x: number; y: number; t: number; }
interface Stroke { points: Point[]; color: string; width: number; }

interface MapViewerProps {
  src: string;
  isInteractive?: boolean;
  onStroke?: (stroke: Stroke) => void;
  onCursorMove?: (x: number, y: number) => void;
  remoteStrokes?: Stroke[];
}

export default function MapViewer({ src, isInteractive = true, onStroke, onCursorMove, remoteStrokes = [] }: MapViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const currentPath = useRef<Point[]>([]);

  // Store local strokes to satisfy "persistence" even if parent doesn't echo them back
  const [localStrokes, setLocalStrokes] = useState<Stroke[]>([]);

  // Function to draw a single stroke
  function drawStroke(ctx: CanvasRenderingContext2D, stroke: Stroke) {
    if (stroke.points.length < 2) return;
    ctx.beginPath();
    ctx.lineWidth = stroke.width;
    ctx.strokeStyle = stroke.color;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
    for (let i = 1; i < stroke.points.length; i++) {
      ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
    }
    ctx.stroke();
  }

  // Redraw Everything (Remote + Local) whenever they change or window resizes
  const redraw = () => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, cvs.width, cvs.height);

    // Draw remote strokes
    remoteStrokes.forEach(s => drawStroke(ctx, s));
    // Draw local strokes
    localStrokes.forEach(s => drawStroke(ctx, s));
  };

  useEffect(() => {
    redraw();
  }, [remoteStrokes, localStrokes]);

  // Handle Resize
  useEffect(() => {
    const ob = new ResizeObserver(() => {
      if (containerRef.current && canvasRef.current) {
        const img = containerRef.current.querySelector('img');
        if (img) {
          // Match canvas size to image exact size
          canvasRef.current.width = img.clientWidth;
          canvasRef.current.height = img.clientHeight;
          redraw();
        }
      }
    });
    if (containerRef.current) ob.observe(containerRef.current);
    return () => ob.disconnect();
  }, [src]); // Re-observe if src (and thus image) changes


  // Helper to get coordinates relative to canvas
  function getCoords(e: React.MouseEvent | React.TouchEvent | MouseEvent | TouchEvent): Point {
    const cvs = canvasRef.current;
    if (!cvs) return { x: 0, y: 0, t: Date.now() };

    const rect = cvs.getBoundingClientRect();
    let clientX = 0;
    let clientY = 0;

    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = (e as React.MouseEvent).clientX;
      clientY = (e as React.MouseEvent).clientY;
    }

    return {
      x: (clientX - rect.left) * (cvs.width / rect.width),
      y: (clientY - rect.top) * (cvs.height / rect.height),
      t: Date.now()
    };
  }

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isInteractive || !canvasRef.current) return;
    if (e.cancelable) e.preventDefault();
    setIsDrawing(true);
    const p = getCoords(e);
    currentPath.current = [p];
  };

  const moveDraw = (e: React.MouseEvent | React.TouchEvent) => {
    const p = getCoords(e);
    // Report cursor position
    if (onCursorMove) onCursorMove(Math.round(p.x), Math.round(p.y));

    if (!isInteractive || !isDrawing || !canvasRef.current) return;
    // e.preventDefault(); // handled in startDraw mostly, but good here too

    currentPath.current.push(p);

    // Immediate draw feedback
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      const prev = currentPath.current[currentPath.current.length - 2];
      if (prev) {
        ctx.beginPath();
        // USER REQUEST: RED COLOR
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#ff0000';
        ctx.lineCap = 'round';
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
      }
    }
  };

  const endDraw = () => {
    if (!isInteractive || !isDrawing) return;
    setIsDrawing(false);

    if (currentPath.current.length > 1) {
      // Create the stroke object
      const newStroke: Stroke = {
        points: [...currentPath.current],
        color: '#ff0000', // RED
        width: 3
      };
      // Add to local history so it persists
      setLocalStrokes(prev => [...prev, newStroke]);
      if (onStroke) onStroke(newStroke);
    }
    currentPath.current = [];
  };

  return (
    <div ref={containerRef} style={{ position: 'relative', display: 'inline-block' }}>
      <img
        src={src}
        alt="Map"
        className="map-img"
        style={{ display: 'block', pointerEvents: 'none', userSelect: 'none', maxWidth: '100%' }}
        onLoad={() => {
          // Trigger resize logic once image loads
          if (containerRef.current && canvasRef.current) {
            const img = containerRef.current.querySelector('img');
            if (img) {
              canvasRef.current.width = img.clientWidth;
              canvasRef.current.height = img.clientHeight;
              redraw();
            }
          }
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          touchAction: 'none',
          cursor: isInteractive ? 'crosshair' : 'default',
          background: 'transparent',
          pointerEvents: isInteractive ? 'auto' : 'none'
        }}
        onMouseDown={startDraw}
        onMouseMove={moveDraw}
        onMouseUp={endDraw}
        onMouseLeave={endDraw}
        onTouchStart={startDraw}
        onTouchMove={moveDraw}
        onTouchEnd={endDraw}
      />
    </div>
  );
}
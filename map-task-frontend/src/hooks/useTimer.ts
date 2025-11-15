import { useEffect, useRef, useState } from 'react';

export function useTimer(initial: number, onEnd?: () => void) {
  const [remain, setRemain] = useState(initial);
  const ref = useRef<number | null>(null);

  useEffect(() => {
    // reset remaining whenever initial changes
    setRemain(initial);
    if (ref.current) window.clearInterval(ref.current);
    ref.current = window.setInterval(() => {
      setRemain((r) => {
        if (r <= 1) {
          if (ref.current) window.clearInterval(ref.current);
          onEnd && onEnd();
          return 0;
        }
        return r - 1;
      });
    }, 1000);
    return () => { if (ref.current) window.clearInterval(ref.current); };
  }, [initial, onEnd]);

  return { remain, setRemain };
}
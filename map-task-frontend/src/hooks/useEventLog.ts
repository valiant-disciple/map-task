import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EventRecord } from '../types';

const LS_KEY = 'event_log_backup';
const SAVE_INTERVAL_MS = 5000; // auto-save every 5 seconds

function loadBackup(): EventRecord[] {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch { /* ignore corrupt data */ }
  return [];
}

function saveBackup(events: EventRecord[]) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(events));
  } catch { /* storage full — ignore */ }
}

export function useEventLog() {
  const [events, setEvents] = useState<EventRecord[]>(() => loadBackup());
  const eventsRef = useRef(events);
  eventsRef.current = events;

  const add = useCallback((type: string, payload?: any, role?: 'director' | 'matcher') => {
    setEvents(prev => [...prev, { t: Date.now(), type, role, payload }]);
  }, []);

  const addRaw = useCallback((rec: EventRecord) => {
    setEvents(prev => [...prev, rec]);
  }, []);

  const clearLog = useCallback(() => {
    setEvents([]);
    try { localStorage.removeItem(LS_KEY); } catch { /* ignore */ }
  }, []);

  // Auto-save to localStorage periodically
  useEffect(() => {
    const id = window.setInterval(() => {
      saveBackup(eventsRef.current);
    }, SAVE_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, []);

  // Also save on page unload
  useEffect(() => {
    const handleUnload = () => saveBackup(eventsRef.current);
    window.addEventListener('beforeunload', handleUnload);
    return () => window.removeEventListener('beforeunload', handleUnload);
  }, []);

  return useMemo(() => ({ events, add, addRaw, clearLog }), [events, add, addRaw, clearLog]);
}

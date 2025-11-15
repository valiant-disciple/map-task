import { useCallback, useMemo, useState } from 'react';
import type { EventRecord } from '../types';

export function useEventLog() {
  const [events, setEvents] = useState<EventRecord[]>([]);

  const add = useCallback((type: string, payload?: any, role?: 'director' | 'matcher') => {
    setEvents(prev => [...prev, { t: Date.now(), type, role, payload }]);
  }, []);

  const addRaw = useCallback((rec: EventRecord) => {
    setEvents(prev => [...prev, rec]);
  }, []);

  return useMemo(() => ({ events, add, addRaw }), [events, add, addRaw]);
}
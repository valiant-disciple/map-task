export const storage = {
    set<T>(k: string, v: T) { localStorage.setItem(k, JSON.stringify(v)); },
    get<T>(k: string, def: T): T { const r = localStorage.getItem(k); return r ? JSON.parse(r) as T : def; },
    del(k: string) { localStorage.removeItem(k); }
  };
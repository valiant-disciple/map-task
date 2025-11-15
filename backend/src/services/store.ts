import type { HrRecord } from '../types/hr.js';

type Ring<T> = {
  push: (item: T) => void;
  all: () => T[];
  latest: () => T | null;
};

function createRing<T>(capacity: number): Ring<T> {
  const data: T[] = [];
  return {
    push(item: T) {
      if (data.length >= capacity) data.shift();
      data.push(item);
    },
    all() {
      return data.slice().reverse();
    },
    latest() {
      return data.length ? data[data.length - 1] : null;
    }
  };
}

export class HrStore {
  private global = createRing<HrRecord>(1000);
  private perDevice = new Map<string, Ring<HrRecord>>();

  add(rec: HrRecord) {
    this.global.push(rec);
    if (!this.perDevice.has(rec.deviceId)) {
      this.perDevice.set(rec.deviceId, createRing<HrRecord>(500));
    }
    this.perDevice.get(rec.deviceId)!.push(rec);
  }

  latest(deviceId?: string): HrRecord | null {
    if (deviceId) return this.perDevice.get(deviceId)?.latest() ?? null;
    return this.global.latest();
  }

  history(deviceId?: string, limit = 100): HrRecord[] {
    const list = deviceId ? this.perDevice.get(deviceId)?.all() ?? [] : this.global.all();
    return list.slice(0, limit);
  }
}

export const store = new HrStore();



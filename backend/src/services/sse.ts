import type { Response } from 'express';

type Client = {
  id: number;
  res: Response;
};

export class SseHub<T> {
  private clients = new Map<number, Client>();
  private seq = 0;

  addClient(res: Response): number {
    const id = ++this.seq;
    this.clients.set(id, { id, res });
    return id;
  }

  removeClient(id: number) {
    this.clients.delete(id);
  }

  broadcast(event: string, data: T) {
    const payload = `event: ${event}\n` + `data: ${JSON.stringify(data)}\n\n`;
    for (const c of this.clients.values()) {
      c.res.write(payload);
    }
  }
}



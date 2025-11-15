import { z } from 'zod';

export const HrPayloadSchema = z.object({
  deviceId: z.string().min(1),
  ts: z.number().int().positive().optional(),
  hr: z.number().int().nonnegative(),
  ibi: z.array(z.number().int().nonnegative()).optional(),
  meta: z.record(z.unknown()).optional()
});

export type HrPayload = z.infer<typeof HrPayloadSchema>;

export type HrRecord = {
  deviceId: string;
  ts: number;
  hr: number;
  ibi: number[];
  meta?: Record<string, unknown>;
};



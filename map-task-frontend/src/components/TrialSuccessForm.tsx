import React, { useState } from 'react';
import Modal from './Modal';

/**
 * Trial Success Form — shown to the Director at the end of each data trial,
 * before the NASA-TLX. Captures:
 *   1. Whether the target destination was reached (binary)
 *   2. Confidence in the path taken (1–7 Likert)
 *   3. Optional brief note
 */

export interface TrialSuccessData {
  targetReached: boolean | null;
  pathConfidence: number;
  note: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: TrialSuccessData) => void;
  trialIndex: number;
}

export default function TrialSuccessForm({ open, onClose, onSubmit, trialIndex }: Props) {
  const [targetReached, setTargetReached] = useState<boolean | null>(null);
  const [pathConfidence, setPathConfidence] = useState(4);
  const [note, setNote] = useState('');

  const canSubmit = targetReached !== null;

  const handleSubmit = () => {
    if (!canSubmit) return;
    onSubmit({ targetReached, pathConfidence, note: note.trim() });
    // Reset for next use
    setTargetReached(null);
    setPathConfidence(4);
    setNote('');
  };

  return (
    <Modal open={open} onClose={onClose}>
      <h3 style={{ marginTop: 0, marginBottom: 4 }}>Trial {trialIndex} — Outcome</h3>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Please evaluate the result of this trial before proceeding to the workload survey.
      </p>

      {/* Q1: Target Reached */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 8 }}>
          Did the Matcher reach the target destination?
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          {([
            { value: true, label: '✅ Yes — Reached', color: '#e8f5e9', border: '#66bb6a' },
            { value: false, label: '❌ No — Did Not Reach', color: '#ffebee', border: '#ef5350' },
          ] as const).map(opt => (
            <button
              key={String(opt.value)}
              onClick={() => setTargetReached(opt.value)}
              style={{
                flex: 1,
                padding: '10px 16px',
                fontSize: 14,
                fontWeight: targetReached === opt.value ? 700 : 400,
                backgroundColor: targetReached === opt.value ? opt.color : '#f5f5f5',
                border: `2px solid ${targetReached === opt.value ? opt.border : '#ddd'}`,
                borderRadius: 8,
                cursor: 'pointer',
                color: '#333',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Q2: Path Confidence */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
          How confident are you that the path taken was correct?
        </div>
        <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
          Even if the destination was reached, the path may have included unnecessary detours or wrong turns.
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#aaa', minWidth: 60, textAlign: 'right' }}>
            Not at all
          </span>
          <input
            type="range"
            min={1}
            max={7}
            step={1}
            value={pathConfidence}
            onChange={e => setPathConfidence(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11, color: '#aaa', minWidth: 60 }}>
            Completely
          </span>
          <span style={{ fontWeight: 600, minWidth: 20, textAlign: 'center' }}>
            {pathConfidence}
          </span>
        </div>
      </div>

      {/* Q3: Optional Note */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
          Notes (optional)
        </div>
        <textarea
          value={note}
          onChange={e => setNote(e.target.value)}
          placeholder="Any observations about this trial (e.g., confusion at a specific landmark, communication breakdown)..."
          style={{
            width: '100%',
            minHeight: 60,
            padding: 8,
            borderRadius: 6,
            border: '1px solid #ddd',
            fontSize: 13,
            resize: 'vertical',
          }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          style={{
            padding: '8px 24px',
            opacity: canSubmit ? 1 : 0.5,
            cursor: canSubmit ? 'pointer' : 'not-allowed',
          }}
        >
          Continue to Workload Survey →
        </button>
      </div>
    </Modal>
  );
}

import React, { useState } from 'react';
import Modal from './Modal';

/**
 * NASA Task Load Index (Raw TLX) — Hart & Staveland, 1988
 * Using the Raw TLX variant (no pairwise weighting) as recommended by Hart (2006).
 * 21-point scale (0–100, step 5) per the original paper.
 */

interface TLXDimension {
  key: string;
  title: string;
  description: string;
  lowLabel: string;
  highLabel: string;
}

const dimensions: TLXDimension[] = [
  {
    key: 'mental',
    title: 'Mental Demand',
    description: 'How much mental and perceptual activity was required (e.g., thinking, deciding, calculating, remembering, looking, searching)? Was the task easy or demanding, simple or complex?',
    lowLabel: 'Low',
    highLabel: 'High',
  },
  {
    key: 'physical',
    title: 'Physical Demand',
    description: 'How much physical activity was required (e.g., pushing, pulling, turning, controlling, activating)? Was the task easy or demanding, slow or brisk, slack or strenuous?',
    lowLabel: 'Low',
    highLabel: 'High',
  },
  {
    key: 'temporal',
    title: 'Temporal Demand',
    description: 'How much time pressure did you feel due to the rate or pace at which tasks or task elements occurred? Was the pace slow and leisurely or rapid and frantic?',
    lowLabel: 'Low',
    highLabel: 'High',
  },
  {
    key: 'performance',
    title: 'Performance',
    description: 'How successful do you think you were in accomplishing the goals of the task set by the experimenter? How satisfied were you with your performance?',
    lowLabel: 'Perfect',
    highLabel: 'Failure',
  },
  {
    key: 'effort',
    title: 'Effort',
    description: 'How hard did you have to work (mentally and physically) to accomplish your level of performance?',
    lowLabel: 'Low',
    highLabel: 'High',
  },
  {
    key: 'frustration',
    title: 'Frustration',
    description: 'How insecure, discouraged, irritated, stressed, and annoyed versus secure, gratified, content, relaxed, and complacent did you feel during the task?',
    lowLabel: 'Low',
    highLabel: 'High',
  },
];

type TLXValues = { [key: string]: number };

export default function TLXForm({
  open,
  onClose,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (v: TLXValues) => void;
}) {
  // All start at 50 (neutral midpoint) — no bias
  const [values, setValues] = useState<TLXValues>(
    Object.fromEntries(dimensions.map(d => [d.key, 50]))
  );

  const update = (key: string, val: number) => {
    setValues(prev => ({ ...prev, [key]: val }));
  };

  const handleSubmit = () => {
    onSubmit(values);
    // Reset for next use
    setValues(Object.fromEntries(dimensions.map(d => [d.key, 50])));
  };

  return (
    <Modal open={open} onClose={onClose}>
      <h3 style={{ marginTop: 0, marginBottom: 4 }}>NASA Task Load Index</h3>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Rate your experience for the task you just completed. Move each slider to indicate your assessment.
      </p>

      <div style={{ maxHeight: '60vh', overflowY: 'auto', paddingRight: 8 }}>
        {dimensions.map(dim => (
          <div
            key={dim.key}
            style={{
              marginBottom: 18,
              padding: '10px 12px',
              border: '1px solid #eee',
              borderRadius: 8,
              backgroundColor: '#fafafa',
            }}
          >
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 2 }}>
              {dim.title}
            </div>
            <div style={{ fontSize: 12, color: '#666', marginBottom: 8, lineHeight: 1.4 }}>
              {dim.description}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11, color: '#888', minWidth: 44, textAlign: 'right' }}>
                {dim.lowLabel}
              </span>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={values[dim.key]}
                onChange={e => update(dim.key, Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ fontSize: 11, color: '#888', minWidth: 44 }}>
                {dim.highLabel}
              </span>
              <span
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  minWidth: 30,
                  textAlign: 'right',
                }}
              >
                {values[dim.key]}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
        <button onClick={handleSubmit} style={{ padding: '8px 24px' }}>
          Submit
        </button>
      </div>
    </Modal>
  );
}

import React, { useState } from 'react';
import Modal from './Modal';

/**
 * Perceived Shared Mental Model (PSMM) Questionnaire
 *
 * Adapted from:
 * - Mathieu et al. (2000) — Task & Team Mental Model distinction
 * - Johnson et al. (2007) — Perceived mental model similarity
 * - Cannon-Bowers et al. (1993) — SMM framework
 *
 * Adapted specifically for the Map Task (Director/Matcher dyadic navigation).
 * 8 items on a 7-point Likert scale (1 = Strongly Disagree, 7 = Strongly Agree).
 */

interface SMMItem {
  factor: 'task_smm' | 'team_smm';
  text: string;
}

const items: SMMItem[] = [
  // Task SMM — shared understanding of the route, landmarks, obstacles, positions
  { factor: 'task_smm', text: 'My partner and I had a shared understanding of the route to follow.' },
  { factor: 'task_smm', text: 'We agreed on which landmarks were important reference points.' },
  { factor: 'task_smm', text: 'We had the same understanding of where the obstacles were and how to navigate around them.' },
  { factor: 'task_smm', text: 'We agreed on the current position on the map throughout the task.' },

  // Team SMM — shared understanding of roles, communication, prediction, coordination
  { factor: 'team_smm', text: 'I could anticipate what my partner would say or do next.' },
  { factor: 'team_smm', text: 'We had a clear shared understanding of each other\'s role.' },
  { factor: 'team_smm', text: 'We communicated effectively about the map and directions.' },
  { factor: 'team_smm', text: 'When misunderstandings arose, we resolved them quickly.' },
];

const scaleLabels: Record<number, string> = {
  1: 'Strongly Disagree',
  2: 'Disagree',
  3: 'Somewhat Disagree',
  4: 'Neutral',
  5: 'Somewhat Agree',
  6: 'Agree',
  7: 'Strongly Agree',
};

export default function PSMMForm({
  open,
  onClose,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (rows: { factor: string; itemNum: number; text: string; value: number }[]) => void;
}) {
  const [values, setValues] = useState<number[]>(Array(items.length).fill(4));

  const handleSubmit = () => {
    const result = values.map((v, i) => ({
      factor: items[i].factor,
      itemNum: i + 1,
      text: items[i].text,
      value: v,
    }));
    onSubmit(result);
    // Reset for next use
    setValues(Array(items.length).fill(4));
  };

  return (
    <Modal open={open} onClose={onClose}>
      <h3 style={{ marginTop: 0, marginBottom: 4 }}>Shared Understanding</h3>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Rate how much you agree with each statement about you and your partner during this trial.
      </p>

      <div className="psmm">
        {items.map((item, i) => {
          const isNewSection = i === 0 || items[i].factor !== items[i - 1].factor;
          return (
            <React.Fragment key={i}>
              {isNewSection && (
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: '#888',
                    textTransform: 'uppercase',
                    letterSpacing: 0.5,
                    marginTop: i === 0 ? 0 : 16,
                    marginBottom: 8,
                    borderBottom: '1px solid #eee',
                    paddingBottom: 4,
                  }}
                >
                  {item.factor === 'task_smm' ? 'Task Understanding' : 'Team Coordination'}
                </div>
              )}
              <div
                style={{
                  marginBottom: 14,
                  padding: '8px 10px',
                  border: '1px solid #eee',
                  borderRadius: 6,
                  backgroundColor: '#fafafa',
                }}
              >
                <div style={{ fontSize: 13, marginBottom: 8, lineHeight: 1.4 }}>
                  {i + 1}. {item.text}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 10, color: '#aaa', minWidth: 50, textAlign: 'right' }}>
                    Strongly Disagree
                  </span>
                  <input
                    type="range"
                    min={1}
                    max={7}
                    step={1}
                    value={values[i]}
                    onChange={e => {
                      const next = [...values];
                      next[i] = Number(e.target.value);
                      setValues(next);
                    }}
                    style={{ flex: 1 }}
                  />
                  <span style={{ fontSize: 10, color: '#aaa', minWidth: 50 }}>
                    Strongly Agree
                  </span>
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      minWidth: 20,
                      textAlign: 'center',
                      color: '#333',
                    }}
                  >
                    {values[i]}
                  </span>
                </div>
                <div style={{ fontSize: 10, color: '#aaa', textAlign: 'center', marginTop: 2 }}>
                  {scaleLabels[values[i]]}
                </div>
          </div>
            </React.Fragment>
          );
        })}
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
        <button onClick={handleSubmit} style={{ padding: '8px 24px' }}>
          Submit
        </button>
      </div>
    </Modal>
  );
}

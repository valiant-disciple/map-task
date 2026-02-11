import React, { useState } from 'react';
import Modal from './Modal';

/**
 * Post-experiment debrief form — shown on the End page after all trials.
 * Captures open-ended reflections, strategy descriptions, overall difficulty,
 * and communication quality ratings.
 */

export interface DebriefData {
  overallDifficulty: number;   // 1–7
  communicationQuality: number; // 1–7
  strategy: string;            // free text: how did you approach the task?
  challenges: string;          // free text: what was hardest?
  feedback: string;            // free text: any issues/suggestions?
  wouldChangeApproach: boolean;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: DebriefData) => void;
}

export default function DebriefForm({ open, onClose, onSubmit }: Props) {
  const [d, setD] = useState<DebriefData>({
    overallDifficulty: 4,
    communicationQuality: 4,
    strategy: '',
    challenges: '',
    feedback: '',
    wouldChangeApproach: false,
  });

  const upd = <K extends keyof DebriefData>(key: K, val: DebriefData[K]) =>
    setD(prev => ({ ...prev, [key]: val }));

  const handleSubmit = () => {
    onSubmit(d);
  };

  const labelStyle: React.CSSProperties = { fontWeight: 600, fontSize: 14, marginBottom: 4, display: 'block' };
  const hintStyle: React.CSSProperties = { fontSize: 12, color: '#888', marginBottom: 8 };
  const textareaStyle: React.CSSProperties = {
    width: '100%', minHeight: 60, padding: 8, borderRadius: 6,
    border: '1px solid #ddd', fontSize: 13, resize: 'vertical',
  };

  return (
    <Modal open={open} onClose={onClose}>
      <h3 style={{ marginTop: 0, marginBottom: 4 }}>Post-Experiment Debrief</h3>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Thank you for completing the experiment! Please answer a few final questions.
      </p>

      {/* Overall Difficulty */}
      <div style={{ marginBottom: 16 }}>
        <label style={labelStyle}>Overall, how difficult was the task? (1 = Very Easy, 7 = Very Difficult)</label>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#aaa', minWidth: 55, textAlign: 'right' }}>Very Easy</span>
          <input type="range" min={1} max={7} value={d.overallDifficulty}
            onChange={e => upd('overallDifficulty', Number(e.target.value))} style={{ flex: 1 }} />
          <span style={{ fontSize: 11, color: '#aaa', minWidth: 70 }}>Very Difficult</span>
          <span style={{ fontWeight: 600, minWidth: 18, textAlign: 'center' }}>{d.overallDifficulty}</span>
        </div>
      </div>

      {/* Communication Quality */}
      <div style={{ marginBottom: 16 }}>
        <label style={labelStyle}>How would you rate the quality of communication with your partner?</label>
        <div style={hintStyle}>1 = Very Poor, 7 = Excellent</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#aaa', minWidth: 55, textAlign: 'right' }}>Very Poor</span>
          <input type="range" min={1} max={7} value={d.communicationQuality}
            onChange={e => upd('communicationQuality', Number(e.target.value))} style={{ flex: 1 }} />
          <span style={{ fontSize: 11, color: '#aaa', minWidth: 70 }}>Excellent</span>
          <span style={{ fontWeight: 600, minWidth: 18, textAlign: 'center' }}>{d.communicationQuality}</span>
        </div>
      </div>

      {/* Strategy */}
      <div style={{ marginBottom: 16 }}>
        <label style={labelStyle}>What strategy did you use to describe/follow the route?</label>
        <div style={hintStyle}>e.g., landmark-based instructions, step-by-step, compass directions, etc.</div>
        <textarea value={d.strategy} onChange={e => upd('strategy', e.target.value)}
          placeholder="Describe your approach..." style={textareaStyle} />
      </div>

      {/* Challenges */}
      <div style={{ marginBottom: 16 }}>
        <label style={labelStyle}>What was the most challenging part?</label>
        <textarea value={d.challenges} onChange={e => upd('challenges', e.target.value)}
          placeholder="e.g., confusing landmarks, miscommunication about directions..." style={textareaStyle} />
      </div>

      {/* Would change approach */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={d.wouldChangeApproach}
            onChange={e => upd('wouldChangeApproach', e.target.checked)} />
          If I did this again, I would change my communication strategy.
        </label>
      </div>

      {/* General feedback */}
      <div style={{ marginBottom: 16 }}>
        <label style={labelStyle}>Any other comments or feedback? (optional)</label>
        <textarea value={d.feedback} onChange={e => upd('feedback', e.target.value)}
          placeholder="Technical issues, suggestions, observations..." style={textareaStyle} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button onClick={handleSubmit} style={{ padding: '8px 24px' }}>
          Submit & Finish
        </button>
      </div>
    </Modal>
  );
}

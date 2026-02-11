import React, { useState } from 'react';

/**
 * Pre-experiment demographics form — collected once per participant in the Lobby.
 * Captures minimal info needed for analysis: age, gender, handedness,
 * native language, familiarity with partner, and prior map-task experience.
 */

export interface Demographics {
  age: string;
  gender: string;
  handedness: 'left' | 'right' | 'ambidextrous';
  nativeLanguage: string;
  englishFluency: number; // 1–7
  partnerFamiliarity: number; // 1–7 (1 = strangers, 7 = close friends)
  priorMapTask: boolean;
  hearingDifficulties: boolean;
  visionCorrected: boolean; // for future gaze tracking
  notes: string;
}

const DEFAULTS: Demographics = {
  age: '',
  gender: '',
  handedness: 'right',
  nativeLanguage: '',
  englishFluency: 5,
  partnerFamiliarity: 1,
  priorMapTask: false,
  hearingDifficulties: false,
  visionCorrected: false,
  notes: '',
};

interface Props {
  onSubmit: (data: Demographics) => void;
}

export default function DemographicsForm({ onSubmit }: Props) {
  const [d, setD] = useState<Demographics>({ ...DEFAULTS });
  const [errors, setErrors] = useState<string[]>([]);

  const upd = <K extends keyof Demographics>(key: K, val: Demographics[K]) =>
    setD(prev => ({ ...prev, [key]: val }));

  const validate = (): string[] => {
    const errs: string[] = [];
    if (!d.age || isNaN(Number(d.age)) || Number(d.age) < 18 || Number(d.age) > 100)
      errs.push('Please enter a valid age (18–100).');
    if (!d.gender.trim()) errs.push('Please select or enter your gender.');
    if (!d.nativeLanguage.trim()) errs.push('Please enter your native language.');
    return errs;
  };

  const handleSubmit = () => {
    const errs = validate();
    if (errs.length > 0) {
      setErrors(errs);
      return;
    }
    setErrors([]);
    onSubmit(d);
  };

  const fieldStyle: React.CSSProperties = { marginBottom: 14 };
  const labelStyle: React.CSSProperties = { display: 'block', fontWeight: 600, fontSize: 13, marginBottom: 4 };
  const hintStyle: React.CSSProperties = { fontSize: 11, color: '#999', marginTop: 2 };
  const inputStyle: React.CSSProperties = { width: '100%', padding: '6px 8px', borderRadius: 4, border: '1px solid #ccc', fontSize: 13 };
  const selectStyle: React.CSSProperties = { ...inputStyle, background: '#fff' };

  return (
    <div style={{ maxWidth: 520, margin: '0 auto' }}>
      <h3 style={{ marginTop: 0, marginBottom: 4 }}>Participant Demographics</h3>
      <p style={{ color: '#666', fontSize: 13, marginBottom: 16 }}>
        Please fill in the following before proceeding. This information is used for analysis only.
      </p>

      {errors.length > 0 && (
        <div style={{ background: '#ffebee', border: '1px solid #ef5350', borderRadius: 6, padding: '8px 12px', marginBottom: 14 }}>
          {errors.map((e, i) => <div key={i} style={{ fontSize: 12, color: '#c62828' }}>{e}</div>)}
        </div>
      )}

      {/* Age */}
      <div style={fieldStyle}>
        <label style={labelStyle}>Age *</label>
        <input
          type="number"
          min={18} max={100}
          value={d.age}
          onChange={e => upd('age', e.target.value)}
          placeholder="e.g. 24"
          style={{ ...inputStyle, width: 120 }}
        />
      </div>

      {/* Gender */}
      <div style={fieldStyle}>
        <label style={labelStyle}>Gender *</label>
        <select
          value={d.gender}
          onChange={e => upd('gender', e.target.value)}
          style={selectStyle}
        >
          <option value="">— Select —</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
          <option value="non-binary">Non-binary</option>
          <option value="prefer-not-to-say">Prefer not to say</option>
          <option value="other">Other</option>
        </select>
      </div>

      {/* Handedness */}
      <div style={fieldStyle}>
        <label style={labelStyle}>Handedness</label>
        <div style={{ display: 'flex', gap: 16 }}>
          {(['right', 'left', 'ambidextrous'] as const).map(h => (
            <label key={h} style={{ fontSize: 13, cursor: 'pointer' }}>
              <input
                type="radio"
                name="handedness"
                value={h}
                checked={d.handedness === h}
                onChange={() => upd('handedness', h)}
                style={{ marginRight: 4 }}
              />
              {h.charAt(0).toUpperCase() + h.slice(1)}
            </label>
          ))}
        </div>
      </div>

      {/* Native Language */}
      <div style={fieldStyle}>
        <label style={labelStyle}>Native Language *</label>
        <input
          type="text"
          value={d.nativeLanguage}
          onChange={e => upd('nativeLanguage', e.target.value)}
          placeholder="e.g. English, Hindi, French"
          style={inputStyle}
        />
      </div>

      {/* English Fluency */}
      <div style={fieldStyle}>
        <label style={labelStyle}>English Fluency (1 = basic, 7 = native/fluent)</label>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#aaa' }}>Basic</span>
          <input
            type="range" min={1} max={7} step={1}
            value={d.englishFluency}
            onChange={e => upd('englishFluency', Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11, color: '#aaa' }}>Native</span>
          <span style={{ fontWeight: 600, minWidth: 18, textAlign: 'center' }}>{d.englishFluency}</span>
        </div>
      </div>

      {/* Partner Familiarity */}
      <div style={fieldStyle}>
        <label style={labelStyle}>Familiarity with Partner (1 = strangers, 7 = close friends)</label>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: '#aaa' }}>Strangers</span>
          <input
            type="range" min={1} max={7} step={1}
            value={d.partnerFamiliarity}
            onChange={e => upd('partnerFamiliarity', Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11, color: '#aaa' }}>Close</span>
          <span style={{ fontWeight: 600, minWidth: 18, textAlign: 'center' }}>{d.partnerFamiliarity}</span>
        </div>
      </div>

      {/* Checkboxes */}
      <div style={fieldStyle}>
        <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={d.priorMapTask} onChange={e => upd('priorMapTask', e.target.checked)} />
          I have participated in a Map Task experiment before
        </label>
        <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={d.hearingDifficulties} onChange={e => upd('hearingDifficulties', e.target.checked)} />
          I have hearing difficulties
        </label>
        <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={d.visionCorrected} onChange={e => upd('visionCorrected', e.target.checked)} />
          I wear corrective lenses (glasses/contacts)
        </label>
        <div style={hintStyle}>Vision info will be used for future eye-tracking calibration.</div>
      </div>

      {/* Notes */}
      <div style={fieldStyle}>
        <label style={labelStyle}>Additional Notes (optional)</label>
        <textarea
          value={d.notes}
          onChange={e => upd('notes', e.target.value)}
          placeholder="Anything the experimenter should know..."
          style={{ ...inputStyle, minHeight: 50, resize: 'vertical' }}
        />
      </div>

      <button
        onClick={handleSubmit}
        style={{ padding: '8px 24px', fontSize: 14, marginTop: 4 }}
      >
        Continue →
      </button>
    </div>
  );
}

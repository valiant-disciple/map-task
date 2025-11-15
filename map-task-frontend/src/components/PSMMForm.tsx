import React, { useState } from 'react';
import Modal from './Modal';

const items = [
  { factor:'equipment', text:'We agree on which tools/displays are key to this task.' },
  { factor:'equipment', text:'We read instruments/sensors in the same way.' },
  { factor:'equipment', text:'We know which resources to use first vs reserve.' },
  { factor:'equipment', text:'We agree on substitutes when tools fail.' },
  { factor:'execution', text:'We agree on the steps needed to execute the task.' },
  { factor:'execution', text:'We share criteria for switching strategies.' },
  { factor:'execution', text:'We share acceptable error tolerances.' },
  { factor:'execution', text:'We agree how to verify progress/completion.' },
  { factor:'interaction', text:'We share who communicates what, to whom, and when.' },
  { factor:'interaction', text:'We agree on confirmation/read-back practices.' },
  { factor:'interaction', text:'We use the same labels/terms for task elements.' },
  { factor:'interaction', text:'We agree how to handle misunderstandings/repairs.' },
  { factor:'composition', text:'We understand each other’s roles/responsibilities.' },
  { factor:'composition', text:'We know each other’s strengths/limitations.' },
  { factor:'composition', text:'We can anticipate each other’s likely actions.' },
  { factor:'composition', text:'We agree who leads in specific situations.' },
  { factor:'temporal', text:'We agree on time priorities and deadlines.' },
  { factor:'temporal', text:'We share a sense of the appropriate pace.' },
  { factor:'temporal', text:'We agree how long decision points should take.' },
  { factor:'temporal', text:'We agree when to escalate or slow down.' }
] as const;

export default function PSMMForm({ open, onClose, onSubmit }: { open: boolean; onClose: () => void; onSubmit: (rows: {factor:string;itemNum:number;value:number;}[]) => void; }) {
  const [vals, setVals] = useState<number[]>(Array(items.length).fill(4));
  return (
    <Modal open={open} onClose={onClose}>
      <h3>Perceived Shared Mental Models (1–7)</h3>
      <div className="psmm">
        {items.map((it, i) => (
          <div key={i} className="row">
            <label>{i+1}. [{it.factor}] {it.text}</label>
            <input type="range" min={1} max={7} value={vals[i]} onChange={(e)=>{ const next=[...vals]; next[i]=Number(e.target.value); setVals(next); }} />
            <span>{vals[i]}</span>
          </div>
        ))}
      </div>
      <div className="row right">
        <button onClick={()=>onSubmit(vals.map((v,i)=>({ factor: items[i].factor, itemNum:i+1, value:v })))}>Submit</button>
      </div>
    </Modal>
  );
}
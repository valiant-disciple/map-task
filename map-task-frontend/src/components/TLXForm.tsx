import React, { useState } from 'react';
import Modal from './Modal';

type TLX = { mental:number; physical:number; temporal:number; performance:number; effort:number; frustration:number; };

export default function TLXForm({ open, onClose, onSubmit }: { open: boolean; onClose: () => void; onSubmit: (v: TLX) => void; }) {
  const [v, setV] = useState<TLX>({ mental:50, physical:10, temporal:50, performance:50, effort:50, frustration:50 });
  const upd = (k: keyof TLX) => (e: React.ChangeEvent<HTMLInputElement>) => setV({ ...v, [k]: Number(e.target.value) });
  return (
    <Modal open={open} onClose={onClose}>
      <h3>NASA TLX</h3>
      {(['mental','physical','temporal','performance','effort','frustration'] as (keyof TLX)[]).map(k => (
        <div key={k} className="row">
          <label>{k}</label>
          <input type="range" min={0} max={100} value={v[k]} onChange={upd(k)} /><span>{v[k]}</span>
        </div>
      ))}
      <div className="row right"><button onClick={()=>onSubmit(v)}>Submit</button></div>
    </Modal>
  );
}
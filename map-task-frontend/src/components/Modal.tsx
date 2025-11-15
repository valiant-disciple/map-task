import React from 'react';

export default function Modal({ open, onClose, children }: { open: boolean; onClose: () => void; children: React.ReactNode; }) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}
import React from 'react';

export default function Modal({ open, onClose, children, dismissable = false }: { open: boolean; onClose: () => void; children: React.ReactNode; dismissable?: boolean; }) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={dismissable ? onClose : undefined}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}
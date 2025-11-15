import React from 'react';
export default function MapViewer({ src }: { src: string }) {
  return <div className="panel"><img src={src} alt="Director Map" className="map-img" /></div>;
}
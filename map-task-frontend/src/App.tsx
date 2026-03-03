import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Lobby from './pages/Lobby';
import Director from './pages/Director';
import Matcher from './pages/Matcher';
import End from './pages/End';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Lobby />} />
      <Route path="/director" element={<Director />} />
      <Route path="/matcher" element={<Matcher />} />
      <Route path="/end" element={<End />} />
    </Routes>
  );
}
# Director–Matcher Map Task: Multimodal Team Coordination Study

A research platform for studying team coordination, workload, and physiological synchrony in collaborative tasks.

## 🎯 Overview

This repository contains a complete experimental system for investigating real-time multimodal team workload and coordination dynamics using the **Map Task paradigm**. The system integrates physiological sensors, eye tracking, communication analysis, and subjective measures to predict coordination breakdowns in dyadic collaboration.

## 🏗️ Project Structure
root/


├── map-task-frontend/    # React/TypeScript dyadic task interface


├── backend/    # Node.js HR monitoring API


├── wear-hr-app/    # Samsung Watch HR streaming app



## ✨ Key Features

### Map Task Frontend
- **Synced dyadic interface** with Director (instruction-giver) and Matcher (path-drawer) roles
- **Real-time synchronization** via Supabase Realtime channels
- **Comprehensive data capture**:
  - Drawing strokes with timestamps
  - Pointer position sampling
  - Event logging (mode changes, trial timing)
  - Final image export per trial
- **Integrated surveys**:
  - NASA-TLX (workload, 6 subscales)
  - PSMM (Perceived Shared Mental Models, 20 items)
- **One-click ZIP export** of complete session data

### Backend HR Monitoring
- Local REST API for heart rate data streaming
- Session-aligned timestamp synchronization
- CSV/NDJSON output formats
- Low-latency buffering (<200ms median)

### Samsung Watch App
- Lightweight Tizen/WearOS app
- Real-time HR and RR interval streaming
- Session ID alignment


   

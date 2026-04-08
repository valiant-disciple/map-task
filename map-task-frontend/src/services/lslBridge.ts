/**
 * LSL Bridge — optional local connection for Director's cross-machine sync.
 *
 * When lsl_flash_sender.py is running on this machine, this module forwards
 * flash events to it over a local WebSocket. The Python script re-emits them
 * as an LSL marker stream discoverable on the LAN.
 *
 * If the local server is NOT running (Matcher machine, or Director without
 * LSL setup), all calls silently no-op. This is intentional — LSL is an
 * optional enhancement, not a dependency.
 */

const LSL_BRIDGE_URL = 'ws://localhost:9001';
const RECONNECT_DELAY_MS = 5000;
const MAX_RECONNECT_ATTEMPTS = 3;

let ws: WebSocket | null = null;
let reconnectAttempts = 0;
let sessionId = '';

function connect() {
  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) return;

  try {
    ws = new WebSocket(LSL_BRIDGE_URL);

    ws.onopen = () => {
      reconnectAttempts = 0;
      console.log('[LSL] Local bridge connected — flash markers will be sent via LSL');
    };

    ws.onclose = () => {
      ws = null;
      reconnectAttempts++;
      if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        setTimeout(connect, RECONNECT_DELAY_MS);
      }
      // Silently stop after MAX_RECONNECT_ATTEMPTS — LSL bridge not running is fine.
    };

    ws.onerror = () => {
      // Suppress error — bridge is optional
    };
  } catch {
    // Suppress — browser may block ws://localhost in some contexts
  }
}

/** Call once at session start with the session ID. */
export function initLslBridge(sid: string) {
  sessionId = sid;
  connect();
}

/** Send a flash event to the local LSL bridge. Silently no-ops if not connected. */
export function sendFlashToLsl(trialIndex: number, flashTs: number) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(JSON.stringify({ type: 'flash', trialIndex, flashTs, sessionId }));
  } catch {
    // Ignore send errors
  }
}

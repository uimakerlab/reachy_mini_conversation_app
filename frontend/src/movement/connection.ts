import { CONFIG, type FullBodyPose } from "./types";

const DEBUG = import.meta.env.DEV;

let ws: WebSocket | null = null;
let daemonUrl: string = CONFIG.DEFAULT_DAEMON_URL;
let connected = false;
let connecting = false;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let onConnectCb: (() => void) | null = null;
let onDisconnectCb: (() => void) | null = null;

export function configure(opts: {
  daemonUrl?: string;
  onConnect?: () => void;
  onDisconnect?: () => void;
}): void {
  if (opts.daemonUrl !== undefined) daemonUrl = opts.daemonUrl.replace(/\/$/, "");
  if (opts.onConnect) onConnectCb = opts.onConnect;
  if (opts.onDisconnect) onDisconnectCb = opts.onDisconnect;
}

export function connect(): Promise<boolean> {
  if (connected) return Promise.resolve(true);
  if (connecting) {
    return new Promise((resolve) => {
      const id = setInterval(() => {
        if (connected) { clearInterval(id); resolve(true); }
        else if (!connecting) { clearInterval(id); resolve(false); }
      }, 100);
    });
  }

  connecting = true;
  return new Promise((resolve) => {
    const base = daemonUrl || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
    const wsUrl = base.replace(/^http/, "ws") + "/api/move/ws/set_target";
    if (DEBUG) console.log("[Movement] Connecting to", wsUrl);

    if (ws) {
      ws.onopen = null; ws.onerror = null; ws.onclose = null;
      try { ws.close(); } catch { /* noop */ }
      ws = null;
    }

    const timeout = setTimeout(() => {
      if (DEBUG) console.warn("[Movement] Connection timeout, proceeding without robot");
      connecting = false;
      resolve(false);
    }, 5000);

    try {
      const socket = new WebSocket(wsUrl);
      ws = socket;

      socket.onopen = () => {
        if (ws !== socket) return;
        clearTimeout(timeout);
        connected = true; connecting = false;
        if (DEBUG) console.log("[Movement] Connected");
        onConnectCb?.();
        resolve(true);
      };
      socket.onerror = () => {
        if (ws !== socket) return;
        clearTimeout(timeout);
        connecting = false;
        resolve(false);
      };
      socket.onclose = () => {
        if (ws !== socket) return;
        const was = connected;
        connected = false; connecting = false; ws = null;
        if (was) { onDisconnectCb?.(); scheduleReconnect(); }
      };
    } catch {
      clearTimeout(timeout);
      connecting = false;
      resolve(false);
    }
  });
}

export function disconnect(): void {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (ws) {
    ws.onopen = null; ws.onerror = null; ws.onclose = null;
    try { ws.close(); } catch { /* noop */ }
    ws = null;
  }
  connected = false; connecting = false;
}

function scheduleReconnect(): void {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, CONFIG.WS_RECONNECT_DELAY);
}

export function sendFullBodyPose(pose: FullBodyPose): boolean {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify({
    target_head_pose: pose.head,
    target_antennas: pose.antennas,
    target_body_yaw: pose.bodyYaw,
  }));
  return true;
}

export function getIsConnected(): boolean { return connected; }

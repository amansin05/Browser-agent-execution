import { useEffect } from "react";

/**
 * Grab the user's attention when the agent is waiting on them (an ask / approval).
 *
 * The browser CANNOT switch tabs to the dev-ui: the Playwright MCP extension only monitors its own
 * group of tabs, and the dev-ui is outside it (see README "Extension-mode notes"). So instead of a
 * tab-switch we alert in-page — which works no matter what tab is focused, because the prompt
 * arrives over the WebSocket, not the browser:
 *   - flash the tab title so a backgrounded tab is noticeable,
 *   - a short beep (allowed: the user already clicked "send"),
 *   - a desktop notification whose click refocuses this window.
 */
let audioCtx: AudioContext | null = null;

function beep() {
  try {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    audioCtx = audioCtx || new Ctx();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.frequency.value = 880;
    const t = audioCtx.currentTime;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.2, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
    osc.start(t);
    osc.stop(t + 0.36);
  } catch {
    /* audio not permitted yet — title + notification still fire */
  }
}

export function useAttention(active: boolean, message: string) {
  useEffect(() => {
    if (!active) return;
    const baseTitle = document.title;
    let flipped = false;
    const flip = window.setInterval(() => {
      flipped = !flipped;
      document.title = flipped ? "🔔 Answer needed — Browser Agent" : baseTitle;
    }, 1000);

    beep();

    let notif: Notification | null = null;
    if ("Notification" in window) {
      const show = () => {
        try {
          notif = new Notification("Browser Agent needs you", { body: message, tag: "agent-ask" });
          notif.onclick = () => { window.focus(); notif?.close(); };
        } catch {
          /* notification construction can throw on some platforms — ignore */
        }
      };
      if (Notification.permission === "granted") show();
      else if (Notification.permission !== "denied") {
        Notification.requestPermission().then((p) => { if (p === "granted") show(); }).catch(() => {});
      }
    }

    return () => {
      window.clearInterval(flip);
      document.title = baseTitle;
      notif?.close();
    };
  }, [active, message]);
}

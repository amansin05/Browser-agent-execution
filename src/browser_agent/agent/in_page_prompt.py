"""Ask the human INSIDE the tab the agent is driving.

The browser can't switch tabs to the dev-ui (it's outside Playwright's tab group), so for an
extension-mode run we render the question as an overlay in the controlled page — which is already
in front of you while you watch automation — and read the answer back by polling the DOM via
`browser_evaluate`. No tab-switch needed.

Both entry points return None when they couldn't show the prompt (e.g. the current page is a
chrome-extension:// page that rejects injection, or the wait timed out) so the caller can fall
back to the WebSocket modal.
"""

import asyncio
import base64
import json
import re

from browser_agent.log import get_logger
from browser_agent.services.mcp_client import tool_result_to_text

log = get_logger(__name__)

OVERLAY_ID = "__agent_prompt__"
APPROVE = "__APPROVE__"
DENY = "__DENY__"
SKIP_SENTINEL = "__no_preference__"  # mirrors orchestrate.SKIP_SENTINEL / the dev-ui AskModal

# Built as a DOM tree (textContent, never innerHTML, for option/question text) so page CSP and
# untrusted label text are both safe. __PAYLOAD__ is replaced with a JSON object; we do NOT use an
# f-string here so the JS braces stay literal.
_INJECT_JS = r"""
() => {
  const P = __PAYLOAD__;
  const ID = '__agent_prompt__';
  const prev = document.getElementById(ID); if (prev) prev.remove();
  window.__agentReply = null;

  const wrap = document.createElement('div');
  wrap.id = ID;
  wrap.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:rgba(15,18,25,.55);display:flex;align-items:center;justify-content:center;font-family:system-ui,Segoe UI,Arial,sans-serif';
  const card = document.createElement('div');
  card.style.cssText = 'background:#fff;color:#10131a;max-width:520px;width:90%;max-height:80vh;overflow:auto;padding:22px;border-radius:14px;box-shadow:0 18px 60px rgba(0,0,0,.35)';
  wrap.appendChild(card);

  const eyebrow = document.createElement('div');
  eyebrow.textContent = P.kind === 'approval' ? 'Approval needed'
    : (P.multi ? 'Select all that apply' : ((P.options && P.options.length) ? 'Select one' : 'Your input'));
  eyebrow.style.cssText = 'font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#6b7280;margin-bottom:8px';
  card.appendChild(eyebrow);

  const q = document.createElement('div');
  q.textContent = P.question;
  q.style.cssText = 'font-size:16px;font-weight:600;line-height:1.4;margin-bottom:14px';
  card.appendChild(q);

  const reply = (v) => { window.__agentReply = v; wrap.remove(); };
  const btn = (text, primary) => {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = text;
    b.style.cssText = 'padding:9px 16px;border-radius:9px;cursor:pointer;font-size:14px;font-weight:600;'
      + (primary ? 'border:0;background:#2563eb;color:#fff' : 'border:1px solid #d7dbe2;background:#fff;color:#10131a');
    return b;
  };

  if (P.kind === 'approval') {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:10px;justify-content:flex-end;margin-top:8px';
    const no = btn('Deny', false); no.onclick = () => reply('__DENY__');
    const yes = btn('Approve', true); yes.onclick = () => reply('__APPROVE__');
    row.appendChild(no); row.appendChild(yes);
    card.appendChild(row);
  } else {
    const multi = !!P.multi;
    const picked = new Set();
    const labelOf = {};
    const opts = (P.options || []).filter(o =>
      !['other', 'others'].includes((String(o.label || o.id || '')).trim().toLowerCase()));
    const list = document.createElement('div');
    list.style.cssText = 'display:flex;flex-direction:column;gap:8px;margin-bottom:12px';
    const style = (sel) => 'text-align:left;width:100%;padding:10px 12px;border-radius:10px;cursor:pointer;font-size:14px;'
      + 'border:1px solid ' + (sel ? '#2563eb' : '#d7dbe2') + ';background:' + (sel ? '#eef2ff' : '#fff');
    opts.forEach(o => {
      labelOf[o.id] = o.label || o.id;
      const b = document.createElement('button'); b.type = 'button'; b.style.cssText = style(false);
      const strong = document.createElement('b'); strong.textContent = o.label || o.id; b.appendChild(strong);
      if (o.detail) { const d = document.createElement('span'); d.style.color = '#6b7280'; d.textContent = ' — ' + o.detail; b.appendChild(d); }
      b.onclick = () => {
        if (multi) { picked.has(o.id) ? picked.delete(o.id) : picked.add(o.id); b.style.cssText = style(picked.has(o.id)); }
        else { reply(o.label || o.id); }
      };
      list.appendChild(b);
    });
    card.appendChild(list);

    let other = null;
    if (P.allow_free_text) {
      other = document.createElement('input');
      other.placeholder = 'Type your own answer…';
      other.style.cssText = 'width:100%;box-sizing:border-box;padding:10px 12px;border:1px solid #d7dbe2;border-radius:10px;font-size:14px;margin-bottom:12px';
      card.appendChild(other);
    }
    const send = () => {
      const chosen = [];
      picked.forEach(id => chosen.push(labelOf[id] || id));
      if (other && other.value.trim()) chosen.push(other.value.trim());
      if (chosen.length) reply(chosen.join(', '));
    };
    if (other) other.onkeydown = (e) => { if (e.key === 'Enter') send(); };

    const actions = document.createElement('div');
    actions.style.cssText = 'display:flex;gap:10px;justify-content:space-between;align-items:center;margin-top:4px';
    const skip = btn('No preference — explore all', false);
    skip.style.fontWeight = '500'; skip.style.fontSize = '13px';
    skip.onclick = () => reply('__no_preference__');
    actions.appendChild(skip);
    if (multi || P.allow_free_text) { const s = btn('Send answer', true); s.onclick = send; actions.appendChild(s); }
    card.appendChild(actions);
  }

  document.body.appendChild(wrap);
  return true;
}
"""

# Returns "__PENDING__" until the user acts, then "<<R>>" + base64(reply) + "<<E>>". Base64 keeps
# the marker free of <,> so the regex below is unambiguous regardless of how the result is wrapped.
_POLL_JS = r"""
() => {
  const r = window.__agentReply;
  if (r === null || r === undefined) return '__PENDING__';
  return '<<R>>' + btoa(unescape(encodeURIComponent(r))) + '<<E>>';
}
"""

_CLEANUP_JS = r"""
() => { const e = document.getElementById('__agent_prompt__'); if (e) e.remove(); window.__agentReply = null; return true; }
"""

_REPLY_RE = re.compile(r"<<R>>([A-Za-z0-9+/=]*)<<E>>")


def build_inject_js(payload: dict) -> str:
    """The injection script with the payload substituted in (kept separate for testing)."""
    return _INJECT_JS.replace("__PAYLOAD__", json.dumps(payload))


def parse_reply(text: str) -> str | None:
    """Extract the decoded reply from a poll result, or None if still pending/absent."""
    m = _REPLY_RE.search(text)
    if not m:
        return None
    try:
        return base64.b64decode(m.group(1)).decode("utf-8")
    except Exception:
        return ""


async def _evaluate(session, fn: str):
    return await session.call_tool("browser_evaluate", {"function": fn})


async def _show_and_wait(session, payload: dict, *, timeout: float, interval: float) -> str | None:
    try:
        res = await _evaluate(session, build_inject_js(payload))
        if getattr(res, "isError", False):
            log.info("in-page prompt: injection rejected (likely a non-web page); falling back")
            return None  # page rejected injection (e.g. chrome-extension:// page) -> caller falls back
    except Exception as e:
        log.info("in-page prompt: could not inject overlay (%r); falling back", e)
        return None
    log.info("in-page prompt: overlay shown (%s); waiting for the user", payload.get("kind"))
    try:
        waited = 0.0
        while waited < timeout:
            try:
                text = tool_result_to_text(await _evaluate(session, _POLL_JS))
            except Exception as e:
                log.info("in-page prompt: poll failed (%r); falling back", e)
                return None
            reply = parse_reply(text)
            if reply is not None:
                return reply
            await asyncio.sleep(interval)
            waited += interval
        log.info("in-page prompt: timed out after %.0fs; falling back", timeout)
        return None  # timed out -> caller falls back
    finally:
        try:
            await _evaluate(session, _CLEANUP_JS)
        except Exception:
            pass


async def ask_in_page(session, question, *, options=None, allow_free_text=True,
                      multi_select=False, timeout=300.0, interval=0.8) -> str | None:
    """Render a question overlay in the current page and return the user's answer string
    (joined option labels / typed text / the skip sentinel), or None if it couldn't be shown."""
    payload = {"kind": "ask", "question": question, "options": options or [],
               "allow_free_text": bool(allow_free_text), "multi": bool(multi_select)}
    return await _show_and_wait(session, payload, timeout=timeout, interval=interval)


async def approve_in_page(session, prompt, *, timeout=300.0, interval=0.8) -> bool | None:
    """Render an Approve/Deny overlay in the current page. Returns True/False, or None if it
    couldn't be shown (caller should fall back)."""
    payload = {"kind": "approval", "question": prompt}
    reply = await _show_and_wait(session, payload, timeout=timeout, interval=interval)
    if reply is None:
        return None
    return reply == APPROVE

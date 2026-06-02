"""Indexed-DOM perception — a numbered, addressable view of the page's interactive elements.

This is the perception upgrade modelled on browser-use / nanobrowser. The Playwright-MCP
accessibility snapshot (`browser_snapshot`) gives the reasoner an a11y *text* dump whose element
refs (`e5`, `e12`) renumber every navigation and miss JS-rendered clickables — the root cause of
the "clicked the wrong thing / looped on a menu / 0 candidates on a product grid" failures.

Instead we inject a `buildDomTree` script via the MCP `browser_evaluate` tool (the same seam the
content extractor uses — see agent/reader.evaluate_json). It walks the live DOM, finds the VISIBLE
INTERACTIVE elements (native controls, ARIA roles, contenteditable, click handlers, and the
cursor:pointer catch-all that surfaces framework widgets the a11y tree misses), assigns each a
stable integer index, and stamps `data-ba-id="<index>"` on the element. The reasoner then acts by
INDEX (`click_element(3)`), and we resolve that index back to the exact element via its data-ba-id —
so a click never depends on a stale ref or a fragile CSS selector.

The module has three parts:
  1. `index_dom(session)`        -> DomState (the numbered element map + page meta), best-effort.
  2. `DomState.render(...)`      -> the LLM-facing text, with NEW elements marked `*[i]`.
  3. `click_element` / `input_text` / `select_option` / `scroll_page` — index-addressed actions
     executed in-page, robust to reflows.

Everything is best-effort: any failure returns None / {"ok": False, ...} and never raises, so the
caller can fall back to the a11y snapshot rather than crash the run.
"""

import json
from dataclasses import dataclass, field

from browser_agent.agent.reader import evaluate_json
from browser_agent.log import get_logger
from browser_agent.services.mcp_client import tool_result_to_text

log = get_logger(__name__)

# Cap how much we pull back so a giant page can't blow the reasoner's context or the MCP result
# size. ~250 interactive elements is far more than any sane page exposes above the fold.
MAX_ELEMENTS = 250
MAX_TEXT = 120  # per-element visible-text cap

# The buildDomTree script. A trimmed, page.evaluate-compatible port of nanobrowser's
# public/buildDomTree.js: no highlight overlays (DOM-only mode), no getEventListeners (it's a
# DevTools-only API, unavailable inside page.evaluate — we lean on the cursor:pointer + role +
# event-attribute heuristics instead). Returns a flat list of interactive elements + page meta as a
# plain object, so Playwright JSON-encodes it once and evaluate_json recovers it directly.
_BUILD_DOM_JS = """
() => {
  const MAX_ELEMENTS = __MAX_ELEMENTS__, MAX_TEXT = __MAX_TEXT__;
  const SKIP = new Set(['script','style','noscript','template','svg','path','head','meta','link']);
  // Tags that ARE an interaction in themselves — we don't descend into them (their text is theirs).
  const ATOMIC = new Set(['a','button','input','select','textarea','option','summary']);
  const INTERACTIVE_TAGS = new Set(
    ['a','button','input','select','textarea','summary','details','option','label']);
  const INTERACTIVE_ROLES = new Set(
    ['button','link','menuitem','menuitemcheckbox','menuitemradio','radio','checkbox','tab',
     'switch','option','combobox','searchbox','textbox','slider','spinbutton','listbox','menu',
     'menubar']);

  const isDisabled = (el) =>
    el.disabled === true || el.getAttribute('aria-disabled') === 'true' || el.hasAttribute('disabled');

  const isVisible = (el) => {
    let s; try { s = window.getComputedStyle(el); } catch (e) { return false; }
    if (!s || s.visibility === 'hidden' || s.display === 'none' || parseFloat(s.opacity) === 0)
      return false;
    return el.offsetWidth > 0 || el.offsetHeight > 0 || el.getClientRects().length > 0;
  };

  const inViewport = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return false;
    return r.bottom >= 0 && r.right >= 0 &&
           r.top <= (window.innerHeight || 0) && r.left <= (window.innerWidth || 0);
  };

  const isInteractive = (el) => {
    if (isDisabled(el)) return false;
    const tag = el.tagName.toLowerCase();
    if (INTERACTIVE_TAGS.has(tag)) return true;
    const role = (el.getAttribute('role') || '').toLowerCase();
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.isContentEditable || el.getAttribute('contenteditable') === 'true') return true;
    if (el.hasAttribute('onclick') || typeof el.onclick === 'function') return true;
    const ti = el.getAttribute('tabindex');
    if (ti !== null && ti !== '-1') return true;
    // The catch-all: a pointer cursor is how most framework-rendered clickables (div/span widgets)
    // announce themselves. This is what the a11y tree misses on retail SPAs.
    try { if (window.getComputedStyle(el).cursor === 'pointer') return true; } catch (e) {}
    return false;
  };

  // Distinct from its already-interactive parent? (so a <div role=button> with nested clickable
  // children still surfaces them, but inherited cursor:pointer alone does NOT duplicate the parent.)
  const isDistinct = (el) => {
    const tag = el.tagName.toLowerCase();
    if (ATOMIC.has(tag)) return true;
    const role = (el.getAttribute('role') || '').toLowerCase();
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.isContentEditable || el.getAttribute('contenteditable') === 'true') return true;
    if (el.hasAttribute('onclick') || typeof el.onclick === 'function') return true;
    if (el.hasAttribute('data-testid') || el.hasAttribute('data-test')) return true;
    return false;
  };

  const FORM = new Set(['input', 'textarea', 'select']);

  // The accessible NAME — the label a human would call the control. Resolved the way browsers do:
  // aria-label, then aria-labelledby's referenced text, then the associated <label> (for/wrapping),
  // then placeholder/title/name/alt. This is the fix for inputs (a search box with only an id +
  // <label> used to come back nameless, so the model couldn't pick it / picked the wrong index).
  const accName = (el) => {
    try {
      const al = el.getAttribute('aria-label');
      if (al && al.trim()) return al.trim().slice(0, MAX_TEXT);
      const lb = el.getAttribute('aria-labelledby');
      if (lb) {
        const t = lb.split(/\\s+/).map(id => {
          const e = document.getElementById(id);
          return e ? (e.innerText || e.textContent || '') : '';
        }).join(' ').replace(/\\s+/g, ' ').trim();
        if (t) return t.slice(0, MAX_TEXT);
      }
      if (el.labels && el.labels.length) {
        const t = Array.from(el.labels).map(l => l.innerText || l.textContent || '')
          .join(' ').replace(/\\s+/g, ' ').trim();
        if (t) return t.slice(0, MAX_TEXT);
      }
      for (const a of ['placeholder', 'title', 'name', 'alt']) {
        const v = el.getAttribute(a);
        if (v && v.trim()) return v.trim().slice(0, MAX_TEXT);
      }
    } catch (e) {}
    return '';
  };

  const ownText = (el) =>
    (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, MAX_TEXT);

  const elements = [];
  let idx = 0;

  const walk = (node, inInteractive) => {
    if (idx >= MAX_ELEMENTS || !node || node.nodeType !== 1) return;
    const tag = node.tagName.toLowerCase();
    if (SKIP.has(tag) || node.id === 'playwright-highlight-container') return;

    let mine = false;
    let visible = false;
    try { visible = isVisible(node); } catch (e) {}
    if (visible && isInteractive(node) && (!inInteractive || isDistinct(node))) {
      const name = accName(node);
      const text = ownText(node);
      const href = tag === 'a' ? (node.getAttribute('href') || '') : '';
      const value = FORM.has(tag) && typeof node.value === 'string' ? node.value.trim().slice(0, MAX_TEXT) : '';
      // Drop label-less decorative matches (an icon/wrapper that only got picked up via
      // cursor:pointer with no name/text/href) — the model can't address them anyway and they're
      // pure noise. Keep all real form controls even when momentarily nameless.
      if (!name && !text && !href && !value && !FORM.has(tag)) {
        // skip this node, but still descend in case a labelled child lives inside it
      } else {
        node.setAttribute('data-ba-id', String(idx));
        let inv = false; try { inv = inViewport(node); } catch (e) {}
        elements.push({
          i: idx, tag,
          role: (node.getAttribute('role') || '').toLowerCase(),
          type: (node.getAttribute('type') || '').toLowerCase(),
          text, name, href, value, inViewport: inv,
        });
        idx++;
        mine = true;
        if (ATOMIC.has(tag)) return;  // don't descend into a button/link/input/select
      }
    }
    const childInInteractive = inInteractive || mine;
    if (node.shadowRoot) {
      for (const c of node.shadowRoot.children) walk(c, childInInteractive);
    }
    for (const c of node.children) walk(c, childInInteractive);
  };

  // Clear ids from a previous pass so a reflowed page can't leave two elements claiming one index.
  document.querySelectorAll('[data-ba-id]').forEach((e) => e.removeAttribute('data-ba-id'));
  if (document.body) walk(document.body, false);

  const docEl = document.documentElement;
  return {
    url: location.href,
    title: document.title || '',
    scrollY: Math.round(window.scrollY || 0),
    scrollHeight: Math.round((docEl && docEl.scrollHeight) || 0),
    innerHeight: Math.round(window.innerHeight || 0),
    elements,
  };
}
"""


@dataclass
class DomElement:
    """One interactive element the reasoner can address by `index`."""
    index: int
    tag: str
    role: str = ""
    type: str = ""
    text: str = ""
    name: str = ""
    href: str = ""
    value: str = ""          # current value of an input/textarea/select (what's already typed in)
    in_viewport: bool = True

    def key(self) -> tuple:
        """A reflow-stable identity (NOT the index, which shifts) for marking NEW elements across
        steps. Two renders of the same logical control share a key even if their index moved.
        (Excludes `value` so typing into a field doesn't make it look 'new' next step.)"""
        return (self.tag, self.role, self.type, self.name, self.href, self.text[:40])

    def render(self) -> str:
        """One LLM-facing line, e.g. `[3] <button> "Add to cart"` or `[1] <input type=search> "Search"`."""
        attrs = f" type={self.type}" if self.type else ""
        attrs += f" role={self.role}" if self.role and self.role not in self.tag else ""
        label = self.name or self.text
        # Prefer the accessible name; show the visible text too when it adds information.
        if self.name and self.text and self.text != self.name:
            label = f"{self.name} | {self.text}"
        label = (label or "").strip()
        href = f" -> {self.href}" if self.href else ""
        value = f' ="{self.value}"' if self.value else ""  # show what's already in the field
        offscreen = "" if self.in_viewport else " (off-screen — scroll to reach)"
        quoted = f' "{label}"' if label else ""
        return f"[{self.index}] <{self.tag}{attrs}>{quoted}{value}{href}{offscreen}"


@dataclass
class DomState:
    """The numbered interactive-element map + page meta for one observation."""
    url: str = ""
    title: str = ""
    scroll_y: int = 0
    scroll_height: int = 0
    viewport_height: int = 0
    elements: list[DomElement] = field(default_factory=list)
    selector_map: dict[int, DomElement] = field(default_factory=dict)

    @property
    def keys(self) -> set:
        """The reflow-stable keys of every element — what the NEXT step diffs against for NEW marks."""
        return {e.key() for e in self.elements}

    def _scroll_hint(self) -> str:
        above = self.scroll_y
        below = max(0, self.scroll_height - self.scroll_y - self.viewport_height)
        if self.scroll_height <= self.viewport_height + 10:
            return "whole page fits the viewport"
        parts = []
        if above > 20:
            parts.append(f"~{above}px above")
        if below > 20:
            parts.append(f"~{below}px below (scroll down for more)")
        return ", ".join(parts) or "at top"

    def render(self, previous_keys: set | None = None) -> str:
        """Render the page for the reasoner. Elements that appeared since `previous_keys` (the prior
        step's keys) are marked `*[i]` so the model notices a dropdown/modal/grid that just opened —
        the browser-use NEW-element cue."""
        prev = previous_keys or set()
        head = (f"[page] {self.url}" + (f' — "{self.title}"' if self.title else "") + "\n"
                f"[scroll] {self._scroll_hint()}")
        if not self.elements:
            return head + "\n\n(no interactive elements detected — the page may be plain text, a "
            "canvas, or still loading; read the extracted content or scroll/navigate)"
        lines = []
        for e in self.elements:
            star = "*" if e.key() not in prev else " "
            lines.append(f"{star}{e.render()}")
        return (head + "\n\nInteractive elements — act on these by their [index] "
                "(click_element / input_text / select_option). `*` marks elements new since the "
                "last step:\n" + "\n".join(lines))


def _parse_dom_state(value) -> DomState | None:
    """Turn the decoded buildDomTree value into a DomState. None if it's not the shape we expect."""
    if not isinstance(value, dict) or not isinstance(value.get("elements"), list):
        return None
    elements = []
    for raw in value["elements"]:
        if not isinstance(raw, dict) or "i" not in raw:
            continue
        elements.append(DomElement(
            index=int(raw["i"]),
            tag=str(raw.get("tag", "")),
            role=str(raw.get("role", "")),
            type=str(raw.get("type", "")),
            text=str(raw.get("text", "")),
            name=str(raw.get("name", "")),
            href=str(raw.get("href", "")),
            value=str(raw.get("value", "")),
            in_viewport=bool(raw.get("inViewport", True)),
        ))
    return DomState(
        url=str(value.get("url", "")),
        title=str(value.get("title", "")),
        scroll_y=int(value.get("scrollY", 0) or 0),
        scroll_height=int(value.get("scrollHeight", 0) or 0),
        viewport_height=int(value.get("innerHeight", 0) or 0),
        elements=elements,
        selector_map={e.index: e for e in elements},
    )


async def index_dom(session) -> DomState | None:
    """Build the indexed-DOM view of the current page via `browser_evaluate`. Best-effort: returns
    None on any failure (the caller then falls back to the a11y snapshot)."""
    js = _BUILD_DOM_JS.replace("__MAX_ELEMENTS__", str(MAX_ELEMENTS)).replace("__MAX_TEXT__", str(MAX_TEXT))
    value = await evaluate_json(session, js)
    state = _parse_dom_state(value)
    if state is None:
        log.debug("index_dom: buildDomTree returned an unexpected shape; falling back")
        return None
    log.debug("index_dom: %d interactive elements on %s", len(state.elements), state.url[:80])
    return state


# --------------------------------------------------------------------------- index-addressed actions
# Each action resolves `index` -> the element stamped `data-ba-id="<index>"` and acts on it in-page.
# This is what makes clicks robust: no stale snapshot ref, no guessed CSS selector. All return a
# dict {"ok": bool, ...}; they never raise (evaluate_json swallows failures and returns None).

def _by_id_prelude(index: int) -> str:
    """JS that resolves the data-ba-id element into `el`, returning early if it's gone (the page
    reflowed and the index is stale — the caller should re-observe)."""
    return ("const el = document.querySelector('[data-ba-id=\"" + str(int(index)) + "\"]');\n"
            "  if (!el) return {ok:false, reason:'element [' + " + str(int(index)) +
            " + '] not found — the page changed; re-observe'};\n")


async def _run_action(session, function: str) -> dict:
    val = await evaluate_json(session, function)
    if isinstance(val, dict):
        return val
    return {"ok": False, "reason": "action returned no result (eval failed)"}


async def click_element(session, index: int) -> dict:
    """Click the element addressed by `index`. Scrolls it into view first."""
    fn = ("() => {\n  " + _by_id_prelude(index) +
          "  el.scrollIntoView({block:'center', inline:'center'});\n"
          "  const tag = el.tagName.toLowerCase();\n"
          "  el.click();\n"
          "  return {ok:true, tag, text:(el.innerText||el.value||'').slice(0,80)};\n}")
    return await _run_action(session, fn)


async def input_text(session, index: int, text: str) -> dict:
    """Type `text` into the input/textarea/contenteditable addressed by `index`. Uses the native
    value setter + input/change events so React/Vue controlled inputs register the change (a bare
    `el.value = x` is silently ignored by those frameworks)."""
    fn = ("() => {\n  " + _by_id_prelude(index) +
          "  el.scrollIntoView({block:'center'});\n  el.focus();\n"
          "  const val = " + json.dumps(text) + ";\n"
          "  if (el.isContentEditable) { el.textContent = val; }\n"
          "  else {\n"
          "    const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype\n"
          "                                            : window.HTMLInputElement.prototype;\n"
          "    const d = Object.getOwnPropertyDescriptor(proto, 'value');\n"
          "    if (d && d.set) { d.set.call(el, val); } else { el.value = val; }\n  }\n"
          "  el.dispatchEvent(new Event('input', {bubbles:true}));\n"
          "  el.dispatchEvent(new Event('change', {bubbles:true}));\n"
          "  return {ok:true, tag:el.tagName.toLowerCase()};\n}")
    return await _run_action(session, fn)


async def select_option(session, index: int, value: str) -> dict:
    """Pick an <option> (by value OR visible label) in the <select> addressed by `index`."""
    fn = ("() => {\n  " + _by_id_prelude(index) +
          "  const want = " + json.dumps(value) + ";\n"
          "  let matched = false;\n"
          "  for (const o of (el.options || [])) {\n"
          "    if (o.value === want || (o.text || '').trim() === want) { o.selected = true; matched = true; break; }\n"
          "  }\n"
          "  el.dispatchEvent(new Event('change', {bubbles:true}));\n"
          "  return {ok:matched, value:el.value, reason: matched ? undefined : 'no option matched ' + want};\n}")
    return await _run_action(session, fn)


async def scroll_page(session, direction: str = "down", amount: int | None = None) -> dict:
    """Scroll the window down/up by `amount` px (defaults to ~one viewport). Used instead of a
    snapshot-ref scroll so it works uniformly across pages."""
    sign = "-" if str(direction).lower() == "up" else "+"
    px = "Math.round(window.innerHeight * 0.85)" if amount is None else str(int(amount))
    fn = ("() => {\n"
          f"  const by = {sign}({px});\n"
          "  window.scrollBy(0, by);\n"
          "  return {ok:true, scrollY:Math.round(window.scrollY), "
          "scrollHeight:Math.round(document.documentElement.scrollHeight)};\n}")
    return await _run_action(session, fn)


# --------------------------------------------------------------------------- the reasoner dispatch
# A single entry point the orchestrator calls per reasoner turn. In-page interactions go through the
# index-addressed actions above; navigation / key presses go to the real Playwright-MCP tools. Every
# return is {"ok": bool, "outcome": str} and nothing raises — the orchestrator turns `outcome` into
# the recent-actions line the reasoner reads next turn.
INTERACTION_ACTIONS = {"click_element", "input_text", "select_option", "scroll_page"}
MCP_ACTIONS = {"navigate", "go_back", "press_key"}
REASONER_ACTION_NAMES = INTERACTION_ACTIONS | MCP_ACTIONS

# The page-change guard for multi-action turns (browser-use's "terminates_sequence"): only these
# actions are safe to CHAIN, because they leave you on the same page with the same elements. Any
# other action (a click, a navigation, a key press, or an input that submits) may change the page,
# so it ENDS the batch — the next step re-perceives before acting again, never against a stale view.
CHAINABLE_ACTIONS = {"select_option", "scroll_page"}


def is_terminating(action: str, args: dict) -> bool:
    """True if `action` may change the page and so must end a multi-action batch."""
    if action == "input_text":
        return bool(args.get("submit"))  # typing is chainable; pressing Enter (submit) navigates
    return action not in CHAINABLE_ACTIONS


def _mcp_outcome(res) -> dict:
    txt = tool_result_to_text(res)
    return {"ok": not getattr(res, "isError", False), "outcome": txt[:200].replace("\n", " ")}


def _inpage_outcome(r: dict) -> dict:
    """Turn an index-action result dict into the standard {ok, outcome}."""
    if not r.get("ok"):
        return {"ok": False, "outcome": str(r.get("reason") or "no effect")[:200]}
    bits = [f"{k}={v}" for k, v in r.items() if k not in ("ok", "reason") and v not in (None, "")]
    return {"ok": True, "outcome": ("ok (" + ", ".join(bits) + ")") if bits else "ok"}


async def execute_action(session, action: str, args: dict) -> dict:
    """Execute one reasoner action. Returns {"ok": bool, "outcome": str}; never raises."""
    try:
        if action == "click_element":
            return _inpage_outcome(await click_element(session, int(args.get("index"))))
        if action == "input_text":
            r = await input_text(session, int(args.get("index")), str(args.get("text", "")))
            if r.get("ok") and args.get("submit"):
                # Press Enter as a REAL keyboard event on the now-focused field (more reliable than
                # synthesising one in JS) — this is what runs a search / submits a form.
                try:
                    await session.call_tool("browser_press_key", {"key": "Enter"})
                    r["submitted"] = True
                except Exception as e:  # noqa: BLE001 — Enter is best-effort; the type already landed
                    r["submit_error"] = repr(e)
            return _inpage_outcome(r)
        if action == "select_option":
            return _inpage_outcome(await select_option(session, int(args.get("index")), str(args.get("value", ""))))
        if action == "scroll_page":
            return _inpage_outcome(await scroll_page(session, str(args.get("direction", "down")), args.get("amount")))
        if action == "navigate":
            return _mcp_outcome(await session.call_tool("browser_navigate", {"url": args.get("url", "")}))
        if action == "go_back":
            return _mcp_outcome(await session.call_tool("browser_navigate_back", {}))
        if action == "press_key":
            return _mcp_outcome(await session.call_tool("browser_press_key", {"key": args.get("key", "Enter")}))
        return {"ok": False, "outcome": f"unknown action {action!r}"}
    except Exception as e:  # noqa: BLE001 — a bad action must surface as feedback, not crash the run
        return {"ok": False, "outcome": f"{action} raised: {e!r}"}

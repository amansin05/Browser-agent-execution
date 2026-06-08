"""DOM-side candidate extraction — reads structured product rows from the LIVE DOM.

The old path (agent/extract.extract_candidates) fed Scout a 40k head-slice of the *accessibility
snapshot* and asked it to reconstruct product cards. The a11y tree flattens each card's
title/price/rating into scattered sibling nodes among nav/filter/sponsored chrome, so re-binding
them is unreliable — fine on uniform pages, broken on Amazon (the "0 candidates despite a full
results page" bug). Head-slicing can also cut the listing region; a single JSON wobble returns [].

Here we read candidates straight from the live DOM via the MCP `browser_evaluate` seam (the same one
agent/reader.evaluate_json and agent/dom_index use). In the live DOM each card's grouping is intact,
so title/price/rating is a deterministic per-card query, NOT an LLM inference — which removes the
per-site variance, the LLM call on the hot path, and the JSON-parse failure mode.

  - SITE-AWARE fast paths for amazon.in / flipkart.com (selectors derived from the real markup),
  - a GENERIC fallback (repeated containers holding a currency token co-located with a rating token)
    for any other / unknown site, so a redesign degrades to the generic reader rather than 0.

Rows are `{name, price, rating, review_count, url, source}` — the schema the scorer / selector /
synthesizer already consume (agent/scoring, extract.select_candidates, EXTRACTOR/SELECTOR prompts).
Dedupe is deterministic (Python); no LLM touches the extraction hot path. Best-effort: any failure
returns [] and the caller falls back to the LLM a11y extractor. Never raises.
"""

import asyncio
import json
import re

from browser_agent.agent.reader import evaluate_json
from browser_agent.log import get_logger

log = get_logger(__name__)

# A checkout / payment / order-review URL (pure, no DOM). Mirrors the URL arm of _AT_CHECKOUT_JS so
# tab-parking can recognize a payment page WITHOUT a live probe — the buy flow stops there for the
# user to place the order, so that tab must be LEFT OPEN, never closed (the "it closed the tab
# instead of letting me buy" bug).
_CHECKOUT_URL_RE = re.compile(
    r"/gp/buy/|/checkout/|/spc/|/payments?\b|buy/spc|go-to-checkout|alm.*checkout|proceedtocheckout",
    re.I)


def is_checkout_url(url: str) -> bool:
    """True if `url` is a checkout / payment / order-review page (not the cart). Used so a buy flow's
    handoff tab is preserved for the user to place the order."""
    return bool(url) and bool(_CHECKOUT_URL_RE.search(url))

MAX_CANDIDATES = 40       # cap rows per extraction (a page rarely needs more to choose from)
_SCROLL_SETTLE_S = 0.6    # let a lazy grid load after a scroll before re-extracting

# In-page extractor. Returns a flat array of {name, price, rating, review_count, url, source} as a
# plain JS value, so Playwright JSON-encodes it once and evaluate_json recovers it directly. Selector
# choices are derived from real markup (scripts probe): Amazon's design-system classes are stable
# (a-price/a-offscreen/a-icon-alt/s-search-result); Flipkart's are obfuscated + rotating, so there we
# anchor on the stable `div[data-id]` card and read fields by structure/regex, never by class.
_EXTRACT_JS = r"""
() => {
  const MAX = __MAX__;
  const host = location.hostname.replace(/^www\./, '');
  // NEVER extract candidates off a search-engine results page: its result blocks look like priced
  // cards to the generic reader below, so a SERP yields snippet junk (titles + numbers scraped from
  // the snippet) with no buyable product URL (the SERP-scrape bug). Bail so the caller re-plans onto
  // a real retailer. Mirrors the SKIP list in agent/grounding.py and utils/domains.is_search_engine.
  if (/(^|\.)(google|bing|duckduckgo|yahoo|baidu|yandex|ecosia|startpage|ask|aol|qwant|brave)\./i.test(location.hostname)) return [];
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const numFrom = (s) => {
    if (s == null) return null;
    const m = String(s).replace(/[,\s]/g, '').match(/\d+(?:\.\d+)?/);
    return m ? parseFloat(m[0]) : null;
  };
  const rows = [];
  const push = (r) => { if (r && (r.name || r.url)) rows.push(r); };

  // ---- amazon.in / amazon.com ----
  if (host.endsWith('amazon.in') || host.endsWith('amazon.com')) {
    for (const c of document.querySelectorAll('div[data-component-type="s-search-result"]')) {
      if (rows.length >= MAX) break;
      const titleEl = c.querySelector('h2 span') || c.querySelector('h2 a') || c.querySelector('h2');
      const a = c.querySelector('h2 a') || c.querySelector('a.a-link-normal.s-link-style');
      const name = clean(titleEl && titleEl.textContent);
      if (!name) continue;
      const priceEl = c.querySelector('.a-price .a-offscreen');
      const ratingEl = c.querySelector('.a-icon-alt') || c.querySelector('[aria-label*="out of 5"]');
      const revEl = c.querySelector('[aria-label$="ratings"], [aria-label$="rating"]');
      const priceNum = priceEl ? numFrom(priceEl.textContent) : null;
      const rating = ratingEl ? numFrom((ratingEl.getAttribute && ratingEl.getAttribute('aria-label')) || ratingEl.textContent) : null;
      // a.href is the browser-resolved ABSOLUTE url; data-asin is the product id (also in /dp/<ASIN>/);
      // search cards expose an add-to-cart control we flag so the buy flow knows it can add from here.
      push({ name, price: (priceNum && priceNum > 0) ? ('₹' + priceNum) : null,
             rating, review_count: revEl ? numFrom(revEl.getAttribute('aria-label')) : null,
             url: (a && a.href) ? a.href : null, asin: c.getAttribute('data-asin') || null,
             add_to_cart: !!c.querySelector('button[name="submit.addToCart"], [data-action="add-to-cart"]'),
             source: host });
    }
    if (rows.length) return rows.slice(0, MAX);
  }

  // ---- flipkart.com ---- (anchor on the stable data-id card; classes rotate, so read by structure)
  if (host.endsWith('flipkart.com')) {
    for (const c of document.querySelectorAll('div[data-id]')) {
      if (rows.length >= MAX) break;
      const titled = c.querySelector('a[title]');
      let a = titled;
      if (!a) a = Array.from(c.querySelectorAll('a[href*="/p/"]'))
        .sort((x, y) => clean(y.textContent).length - clean(x.textContent).length)[0];
      const name = clean(a && ((a.getAttribute && a.getAttribute('title')) || a.textContent));
      if (!name) continue;
      const txt = clean(c.innerText);
      const pm = txt.match(/₹\s?[\d,]+/);                 // first ₹ = selling price (MRP is struck, after)
      const rm = txt.match(/(\d(?:\.\d)?)\s*\((\d[\d,]*)\)/);   // "4.5(21,877)"
      const purl = (c.querySelector('a[href*="/p/"]') || a || {}).href || null;
      push({ name, price: pm ? pm[0].replace(/\s/g, '') : null,
             rating: rm ? parseFloat(rm[1]) : null,
             review_count: rm ? numFrom(rm[2]) : null, url: purl,
             asin: c.getAttribute('data-id') || null, add_to_cart: false, source: host });
    }
    if (rows.length) return rows.slice(0, MAX);
  }

  // ---- GENERIC fallback: repeated containers holding a currency token (+ optional rating) ----
  const PRICE = /(?:₹|Rs\.?\s?|\$|€|£)\s?[\d,]{2,}/;
  const ownText = (el) => Array.from(el.childNodes || [])
    .filter(n => n.nodeType === 3).map(n => n.textContent).join(' ');
  const priceLeaves = [];
  for (const el of document.querySelectorAll('span,div,b,strong,p')) {
    if (priceLeaves.length >= 60) break;
    const t = ownText(el).trim();
    if (t && t.length < 40 && PRICE.test(t)) priceLeaves.push(el);
  }
  const usedCards = new Set();
  for (const p of priceLeaves) {
    if (rows.length >= MAX) break;
    let el = p, card = null;
    for (let d = 0; d < 9 && el && el.parentElement; d++, el = el.parentElement) {
      const cls = (typeof el.className === 'string') ? el.className.trim() : '';
      const sibs = Array.from(el.parentElement.children).filter(c =>
        c.tagName === el.tagName && ((typeof c.className === 'string' ? c.className.trim() : '') === cls));
      if (sibs.length >= 3 && el.querySelector('a[href]')) { card = el; break; }
    }
    card = card || p.parentElement;
    if (!card || usedCards.has(card)) continue;
    usedCards.add(card);
    let name = '';
    const titled = card.querySelector('a[title]');
    if (titled) name = clean(titled.getAttribute('title'));
    if (!name) {
      const links = Array.from(card.querySelectorAll('a')).map(x => clean(x.textContent))
        .filter(Boolean).sort((x, y) => y.length - x.length);
      name = links[0] || '';
    }
    if (!name) { const h = card.querySelector('h1,h2,h3,h4'); name = clean(h && h.textContent); }
    if (!name) continue;
    const ctext = clean(card.innerText);
    const pm = ctext.match(PRICE);
    const rm = ctext.match(/(\d(?:\.\d)?)\s*(?:out of 5|stars?)/i) || ctext.match(/(\d\.\d)\s*\(\s*([\d,]+)\s*\)/);
    const a = card.querySelector('a[href]');
    push({ name: name.slice(0, 140), price: pm ? pm[0].replace(/\s/g, '') : null,
           rating: rm ? parseFloat(rm[1]) : null,
           review_count: (rm && rm[2]) ? numFrom(rm[2]) : null,
           url: (a && a.href) ? a.href : null, asin: null, add_to_cart: false, source: host });
  }
  return rows.slice(0, MAX);
}
"""


# Probe the cart/bag item count, for the pre-checkout confirmation gate (orchestrate, fix 3). Returns
# an integer (0 = confirmed empty) or null (couldn't determine). Site-aware for Amazon's stable
# `#nav-cart-count`; generic otherwise (a cart/bag element with a number, or an "N item(s)" badge).
_CART_COUNT_JS = r"""
() => {
  const host = location.hostname.replace(/^www\./, '');
  const toInt = (s) => { const m = String(s == null ? '' : s).replace(/[,\s]/g, '').match(/\d+/); return m ? parseInt(m[0], 10) : null; };
  if (host.endsWith('amazon.in') || host.endsWith('amazon.com')) {
    const el = document.querySelector('#nav-cart-count');
    const n = el ? toInt(el.textContent) : null;
    if (n != null) return n;
  }
  let best = null;
  for (const el of document.querySelectorAll('[aria-label],[title],span,a,div')) {
    const t = (((el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('title'))) || el.textContent) || '').trim();
    if (!t || t.length > 40) continue;
    const m = t.match(/(?:cart|bag|basket)\D{0,8}(\d+)/i) || t.match(/(\d+)\s*items?\b/i);
    if (m) { const n = parseInt(m[1], 10); if (!isNaN(n) && (best == null || n > best)) best = n; }
  }
  return best;
}
"""


async def current_url(session) -> str:
    """The live page URL (a cheap `location.href` eval). '' if it can't be read. Never raises.
    Used to detect when the agent has ended up on a search-engine results page before extracting."""
    val = await evaluate_json(session, "() => location.href")
    return val if isinstance(val, str) else ""


async def cart_count(session) -> int | None:
    """Best-effort cart/bag item count: an int (0 = confirmed empty) or None if it can't be read.
    Used to confirm an item was actually added before navigating to checkout. Never raises."""
    val = await evaluate_json(session, _CART_COUNT_JS)
    return val if isinstance(val, int) and not isinstance(val, bool) else None


# Detect the post-add-to-cart CONFIRMED state on the CURRENT page (no navigation to /cart). Amazon
# clicks of #add-to-cart-button land on a `/cart/smart-wagon` interstitial with a confirmation panel
# and the header badge incremented — NOT the full cart (so "go to /cart and read contents" misses
# it). Signals, derived from a live add-to-cart (scripts probe): cart-count badge > 0, a smart-wagon /
# attach confirmation panel, a `/cart/smart-wagon|/gp/huc` URL, body text "added to cart", or the
# selected item's title on the page. `__TITLE__` is replaced with the lowercased selected title.
_CART_CONFIRMED_JS = r"""
() => {
  const title = __TITLE__;
  const toInt = (s) => { const m = String(s == null ? '' : s).replace(/[,\s]/g, '').match(/\d+/); return m ? parseInt(m[0], 10) : null; };
  let badge = null;
  const el = document.querySelector('#nav-cart-count');
  if (el) badge = toInt(el.textContent);
  if (badge == null) {
    const navc = document.querySelector('#nav-cart, #nav-cart-count-container');
    const aria = navc && navc.getAttribute ? navc.getAttribute('aria-label') : null;
    if (aria) badge = toInt(aria);
  }
  if (badge != null && badge > 0) return { ok: true, signal: 'cart badge ' + badge, badge };
  const PANELS = ['#sw-atc-confirmation', '#NATC_SMART_WAGON_CONF_MSG_SUCCESS',
    '#attach-added-to-cart-confirmation', '#huc-v2-order-row-confirm-text', '#sw-atc-details-single-container'];
  for (const s of PANELS) if (document.querySelector(s)) return { ok: true, signal: 'confirmation panel ' + s, badge };
  if (/\/cart\/smart-wagon|\/gp\/huc|added-to-cart/i.test(location.href)) return { ok: true, signal: 'smart-wagon url', badge };
  const body = (document.body ? document.body.innerText : '').toLowerCase();
  if (body.includes('added to cart') || body.includes('added to your cart')) return { ok: true, signal: 'body says added to cart', badge };
  if (title && body.includes(title)) return { ok: true, signal: 'selected item shown', badge };
  return { ok: false, badge };
}
"""


# Find + click the Add-to-Cart control on a product page (fix 2). Primary: Amazon's stable
# #add-to-cart-button (an <input value="Add to cart">); fallback: any button/input/role=button whose
# visible text matches /add to (cart|bag|basket)/i. Clicking in-page is robust to index instability
# and means the reasoner never has to scroll-hunt for it (the regression).
_ADD_TO_CART_JS = r"""
() => {
  let el = document.querySelector('#add-to-cart-button') ||
           document.querySelector('input#add-to-cart-button') ||
           document.querySelector('#addToCart, #add-to-cart, button[name="submit.add-to-cart"]');
  if (!el) {
    for (const c of document.querySelectorAll('button, input[type=submit], input[type=button], a[role=button], [role=button]')) {
      const t = (c.value || c.textContent || c.getAttribute('aria-label') || '').trim();
      if (/add to (cart|bag|basket)/i.test(t)) { el = c; break; }
    }
  }
  if (!el) return { ok: false, reason: 'no add-to-cart button on this page' };
  try {
    el.scrollIntoView({ block: 'center' });
    const label = (el.value || el.textContent || el.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim().slice(0, 40);
    el.click();
    return { ok: true, text: label || 'Add to cart' };
  } catch (e) { return { ok: false, reason: String(e) }; }
}
"""


async def click_add_to_cart(session) -> dict:
    """Locate and click the Add-to-Cart control on the current product page (in-page, no index).
    Returns {"ok": True, "text": <label>} or {"ok": False, "reason": ...}. Never raises."""
    val = await evaluate_json(session, _ADD_TO_CART_JS)
    return val if isinstance(val, dict) else {"ok": False, "reason": "add-to-cart eval failed"}


# Select a REQUIRED size/variant on a product page so Add-to-Cart isn't blocked by an unselected
# option (the asics/superkicks "clicked Add to Cart but it never confirmed" failure — Shopify and most
# fashion/footwear PDPs refuse the add until a size is picked). Three patterns, in order: a size
# <select>; size radio inputs (Shopify variant radios); or a swatch group of size-token buttons/links.
# Picks `__PREF__` (a lowercased preferred size) when it matches an available option, else the first
# IN-STOCK one; skips sold-out/disabled options. Returns {ok, kind, selected?/reason?}: ok=true when it
# selected something; kind 'already' (a variant was already chosen — no-op), 'none' (no variant UI — a
# single-variant product), or 'size' + reason (a size is required but none is available). Best-effort.
_SELECT_VARIANT_JS = r"""
() => {
  const PREF = __PREF__;
  const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
  const view = document.defaultView || window;
  const visible = (el) => {
    if (!el) return false;
    let st = null; try { st = view.getComputedStyle(el); } catch (e) {}
    if (st && (st.display === 'none' || st.visibility === 'hidden')) return false;
    const r = el.getBoundingClientRect ? el.getBoundingClientRect() : { width: 1, height: 1 };
    return !(r.width === 0 && r.height === 0 && !el.offsetParent);
  };
  const soldOut = (el) => {
    if (!el) return false;
    if (el.disabled || (el.getAttribute && el.getAttribute('aria-disabled') === 'true')) return true;
    const cls = (typeof el.className === 'string' ? el.className : '').toLowerCase();
    if (/sold[-\s]?out|unavailable|out[-\s]?of[-\s]?stock|\boos\b|is-disabled|disabled/.test(cls)) return true;
    const t = norm(el.textContent || el.value).toLowerCase();
    return /sold out|out of stock|notify me|coming soon/.test(t);
  };
  const SIZE_WORD = /\b(size|uk|us|eu|eur)\b/i;
  const SIZE_TOKEN = /^(?:uk|us|eu|eur)?\s?\d{1,2}(?:\.\d)?$|^(?:xxs|xs|s|m|l|xl|xxl|xxxl)$/i;
  const labelFor = (el) => {
    if (el.labels && el.labels[0]) return norm(el.labels[0].textContent);
    if (el.id) { const l = document.querySelector('label[for="' + (window.CSS && CSS.escape ? CSS.escape(el.id) : el.id) + '"]'); if (l) return norm(l.textContent); }
    return norm(el.value);
  };
  const matchesPref = (txt) => PREF && norm(txt).toLowerCase().includes(PREF);

  // ---- 1) a <select> for size / variant ----
  for (const sel of document.querySelectorAll('select')) {
    if (!visible(sel)) continue;
    const tag = (sel.getAttribute('aria-label') || sel.name || sel.id || '').toLowerCase();
    const opts = Array.from(sel.options || []);
    const looksSize = SIZE_WORD.test(tag) || opts.some(o => SIZE_TOKEN.test(norm(o.textContent)));
    if (!looksSize) continue;
    const valid = (o) => o && o.value && !o.disabled && !/^(select|choose|pick|size|please)/i.test(norm(o.textContent));
    const cur = sel.options[sel.selectedIndex];
    if (cur && valid(cur) && norm(cur.textContent)) return { ok: false, kind: 'already', selected: norm(cur.textContent) };
    const choices = opts.filter(valid);
    if (!choices.length) return { ok: false, kind: 'size', reason: 'no selectable size options' };
    let pick = PREF ? choices.find(o => matchesPref(o.textContent)) : null;
    pick = pick || choices[0];
    sel.value = pick.value;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    sel.dispatchEvent(new Event('input', { bubbles: true }));
    return { ok: true, kind: 'select', selected: norm(pick.textContent) };
  }

  // ---- 2) size radio inputs (Shopify variant radios) ----
  const radios = Array.from(document.querySelectorAll('input[type=radio]')).filter(visible);
  const sizeRadios = radios.filter(r => {
    const nm = (r.name || r.id || r.getAttribute('aria-label') || '').toLowerCase();
    return SIZE_WORD.test(nm) || SIZE_TOKEN.test(labelFor(r));
  });
  if (sizeRadios.length) {
    const already = sizeRadios.find(r => r.checked && !soldOut(r));
    if (already) return { ok: false, kind: 'already', selected: labelFor(already) };
    const avail = sizeRadios.filter(r => !soldOut(r) && !soldOut(r.labels && r.labels[0]));
    if (!avail.length) return { ok: false, kind: 'size', reason: 'all sizes sold out' };
    let pick = PREF ? avail.find(r => matchesPref(labelFor(r))) : null;
    pick = pick || avail[0];
    try { pick.checked = true; } catch (e) {}
    const target = (pick.labels && pick.labels[0]) ? pick.labels[0] : pick;
    target.click();
    pick.dispatchEvent(new Event('change', { bubbles: true }));
    return { ok: true, kind: 'radio', selected: labelFor(pick) };
  }

  // ---- 3) swatch buttons/links: a group of >=2 size-token clickables ----
  const clickables = Array.from(document.querySelectorAll('button, a[role=button], [role=radio], li, label')).filter(visible);
  const swatches = clickables.filter(el => { const t = norm(el.textContent); return t.length <= 6 && SIZE_TOKEN.test(t); });
  if (swatches.length >= 2) {
    const chosen = swatches.find(el => {
      const cls = (typeof el.className === 'string' ? el.className : '').toLowerCase();
      const on = (el.getAttribute && (el.getAttribute('aria-checked') === 'true' || el.getAttribute('aria-pressed') === 'true')) || /selected|active|is-active/.test(cls);
      return on && !soldOut(el);
    });
    if (chosen) return { ok: false, kind: 'already', selected: norm(chosen.textContent) };
    const avail = swatches.filter(el => !soldOut(el));
    if (!avail.length) return { ok: false, kind: 'size', reason: 'all sizes sold out' };
    let pick = PREF ? avail.find(el => matchesPref(el.textContent)) : null;
    pick = pick || avail[0];
    pick.click();
    return { ok: true, kind: 'swatch', selected: norm(pick.textContent) };
  }

  return { ok: false, kind: 'none' };
}
"""


async def select_required_variant(session, preferred: str | None = None) -> dict:
    """Select a required size/variant on the current product page (in-page, no index) so Add-to-Cart
    isn't blocked by an unselected option. Picks `preferred` when it matches an available option, else
    the first in-stock one. Returns {ok, kind, selected?/reason?} — ok=True when a selection was made;
    kind 'already' (already chosen), 'none' (single-variant product), or 'size'+reason (size required
    but none available). Never raises."""
    js = _SELECT_VARIANT_JS.replace("__PREF__", json.dumps((preferred or "").lower()[:20]))
    val = await evaluate_json(session, js)
    return val if isinstance(val, dict) else {"ok": False, "kind": "none"}


# Click "Proceed to checkout" on the cart page (in-page, no index). Site-aware for Amazon's stable
# control (input[name="proceedToCheckout"] / #sc-buy-box-ptc-button); generic otherwise. CRUCIALLY it
# matches ONLY proceed/continue-to-checkout labels — never "Place your order" / "Buy now", which COMMIT
# a purchase (the agent stops AT the payment page, the user places the order — orders are secured).
_PROCEED_CHECKOUT_JS = r"""
() => {
  const SELS = ['input[name="proceedToCheckout"]', '#sc-buy-box-ptc-button input', '#sc-buy-box-ptc-button',
                '#hlb-ptc-btn-native', 'input[name="proceedToRetailCheckout"]', '[name="proceedToALMCheckout"]',
                '#sc-ptc-button', 'a#hlb-ptc-btn-native'];
  let el = null;
  for (const s of SELS) { el = document.querySelector(s); if (el) break; }
  if (!el) {
    for (const c of document.querySelectorAll('input[type=submit], button, a[role=button], a.a-button-text, a')) {
      const t = (c.value || c.textContent || (c.getAttribute && c.getAttribute('aria-label')) || '').replace(/\s+/g, ' ').trim();
      if (/^(proceed to (checkout|buy)|continue to checkout|go to checkout|checkout)$/i.test(t)) { el = c; break; }
    }
  }
  if (!el) return { ok: false, reason: 'no proceed-to-checkout control on this page' };
  try { el.scrollIntoView({ block: 'center' }); el.click();
        return { ok: true, text: (el.value || el.textContent || 'Proceed to checkout').replace(/\s+/g, ' ').trim().slice(0, 40) };
  } catch (e) { return { ok: false, reason: String(e) }; }
}
"""

# Are we on the checkout / payment-selection / order-review page (NOT the cart)? Signals: a checkout
# pipeline URL, a payment/place-order element, or page text asking to select a payment method / address.
# Used so the checkout subgoal completes the moment the payment page is reached (the human's secured
# step) instead of the reasoner looping on payment radios.
_AT_CHECKOUT_JS = r"""
() => {
  const href = location.href;
  if (/\/gp\/buy\/|\/checkout\/|\/spc\/|\/payments\b|buy\/spc|go-to-checkout|\/gp\/cart\/desktop\/go-to-checkout|alm.*checkout|proceedToCheckout/i.test(href))
    return { ok: true, signal: 'checkout url' };
  const MARK = ['#payment', '#paymentMethod', '#spc-orders', '#submitOrderButtonId', '#placeYourOrder',
                '[name="placeYourOrder1"]', '#turbo-checkout-pyo-button', '#checkout-pyo-button',
                '#puc-overlay', '#payment-information'];
  for (const s of MARK) if (document.querySelector(s)) return { ok: true, signal: 'checkout element ' + s };
  const body = (document.body ? document.body.innerText : '').toLowerCase();
  if (/select a payment method|choose a payment method|how do you want to pay|place your order|delivery address|select a delivery address/.test(body))
    return { ok: true, signal: 'checkout page text' };
  return { ok: false };
}
"""

# Best-effort: select Cash on Delivery on the payment page. Finds a COD radio/option (by value /
# label / aria), selects it, and clicks a "Use this payment method" / "Continue" confirm if present.
# It NEVER clicks "Place your order" — reaching the payment page is the success; COD pre-select is a
# convenience the user asked for, not a commit. Returns {ok, reason?}. Never raises.
_SELECT_COD_JS = r"""
() => {
  const isCOD = (t) => /cash on delivery|\bcod\b|cash\/on\/delivery|instrumentid=[^&]*cash/i.test((t || '').toLowerCase());
  let el = null;
  for (const c of document.querySelectorAll('input[type=radio], [role=radio], label, span, div, a')) {
    const t = ((c.value || '') + ' ' + (c.textContent || '') + ' ' + ((c.getAttribute && (c.getAttribute('aria-label') || c.getAttribute('name'))) || ''));
    if (isCOD(t)) {
      el = (c.tagName === 'INPUT' || (c.getAttribute && c.getAttribute('role') === 'radio')) ? c
           : ((c.querySelector && c.querySelector('input[type=radio],[role=radio]')) || c);
      break;
    }
  }
  if (!el) return { ok: false, reason: 'no Cash on Delivery option on this page' };
  try { el.scrollIntoView({ block: 'center' }); if (el.tagName === 'INPUT') el.checked = true; el.click();
        el.dispatchEvent(new Event('change', { bubbles: true })); } catch (e) {}
  for (const c of document.querySelectorAll('input[type=submit], button, span.a-button-inner input, a[role=button]')) {
    const t = (c.value || c.textContent || '').replace(/\s+/g, ' ').trim();
    if (/^(use this payment method|continue|use this method)$/i.test(t)) { try { c.click(); } catch (e) {} break; }
  }
  return { ok: true };
}
"""


async def proceed_to_checkout(session) -> dict:
    """Click 'Proceed to checkout' on the cart page (in-page). Returns {"ok": True, "text": ...} or
    {"ok": False, "reason": ...}. Never clicks an order-COMMIT button. Never raises."""
    val = await evaluate_json(session, _PROCEED_CHECKOUT_JS)
    return val if isinstance(val, dict) else {"ok": False, "reason": "proceed-to-checkout eval failed"}


async def at_checkout(session) -> tuple[bool, str]:
    """Is the current page the checkout / payment-selection / order-review page (not the cart)?
    Returns (reached, signal). Best-effort — never raises."""
    val = await evaluate_json(session, _AT_CHECKOUT_JS)
    if isinstance(val, dict) and val.get("ok"):
        return True, str(val.get("signal") or "checkout page")
    return False, "not on the checkout/payment page yet"


async def select_cod(session) -> dict:
    """Best-effort select Cash on Delivery on the payment page (in-page). Returns {"ok": bool, ...}.
    NEVER places the order. Never raises."""
    val = await evaluate_json(session, _SELECT_COD_JS)
    return val if isinstance(val, dict) else {"ok": False, "reason": "select-cod eval failed"}


async def cart_confirmed(session, item_title: str | None = None) -> tuple[bool, str]:
    """Was the add-to-cart CONFIRMED on the current page? Checks several signals (badge > 0, a
    post-add confirmation panel, a smart-wagon URL, 'added to cart' text, or the item title) so we
    don't have to navigate to /cart and read its contents. Returns (confirmed, signal/reason).
    Best-effort — never raises."""
    js = _CART_CONFIRMED_JS.replace("__TITLE__", json.dumps((item_title or "").lower()[:60]))
    val = await evaluate_json(session, js)
    if isinstance(val, dict) and val.get("ok"):
        return True, str(val.get("signal") or "cart confirmed")
    return False, "no add-to-cart confirmation visible (no cart badge > 0, confirmation panel, or 'added to cart')"


def _key(row: dict) -> tuple:
    """Dedupe identity: the product URL (sans query) if present, else the lowercased name."""
    url = (row.get("url") or "").split("?", 1)[0].rstrip("/")
    return ("url", url) if url else ("name", (row.get("name") or "").strip().lower())


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


async def extract_candidates_dom(session, explore_spec=None, *, max_items: int = MAX_CANDIDATES) -> list[dict]:
    """Read structured candidate rows from the LIVE DOM (one in-page query, no LLM). Returns a list
    of {name, price, rating, review_count, url, source}; [] on any failure (the caller falls back to
    the LLM a11y extractor). `explore_spec` is accepted for call-site symmetry; the DOM reader needs
    no spec. Never raises."""
    js = _EXTRACT_JS.replace("__MAX__", str(int(max_items)))
    rows = await evaluate_json(session, js)
    if not isinstance(rows, list):
        return []
    out = _dedupe([r for r in rows if isinstance(r, dict) and (r.get("name") or r.get("url"))])
    log.debug("dom extractor: %d candidate row(s) on this page", len(out))
    return out[:max_items]


async def extract_with_scroll(session, explore_spec=None, *, max_items: int = MAX_CANDIDATES,
                              max_scrolls: int = 8) -> list[dict]:
    """Fix 3 — bounded scroll-to-reveal for lazy grids. Extract, scroll, re-extract until a scroll
    reveals no new candidates or the scroll budget is hit; dedupe across passes. The candidate
    SELECTOR (not the a11y snapshot) is what we re-read, so lazy loading no longer hides products.
    Best-effort: returns whatever was gathered; never raises."""
    # Local import avoids a module cycle (dom_index doesn't import this, but keep it lazy + cheap).
    from browser_agent.agent.dom_index import scroll_page

    seen: dict = {}
    for r in await extract_candidates_dom(session, explore_spec, max_items=max_items):
        seen.setdefault(_key(r), r)
    for _ in range(max(0, max_scrolls)):
        if len(seen) >= max_items:
            break
        before = len(seen)
        try:
            await scroll_page(session, "down")
            await asyncio.sleep(_SCROLL_SETTLE_S)  # let lazy content load before re-reading
        except Exception:  # noqa: BLE001 — scroll is best-effort; extract what we have
            break
        for r in await extract_candidates_dom(session, explore_spec, max_items=max_items):
            seen.setdefault(_key(r), r)
        if len(seen) == before:        # a dry scroll -> no more lazy content is loading
            break
    out = list(seen.values())[:max_items]
    log.debug("dom extractor (scroll): %d candidate row(s) after scroll-to-reveal", len(out))
    return out

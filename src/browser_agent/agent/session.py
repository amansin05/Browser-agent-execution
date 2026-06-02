"""AgentSession — a long-lived session: ONE persistent Playwright MCP browser + session memory,
many tasks.

Tab parking: when a task finishes we CLOSE the working tabs it opened (the Playwright MCP tab
group) instead of leaving them open, but first record their URLs. On the next task we ask a cheap
LLM whether it's a FOLLOW-UP of the previous one; if so we reopen those same tabs so the agent
resumes where it left off, otherwise the parked URLs are dropped and the task starts clean. The
chat/origin tab (the one the user was on before automation) is never parked or closed.

The browser is torn down only on close() — which, in smoke mode, closes the browser/tabs, and
in extension mode merely disconnects (your real Chrome is never closed by us).
"""

from browser_agent.agent.context import build_basic_context, context_text
from browser_agent.agent.flat import agent_loop
from browser_agent.agent.in_page_prompt import approve_in_page, ask_in_page
from browser_agent.agent.orchestrate import orchestrate
from browser_agent.agent.prompts import FOLLOWUP_SYSTEM
from browser_agent.agent.reflect import reflect
from browser_agent.agent.tools import build_reasoner_tools, flat_tools
from browser_agent.config import MODEL
from browser_agent.log import EventRecorder, get_logger
from browser_agent.services.groq_service import make_groq_client
from browser_agent.services.mcp_client import (
    open_session, parse_tabs, tool_result_to_text, wait_for_real_tab,
)
from browser_agent.utils.io import cli_approve, cli_ask, maybe_await
from browser_agent.utils.text import extract_json

log = get_logger(__name__)


def _same_page(a: str, b: str) -> bool:
    """Lenient URL match for tab identity: ignore trailing slashes and #hash routes so the chat
    tab still matches after the SPA appends a route (e.g. localhost:5173 vs localhost:5173/#/run)."""
    if not a or not b:
        return False
    norm = lambda u: u.split("#", 1)[0].rstrip("/").lower()
    na, nb = norm(a), norm(b)
    return na == nb or na.startswith(nb) or nb.startswith(na)


class AgentSession:
    def __init__(self, *, use_extension=False, browser="chrome", pick_tab=False,
                 memory=None, profile=None, approve=cli_approve, ask=cli_ask, focus_url=None):
        self.use_extension = use_extension
        self.browser = browser
        self.pick_tab = pick_tab
        self.memory = memory          # optional MemorySession (rolling context + scratchpad)
        self.profile = profile        # optional ProfileMemory (persona/preferences/history)
        self.approve = approve
        self.ask = ask
        self.focus_url = focus_url     # chat-UI tab URL to switch back to when asking the human
        self.groq = make_groq_client()
        self._cm = None
        self.session = None
        self.params = None
        self.reasoner_tools: list[dict] = []
        self.capabilities: list[str] = []
        self.flat_tools: list[dict] = []
        # (idx, url) of the tab that was active right BEFORE the agent started navigating — the
        # tab the user was looking at. We switch back to THIS to surface a prompt / land the result,
        # which is more robust than string-matching focus_url (the chat URL may be formatted
        # differently in the tab list). Captured once, on the first task of the session.
        self._origin_tab: tuple[int, str] | None = None
        # Tab parking: URLs of the working tabs the LAST task left open. We closed them at the end
        # of that task and stashed the URLs here; if the NEXT task is a follow-up we reopen them.
        self._parked_tabs: list[str] = []
        # The previous task's text — used only to classify whether the next task is a follow-up.
        self._last_task: str | None = None

    async def open(self) -> "AgentSession":
        """Launch (or attach to) the browser once and cache the tool schemas."""
        self._cm = open_session(self.use_extension, self.browser)
        self.session, self.params = await self._cm.__aenter__()
        log.info("launched Playwright MCP: %s %s", self.params.command, " ".join(self.params.args))
        listed = await self.session.list_tools()
        self.reasoner_tools, self.capabilities = build_reasoner_tools(listed.tools)
        self.flat_tools = flat_tools(listed.tools)
        if self.use_extension:
            if self.pick_tab:
                await wait_for_real_tab(self.session)
            log.info("extension: working in a DEDICATED tab; your picked tab is preserved")
            # Capture the tab you're on BEFORE opening our own — it's the origin we never close and
            # land back on when the run ends. (run_task's later capture becomes a no-op.)
            self._origin_tab = await self._current_tab()
            if self._origin_tab:
                log.info("remembered origin tab #%s: %s", self._origin_tab[0], self._origin_tab[1][:80])
            await self._ensure_working_tab()
        return self

    async def _ensure_working_tab(self) -> None:
        """Extension mode: guarantee the agent has its OWN tab to drive, so it never navigates or
        closes the tab you were on. The origin tab (and any chat/dev-ui tab) is preserved; a fresh
        working tab is opened + selected only when the agent has NO working tab yet — at session
        start, or after a previous task parked (closed) its working tabs and the new task isn't a
        follow-up. No-op when a working tab already exists (e.g. parked tabs were just reopened)."""
        if not (self.session and self.use_extension):
            return
        try:
            tabs = await self._list_tabs()
        except Exception:
            return
        keep = self._find_chat_tab(tabs)
        keep_idx = keep[0] if keep else None
        origin_idx = self._origin_tab[0] if self._origin_tab else None
        has_working = any(
            idx not in (keep_idx, origin_idx) and not _same_page(url, self.focus_url)
            for idx, _is_cur, url in tabs)
        if has_working:
            return  # the agent already has a tab to work in (fresh or restored)
        try:
            before = {i for i, _, _ in tabs}
            await self.session.call_tool("browser_tabs", {"action": "new"})
            after = await self._list_tabs()
            new = next((i for i, _, _ in after if i not in before), None)
            if new is not None:
                await self.session.call_tool("browser_tabs", {"action": "select", "index": new})
                log.info("opened a dedicated working tab #%s (origin tab #%s preserved)", new, origin_idx)
        except Exception as e:
            log.warning("could not open a dedicated working tab (%r); driving the current tab", e)

    async def run_task(self, task: str, *, agent="two-tier", allow=None, max_steps=12,
                       max_replans=3, read_content=True, ground=True, approve=None, ask=None,
                       emit=None, trace=True) -> str | None:
        """Run one task against the persistent browser. Memory (if attached) seeds the planner
        and is updated with the task/result + any observations afterward. When `trace` is set, every
        emit event is also appended to a per-run JSONL trace file under the log dir (set trace=False
        when the caller already records its own trace, e.g. the server)."""
        if self.session is None:
            raise RuntimeError("AgentSession.open() must be called first")
        approve = approve or self.approve
        ask = ask or self.ask
        observations: list[str] = []

        # Per-run trace: append every emit event to a replayable JSONL file under the log dir, so a
        # CLI / programmatic run leaves the same debuggable trail the server already records. The
        # caller's emit (if any) still fires — we wrap it. Run boundaries are written straight to the
        # file (not through emit) so we don't echo extra events to a UI that emits its own.
        recorder = EventRecorder(getattr(self.memory, "id", None) or "cli") if trace else None
        if recorder and recorder.path:
            log.info("trace -> %s", recorder.path)
        _caller_emit = emit

        def emit(event):  # noqa: F811 — shadow the param with a wrapper that also records the trace
            if recorder:
                recorder.record(event)
            if _caller_emit:
                _caller_emit(event)

        # Remember the tab the user was on before any automation, so a pause/finish can return
        # there. Captured on the first task only — later tasks reuse the same browsing tabs.
        if self.use_extension and self._origin_tab is None:
            self._origin_tab = await self._current_tab()
            if self._origin_tab:
                log.info("remembered origin tab #%s: %s", self._origin_tab[0], self._origin_tab[1][:80])

        # Tab parking, resume side: the previous task closed its working tabs and stashed their
        # URLs in self._parked_tabs. If THIS task is a follow-up of the last one, reopen them so the
        # agent picks up where it left off; otherwise drop them and start clean. Best-effort — any
        # failure leaves us with no parked tabs and the task just runs from the current page.
        if self._parked_tabs:
            try:
                if await self._is_followup(task):
                    reopened = await self._restore_parked_tabs()
                    if reopened:
                        emit({"type": "tabs_reopened", "urls": reopened})
                else:
                    log.info("new task isn't a follow-up; discarding %d parked tab(s)",
                             len(self._parked_tabs))
                    self._parked_tabs = []
            except Exception as e:
                log.warning("tab restore step failed (%r); continuing without reopening", e)
                self._parked_tabs = []
        self._last_task = task

        # Make sure the agent has its own working tab so it never drives/closes your origin tab.
        # (A follow-up just reopened its parked tabs -> no-op; otherwise open a fresh one, since the
        # previous task parked & closed its working tab.)
        if self.use_extension:
            await self._ensure_working_tab()

        # How we reach the human:
        #   1. In extension mode, render the prompt as an OVERLAY in the tab the agent is driving —
        #      you're already looking at it, so nothing has to switch (the dev-ui is unreachable;
        #      it lives outside Playwright's tab group). See in_page_prompt.
        #   2. If that can't render (e.g. a chrome-extension:// page) or times out, fall back to the
        #      WebSocket modal in the dev-ui (the focus_chat_tab dance is best-effort for that).
        paused_for_human = False

        async def approve_w(prompt):
            nonlocal paused_for_human
            paused_for_human = True
            if self.use_extension:
                verdict = await approve_in_page(self.session, prompt)
                if verdict is not None:
                    return verdict
            working = await self.focus_chat_tab()
            try:
                return await maybe_await(approve(prompt))
            finally:
                await self._restore_tab(working)

        async def ask_w(question, **kw):
            nonlocal paused_for_human
            paused_for_human = True
            if self.use_extension:
                answer = await ask_in_page(
                    self.session, question, options=kw.get("options"),
                    allow_free_text=kw.get("allow_free_text", True),
                    multi_select=kw.get("multi_select", False))
                if answer is not None:
                    return answer
            working = await self.focus_chat_tab()
            try:
                return await maybe_await(ask(question, **kw))
            finally:
                await self._restore_tab(working)

        if recorder:
            recorder.record({"type": "run_started", "agent": agent, "task": task})
        try:
            try:
                if agent == "flat":
                    history = self.memory.history_messages() if self.memory else None
                    result = await agent_loop(self.groq, self.session, self.flat_tools, task,
                                              max_steps, emit=emit, history=history, ask=ask_w)
                else:
                    allowlist = {d.strip().lower() for d in (allow or []) if d.strip()}
                    result = await orchestrate(
                        self.groq, self.session, self.reasoner_tools, self.capabilities, task,
                        allowlist=allowlist, approve=approve_w, ask=ask_w, max_steps=max_steps,
                        max_replans=max_replans, read_content=read_content, ground=ground, emit=emit,
                        preamble=self._planner_preamble(task), observations=observations)
            finally:
                # Whole flow done — if we paused for a human at any point, land on the chat-UI tab
                # (the one in front of the user before the run started) so the result is visible and
                # the next question continues from there. If no question was raised, leave focus on
                # the working tab. Either way, the working tab stays open for the follow-up.
                if paused_for_human:
                    await self.focus_chat_tab()

            if self.memory:
                self.memory.record_task(task, result)
                for note in observations[-8:]:
                    self.memory.note(note)
            if self.profile:
                await self._learn(task, result, observations, emit)
            # Tab parking, close side: don't leave the working tabs open. Record their URLs and
            # close them; a follow-up task will reopen them. Never touches the chat/origin tab.
            try:
                parked = await self._park_working_tabs()
                if parked:
                    emit({"type": "tabs_parked", "urls": parked})
            except Exception as e:
                log.warning("tab parking failed (%r); leaving working tabs open", e)
            if recorder:
                recorder.record({"type": "run_finished", "result": result})
            return result
        except Exception as e:  # record the failure into the trace, then propagate unchanged
            if recorder:
                recorder.record({"type": "run_error", "message": f"{type(e).__name__}: {e}"})
            raise
        finally:
            if recorder:
                recorder.close()

    async def _list_tabs(self):
        res = await self.session.call_tool("browser_tabs", {"action": "list"})
        return parse_tabs(tool_result_to_text(res))

    async def _current_tab(self) -> tuple[int, str] | None:
        """(idx, url) of the currently-focused tab the extension can see, or None."""
        try:
            tabs = await self._list_tabs()
        except Exception:
            return None
        return next(((idx, url) for idx, is_cur, url in tabs if is_cur), None)

    async def _working_tabs(self) -> list[tuple[int, str]]:
        """The agent's working tabs — every http(s) tab in the Playwright MCP group EXCEPT the
        chat/origin tab (the one the user was on before automation, which we never close). These
        are 'the group of tabs' that get parked & reopened. Returns [] when there's no identifiable
        keep tab (e.g. CLI/smoke with no origin), which deliberately disables parking there so we
        never close the last/only tab and orphan the browser context."""
        try:
            tabs = await self._list_tabs()
        except Exception:
            return []
        keep = self._find_chat_tab(tabs)
        if keep is None:
            return []
        keep_idx = keep[0]
        origin_idx = self._origin_tab[0] if self._origin_tab else None
        out = []
        for idx, _is_cur, url in tabs:
            if idx == keep_idx or idx == origin_idx:
                continue
            if _same_page(url, self.focus_url):  # any chat-UI tab, regardless of index
                continue
            if not url.startswith("http"):       # skip extension/picker/blank pages
                continue
            out.append((idx, url))
        return out

    async def _park_working_tabs(self) -> list[str]:
        """Record the working tabs' URLs, then close those tabs. Returns the parked URLs (or [] if
        there was nothing to park). Closes highest index first so the lower indices we still hold
        don't shift mid-loop. The chat/origin tab is left untouched, so focus falls back to it."""
        tabs = await self._working_tabs()
        if not tabs:
            self._parked_tabs = []
            return []
        self._parked_tabs = [url for _, url in tabs]
        if self.memory:
            self.memory.note("parked tabs (reopened on a follow-up): "
                             + " | ".join(self._parked_tabs), key="parked_tabs")
        for idx, _url in sorted(tabs, key=lambda t: t[0], reverse=True):
            try:
                await self.session.call_tool("browser_tabs", {"action": "close", "index": idx})
            except Exception as e:
                log.warning("could not close working tab #%s while parking (%r)", idx, e)
        log.info("parked + closed %d working tab(s): %s",
                 len(self._parked_tabs), [u[:60] for u in self._parked_tabs])
        return list(self._parked_tabs)

    async def _restore_parked_tabs(self) -> list[str]:
        """Reopen the parked URLs, one fresh tab each, and focus the first. Clears the parked list.
        Returns the URLs actually reopened. Leaves the chat/origin tab in place."""
        urls = self._parked_tabs
        self._parked_tabs = []
        reopened, first_new = [], None
        for url in urls:
            try:
                before = {i for i, _, _ in await self._list_tabs()}
                await self.session.call_tool("browser_tabs", {"action": "new"})
                after = await self._list_tabs()
                new = next((i for i, _, _ in after if i not in before), None)
                if new is None:
                    continue
                await self.session.call_tool("browser_tabs", {"action": "select", "index": new})
                await self.session.call_tool("browser_navigate", {"url": url})
                reopened.append(url)
                if first_new is None:
                    first_new = new
            except Exception as e:
                log.warning("could not reopen parked tab %s (%r)", url[:60], e)
        if first_new is not None:
            try:
                await self.session.call_tool("browser_tabs", {"action": "select", "index": first_new})
            except Exception:
                pass
        log.info("reopened %d parked tab(s) for follow-up: %s",
                 len(reopened), [u[:60] for u in reopened])
        return reopened

    async def _is_followup(self, task: str) -> bool:
        """Decide whether `task` continues the previous task (so we reopen the parked tabs) vs.
        starts something unrelated. A cheap yes/no LLM call. Defaults to True — with no prior task,
        no Groq client, or on any failure — because the user just had those pages open and losing
        that context is worse than reopening a few tabs the agent can ignore."""
        prev = self._last_task
        if not prev or self.groq is None:
            return True
        pages = ", ".join(u[:80] for u in self._parked_tabs) or "(none)"
        user = (f"Previous task: {prev}\n"
                f"Pages it left open: {pages}\n"
                f"New task: {task}\n\n"
                "Is the new task a follow-up that should reopen those pages?")
        try:
            resp = await self.groq.chat.completions.create(
                model=MODEL, temperature=0.0, max_completion_tokens=30,
                messages=[{"role": "system", "content": FOLLOWUP_SYSTEM},
                          {"role": "user", "content": user}])
            data = extract_json(resp.choices[0].message.content)
        except Exception as e:
            log.warning("follow-up classification failed (%r); assuming follow-up", e)
            return True
        return bool(data.get("followup", True)) if isinstance(data, dict) else True

    def _find_chat_tab(self, tabs):
        """Locate the tab to surface a prompt on. Prefer one matching the chat URL the dev-ui
        reported; fall back to the origin tab (the one the user was on before automation) — by URL
        first, then by its recorded index. Returns (idx, url) or None. The fallback is what makes
        'switch back to the tab that was there before automation' work even when the chat URL
        doesn't string-match a tab in the list."""
        chat = next(((idx, url) for idx, _, url in tabs if _same_page(url, self.focus_url)), None)
        if chat is not None or not self._origin_tab:
            return chat
        oidx, ourl = self._origin_tab
        by_url = next(((i, u) for i, _, u in tabs if _same_page(u, ourl)), None)
        if by_url is not None:
            return by_url
        if any(i == oidx for i, _, _ in tabs):  # url moved on, but the slot is still there
            return (oidx, ourl)
        return None

    async def focus_chat_tab(self):
        """Switch the live browser to the chat-UI tab (extension mode only) so a pause for
        approval / a question is actually seen by the user.

        Returns (index, url) of the tab that was current beforehand — the agent's working tab —
        so the caller can switch back after the human answers. Returns None when no switch was
        needed (not extension mode, no chat URL, chat tab missing, or already on it). Never
        closes any tab."""
        if not (self.use_extension and self.focus_url and self.session):
            return None
        try:
            tabs = await self._list_tabs()
        except Exception:
            return None
        working = next(((idx, url) for idx, is_current, url in tabs if is_current), None)
        chat = self._find_chat_tab(tabs)
        if chat is None:
            # Neither the chat-UI tab nor the origin tab is visible to the extension. In --extension
            # mode a tab only shows up in browser_tabs once connected via connect.html (see README),
            # so a tab merely open in Chrome can't be raised. This is NOT fatal: the question is
            # still delivered to the dev-ui over the WebSocket (ask_request) and answered there.
            log.info("focus: no reachable chat/origin tab in browser_tabs (%s); the question still "
                     "appears in the dev-ui to answer", [u for _, _, u in tabs])
            return None
        if working and working[0] == chat[0]:
            return None  # already on the target tab — nothing to restore
        try:
            await self.session.call_tool("browser_tabs", {"action": "select", "index": chat[0]})
            log.info("focus: switched to tab #%s to surface a prompt", chat[0])
        except Exception:
            return None
        return working

    async def _restore_tab(self, working) -> None:
        """Switch focus back to the agent's working tab after a human pause. Matches by URL first
        (robust to index shifts), falling back to the original index. Only refocuses — never
        closes — so the tab's context is kept for the agent to continue from."""
        if not working or not self.session:
            return
        idx, url = working
        try:
            tabs = await self._list_tabs()
            target = next((i for i, _, u in tabs if _same_page(u, url)), None)
            if target is None and any(i == idx for i, _, _ in tabs):
                target = idx
            if target is not None:
                await self.session.call_tool("browser_tabs", {"action": "select", "index": target})
        except Exception:
            pass

    def _planner_preamble(self, task: str) -> str:
        """Assemble the planner's memory/context preamble: Tier-1 basic context + persona &
        preferences + relevant history + this session's rolling context."""
        parts = [context_text(build_basic_context(self.profile))]
        if self.profile:
            parts.append(self.profile.core_blocks_text())
            parts.append(self.profile.history_text(task))
        if self.memory:
            parts.append(self.memory.preamble())
        return "\n\n".join(p for p in parts if p)

    async def _learn(self, task, result, observations, emit) -> None:
        """Memory-update step (B2/M11): append an episode and recency-weight preferences."""
        r = await reflect(self.groq, task, result, observations)
        self.profile.append_episode(task, r.get("outcome", "unknown"), chosen=r.get("chosen"),
                                    rejected=r.get("rejected"), on_time=r.get("on_time"))
        self.profile.refine_preferences(r.get("preference_updates", {}))
        self.profile.prune_history()
        if emit:
            emit({"type": "memory_update", "outcome": r.get("outcome"),
                  "preference_updates": r.get("preference_updates", {})})

    async def close_browser(self) -> None:
        """Close the browser/connection but KEEP memory (used when reopening in a new mode)."""
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self.session = None

    async def close(self) -> None:
        """End the session entirely: close the browser (smoke) / disconnect (extension), and
        finalize memory. Tabs are only closed here — never between tasks."""
        await self.close_browser()
        if self.memory:
            self.memory.close()

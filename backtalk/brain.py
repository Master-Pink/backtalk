# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The warm brain — a persistent Claude session via the Agent SDK,
streaming.

One ClaudeSDKClient lives for the whole voice session: no per-turn
process spawn, no per-turn context reload. Partial-message streaming
means sentences are yielded the moment they're complete, so the mouth
starts speaking while the rest of the thought is still forming.

The session's cwd is YOUR agent's folder (agent_dir in backtalk.json) —
whatever CLAUDE.md lives there defines who is speaking. backtalk adds
only the spoken-delivery discipline (config.DISCIPLINE): the medium,
never the character.
"""
import asyncio
import os
import re
import warnings
from datetime import datetime

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

try:
    from claude_agent_sdk import CanUseToolShadowedWarning
except ImportError:                       # older SDKs: nothing to silence
    CanUseToolShadowedWarning = None

from backtalk import signals
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")


# --- usage-limit / dead-turn detection -------------------------------
# When the plan hits its ceiling mid-turn the model call stops without a
# real answer, and depending on the SDK build that surfaces as a lone
# complete AssistantMessage ("Claude AI usage limit reached|<ts>"), a
# ResultMessage flagged is_error, a raised exception, or the response
# stream simply exhausting with nothing in it. Left unspoken it looks
# exactly like a hang: the face sticks on "thinking" and every later
# message drains as "0 stale messages". So name it out loud instead.
_LIMIT_MARKERS = (
    "usage limit reached",
    "claude ai usage limit",
    "rate limit exceeded",
    "rate_limit_error",
    "quota exceeded",
    "credit balance is too low",
    "insufficient credit",
)
# A transient server condition, NOT the plan's ceiling — it clears on its
# own in seconds. Kept OUT of _LIMIT_MARKERS so it stops being announced
# as "you've hit the usage limit". "resets_at" came out of the markers
# too: it's a field name in the healthy get_usage telemetry JSON, not a
# limit signal, and _reset_clause has its own regex for a real reset time.
_OVERLOAD_MARKERS = ("overloaded_error", "overloaded")


def _looks_like_limit(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _LIMIT_MARKERS)


def _looks_like_overload(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _OVERLOAD_MARKERS)


def _overload_sentence() -> str:
    return ("Anthropic's servers are briefly overloaded — that's not a "
            "usage limit. Give it a few seconds and ask again.")


def _reset_clause(text: str) -> str:
    """" It should reset around 3:00 PM." when the CLI handed back a
    reset time — it appends "|<unix ts>" to a usage-limit message, and
    the API error spells it resets_at. Empty string when there's none."""
    m = re.search(r"\|\s*(\d{9,13})\b", text or "")
    if not m:
        m = re.search(r"resets?_?at['\"]?\s*[:=]\s*['\"]?(\d{9,13})",
                      (text or "").lower())
    if not m:
        return ""
    try:
        ts = int(m.group(1))
        if ts > 1_000_000_000_000:          # milliseconds -> seconds
            ts //= 1000
        when = datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")
        return f" It should reset around {when}."
    except Exception:
        return ""


def _limit_sentence(text: str = "") -> str:
    return ("I've hit the Claude usage limit, so I can't answer until "
            "the plan resets." + _reset_clause(text))


class WarmBrain:
    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None):
        # Full model id ON PURPOSE — never a bare alias. The SDK
        # resolves aliases through its own bundled CLI and can silently
        # land on an older model.
        self.model = model or CFG["model"]
        # The spoken permission gate (main.py builds it). Wired at
        # connect in EVERY mode, so a live mode flip needs no reconnect;
        # bypass simply never consults it.
        self._can_use_tool = can_use_tool
        # Session usage, spoken on request ("usage report").
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._client: ClaudeSDKClient | None = None
        # The session to reattach to at the FIRST start only (config key
        # resume_last_session). Consumed on use: a desync rebuild in
        # reset_turn() must always start FRESH: a rebuild means a turn
        # went sideways mid-stream, the wrong moment to gamble on
        # reattaching. (Community proposal, issue #1.)
        self._resume_id = resume_id
        # True while a query's response hasn't been consumed through its
        # ResultMessage — i.e. the shared message pipe may hold leftovers.
        self._dirty = False
        # Set when the last ask_stream turn came back with no real answer
        # (usage limit, error result, empty stream). Cleared at the start
        # of every turn. The startup warmup checks it so an out-of-usage
        # launch still fails loudly instead of booting "warm".
        self._last_turn_failed = False

    async def start(self):
        mode = CFG["permission_mode"]
        if mode == "default":
            mode = "ask"     # legacy alias, see config.py
        # backtalk's "ask" = the SDK's "default" mode with gated calls
        # routed to the spoken can_use_tool gate.
        sdk_mode = "default" if mode == "ask" else mode
        if sdk_mode == "bypassPermissions" and self._can_use_tool \
                and CanUseToolShadowedWarning:
            # Deliberate auto-approve: the SDK warns that the callback is
            # shadowed. That IS the chosen behavior, so boot quietly.
            warnings.filterwarnings("ignore",
                                    category=CanUseToolShadowedWarning)
        resume, self._resume_id = self._resume_id, None   # consume once

        def _opts(rid):
            return ClaudeAgentOptions(
                cwd=CFG["agent_dir"],
                model=self.model,
                system_prompt={"type": "preset", "preset": "claude_code",
                               "append": DISCIPLINE},
                include_partial_messages=True,
                permission_mode=sdk_mode,
                can_use_tool=self._can_use_tool,
                add_dirs=CFG["extra_dirs"],
                skills=CFG["visible_skills"],
                resume=rid,
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                log(f"[brain] resumed session {resume[:8]}")
                return
            except Exception as e:
                # a stale or invalid saved session must never brick the
                # launch. Fall back to a fresh conversation and say so.
                log(f"[brain] resume failed ({str(e)[:80]}), "
                    f"starting fresh")
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
        self._client = ClaudeSDKClient(options=_opts(None))
        await self._client.connect()

    async def set_permission_mode(self, backtalk_mode: str):
        """Live flip, no reconnect, conversation intact ("ask" maps to
        the SDK's "default", whose gated calls hit the spoken gate)."""
        if self._client:
            sdk_mode = "default" if backtalk_mode == "ask" \
                else backtalk_mode
            await self._client.set_permission_mode(sdk_mode)

    async def context_usage(self):
        """The CLI's own context-window breakdown, or None."""
        try:
            return await self._client.get_context_usage()
        except Exception:
            return None

    def _remember_session(self, rm):
        """Persist the session id after a completed turn, so the next
        launch can reattach (config: resume_last_session). Must never
        break a turn; silence on any failure."""
        if not CFG.get("resume_last_session"):
            return
        sid = getattr(rm, "session_id", None)
        if not sid:
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(sid)
        except OSError:
            pass

    def _tally(self, rm, count_turn=True):
        """Session usage bookkeeping. Must never break a turn."""
        try:
            u = getattr(rm, "usage", None) or {}
            s = self.session
            if count_turn:
                s["turns"] += 1
            s["out_tokens"] += int(u.get("output_tokens") or 0)
            s["in_tokens"] += (int(u.get("input_tokens") or 0)
                               + int(u.get("cache_read_input_tokens")
                                     or 0))
            c = getattr(rm, "total_cost_usd", None)
            if c:
                s["cost"] += float(c)
        except Exception:
            pass

    async def _pull_rate_limits(self):
        """Ask the CLI outright how much of the plan is spent.

        A DIRECT QUERY, not the RateLimitEvent stream. The event fires
        rarely and usually arrives carrying resets_at with no utilization
        at all, so a listener built on it reports nothing most of the
        time -- which is exactly how this feature looked broken for its
        whole life. (Community fix, ai-visualizer issue #1.)

        THIS REACHES PAST THE SDK'S PUBLIC SURFACE ON PURPOSE, and a
        reader should know it rather than discover it. `get_usage` is a
        control request the bundled CLI answers but the SDK never wraps,
        so there is no supported call to make. The supported-looking
        alternative is a dead end and was tested as one: the terminal
        status line never fires in a headless session, so its numbers
        are unreachable from here.

        Which means this can stop working without anyone doing anything
        wrong, and the containment is the point. Every failure is
        swallowed and the readout simply goes quiet. It must never cost
        a turn, so it is also bounded -- an unanswered control request
        would otherwise hang the voice line mid-conversation."""
        if not CFG.get("show_usage"):
            return
        try:
            usage = await asyncio.wait_for(
                self._client._query._send_control_request(
                    {"subtype": "get_usage"}), 5)
            for window in ("five_hour", "seven_day"):
                w = (usage.get("rate_limits") or {}).get(window)
                if not w:
                    continue
                # Two spellings accepted deliberately: this shape is not
                # documented anywhere, so the cheap tolerance is worth
                # more than the tidiness. Both are percentages, and the
                # rest of the pipeline wants a 0..1 fraction.
                pct = w.get("utilization")
                if pct is None:
                    pct = w.get("used_percentage")
                pct = pct / 100 if pct is not None else None
                resets = w.get("resets_at")
                if isinstance(resets, str):
                    resets = int(datetime.fromisoformat(resets).timestamp())
                signals.set_rate_limit(window, pct, resets)
        except Exception:
            pass

    async def command(self, cmd: str) -> str:
        """Run a console slash command (/clear, /compact, /model,
        /effort) through the normal stream and return whatever text the
        CLI answered with (confirmations, errors). Slash-command replies
        arrive as COMPLETE AssistantMessages, not stream deltas, so
        ask_stream cannot see them. Bounded like reset_turn is: this
        stream is not trusted to always deliver, and an unbounded await
        here would deafen the whole voice loop. On timeout the pipe is
        left marked dirty so the next reset_turn drains or rebuilds."""
        self._dirty = True
        await self._client.query(cmd)
        texts = []

        async def _collect():
            async for msg in self._client.receive_response():
                t = type(msg).__name__
                if t == "AssistantMessage":
                    for b in getattr(msg, "content", []) or []:
                        txt = getattr(b, "text", None)
                        if txt:
                            texts.append(txt)
                elif t == "ResultMessage":
                    self._dirty = False
                    self._tally(msg, count_turn=False)
                    self._remember_session(msg)
                    break

        try:
            await asyncio.wait_for(_collect(), 90)
        except asyncio.TimeoutError:
            log(f"[brain] console command timed out: {cmd!r}")
            return "error: the command timed out"
        return " ".join(texts).strip()

    async def interrupt(self):
        if self._client:
            await self._client.interrupt()

    async def _rebuild(self):
        """Drop a wedged client and start a fresh session. Loses this
        voice session's in-memory conversation — the alternative is a
        client that answers nothing until someone restarts backtalk."""
        try:
            await self._client.disconnect()
        except Exception:
            pass
        self._client = None
        await self.start()
        self._dirty = False

    async def reset_turn(self, timeout: float = 8.0):
        """Re-align the message pipe after an interrupted/failed turn.

        THE OFF-BY-ONE BUG, and why this method exists: the SDK client
        has ONE shared message stream and receive_response() stops at
        the FIRST ResultMessage it sees — there is no pairing between a
        query and its response. A cancelled turn stops consuming
        mid-stream, leaving the dead turn's remaining messages
        (including its ResultMessage) buffered. The next query then
        pairs with those leftovers: the first ask lands on the stale
        ResultMessage and yields nothing, and every ask after that
        answers the PREVIOUS question — for the rest of the session.
        So: interrupt the dead turn, then drain the pipe through its
        stale ResultMessage before the next query goes out. No-op when
        the last turn was consumed clean."""
        if not self._client or not self._dirty:
            return
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            pass  # turn may already be over — the drain below is the point

        async def _drain() -> int:
            n = 0
            async for msg in self._client.receive_response():
                n += 1
                if type(msg).__name__ == "ResultMessage":
                    break
            return n

        try:
            drained = await asyncio.wait_for(_drain(), timeout)
            log(f"[brain] interrupted turn drained ({drained} stale messages)")
            self._dirty = False
        except Exception:
            # Can't re-align — rebuild the session rather than run
            # desynced. Loses this voice session's conversation memory;
            # better than answering every question one turn late for the
            # rest of the day.
            log("[brain] stream desynced beyond repair — rebuilding the "
                "session (conversation memory for this session resets)")
            await self._rebuild()

    async def stop(self):
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model."""
        self._dirty = True             # in flight until its ResultMessage
        self._last_turn_failed = False
        await self._client.query(utterance)
        buf = ""
        yielded = False                # any real answer text went out
        saw_result = False             # the turn reached its ResultMessage
        try:
            async for msg in self._client.receive_response():
                t = type(msg).__name__
                if t == "StreamEvent":
                    ev = getattr(msg, "event", {}) or {}
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {}) or {}
                        if delta.get("type") == "text_delta":
                            buf += delta.get("text", "")
                            # emit any complete sentences
                            while True:
                                m = _SENTENCE_END.search(buf)
                                if not m:
                                    break
                                sentence, buf = (buf[:m.end()].strip(),
                                                 buf[m.end():])
                                if sentence:
                                    yielded = True
                                    yield sentence
                    elif ev.get("type") == "content_block_stop":
                        # End of a speech block (e.g. right before a tool
                        # call): flush NOW. Without this, pre-tool filler
                        # ("On it — let me grab that.") sits silent in the
                        # buffer through the whole tool run, then plays
                        # glued to the answer: long dead air, then two
                        # thoughts at once.
                        tail = buf.strip()
                        buf = ""
                        if tail:
                            yielded = True
                            yield tail
                elif t == "AssistantMessage" and not yielded and not buf:
                    # A usage-limit stop can arrive as ONE complete
                    # assistant message with no stream deltas at all —
                    # the StreamEvent branch never sees it. Only trust it
                    # when nothing has streamed this turn.
                    whole = " ".join(
                        getattr(b, "text", "") or ""
                        for b in getattr(msg, "content", []) or []).strip()
                    if _looks_like_limit(whole):
                        log(f"[brain] usage limit reached (assistant "
                            f"message): {whole[:160]}")
                        self._last_turn_failed = True
                        yielded = True
                        yield _limit_sentence(whole)
                    elif _looks_like_overload(whole):
                        log(f"[brain] server overload (assistant "
                            f"message): {whole[:160]}")
                        self._last_turn_failed = True
                        yielded = True
                        yield _overload_sentence()
                elif t == "ResultMessage":
                    saw_result = True
                    self._dirty = False   # turn consumed — pipe aligned
                    self._tally(msg)
                    self._remember_session(msg)
                    await self._pull_rate_limits()
                    if not yielded:
                        detail = str(getattr(msg, "result", "")
                                     or getattr(msg, "subtype", "") or "")
                        if _looks_like_overload(detail):
                            log(f"[brain] turn ended on server overload: "
                                f"{detail[:160]}")
                            self._last_turn_failed = True
                            yielded = True
                            yield _overload_sentence()
                        elif _looks_like_limit(detail):
                            log(f"[brain] turn ended at usage limit: "
                                f"{detail[:160]}")
                            self._last_turn_failed = True
                            yielded = True
                            yield _limit_sentence(detail)
                        elif getattr(msg, "is_error", False):
                            log(f"[brain] turn ended with an error: "
                                f"{detail[:160]}")
                            self._last_turn_failed = True
                            yielded = True
                            yield ("That turn ended with an error before "
                                   "I could get you an answer. Check "
                                   "this window for the details.")
                    break
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as e:
            if _looks_like_overload(repr(e)):
                log(f"[brain] server overload: {str(e)[:200]}")
                self._last_turn_failed = True
                yield _overload_sentence()
                return
            if _looks_like_limit(repr(e)):
                log(f"[brain] usage limit reached: {str(e)[:200]}")
                self._last_turn_failed = True
                yield _limit_sentence(repr(e))
                return
            raise
        tail = buf.strip()
        if tail:
            yield tail
        elif not yielded and not saw_result:
            # Stream exhausted with no ResultMessage and no text. A real
            # usage-limit stop ALWAYS arrives as a ResultMessage or a lone
            # AssistantMessage carrying the limit text (both handled
            # above) — never as pure silence. So this is a dropped
            # transport: the SDK's CLI subprocess has crashed or exited,
            # and every turn after it drains as "0 stale messages" and
            # comes back empty. Rebuild the session so the line self-heals
            # instead of parroting "usage limit" until someone restarts
            # backtalk. (This exact failure sent Master Pink chasing a
            # phantom usage limit on 2026-09-05.)
            log("[brain] empty turn, no result message — SDK session "
                "dropped; rebuilding")
            self._last_turn_failed = True
            try:
                await self._rebuild()
            except Exception as e:
                log(f"[brain] rebuild after dropped session failed: "
                    f"{str(e)[:160]}")
                yield ("My session to the model dropped and I couldn't "
                       "reconnect. Check this window.")
                return
            yield ("My session to the model dropped just then — I've "
                   "reconnected. Ask me that again.")


if __name__ == "__main__":
    import time

    async def demo():
        b = WarmBrain()
        await b.start()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            t0 = time.time()
            async for s in b.ask_stream(prompt):
                print(f"  ({time.time()-t0:4.1f}s) {s}", flush=True)
        await b.stop()

    asyncio.run(demo())

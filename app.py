"""Streamlit application layer for the permission-aware knowledge system.

Thin client over the Phase 5 API. This process holds no retrieval, ranking or
permission logic: it selects a prototype identity and renders what the server
returns. The UI is not the security boundary.
"""
from __future__ import annotations

import json
import os

import streamlit as st
import streamlit.components.v1 as components

import api_client

PAGE_TITLE = "Enterprise Knowledge AI"
PAGE_ICON = "\N{LOCK}"
MAX_UI_TURNS = 6
MAX_QUESTION_CHARS = 1000

SELLER_LABELS = {
    "S001": "Aurelia Home Decor",
    "S002": "BluePeak Electronics",
    "S003": "GreenCart Organics",
    "S004": "StrideOne Footwear",
}
SELLER_PROMPTS = {
    "S001": "What caused the settlement shortfall for Aurelia Home Decor?",
    "S002": "Summarise the inventory sync drift for BluePeak Electronics.",
    "S003": "What is the status of the GreenCart Organics tier upgrade?",
    "S004": "Why did returns spike for StrideOne Footwear?",
}
ORG_PROMPTS = [
    "What does the payment settlement policy say about adjustments?",
    "What are the steps in the payment recovery runbook?",
]

AVATARS = {"user": "\N{BUST IN SILHOUETTE}", "assistant": "\N{ROBOT FACE}"}


def _safe(text: str | None) -> str:
    return text or ""


def _copy_button(text: str, key: str) -> None:
    if not text:
        return
    payload = json.dumps(text).replace("</", "<\\/")
    html = f"""
    <style>
      #{key}{{
        display:inline-flex; align-items:center; gap:.35rem;
        font: 600 12px/1 'Segoe UI', sans-serif;
        color:var(--muted-fg, #64748b); background:transparent; cursor:pointer;
        border:1px solid var(--border, #e2e8f0); border-radius:8px;
        padding:5px 11px; transition:all .12s ease;
      }}
      #{key}:hover{{ color:var(--primary, #0f172a); border-color:var(--primary, #cbd5e1); }}
      #{key}.done{{ color:#16a34a; border-color:#bbf7d0; background:#f0fdf4; }}
      #{key} svg{{ flex:0 0 auto; }}
    </style>
    <button id="{key}" onclick="copyText()">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      <span>Copy</span>
    </button>
    <script>
      const TXT = {payload};
      const BTN = document.getElementById('{key}');
      function done(){{
        BTN.classList.add('done');
        BTN.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {{
          BTN.classList.remove('done');
          BTN.querySelector('span').textContent = 'Copy';
        }}, 1600);
      }}
      function legacy(){{
        const ta = document.createElement('textarea');
        ta.value = TXT;
        ta.style.cssText = 'position:fixed;opacity:0';
        document.body.appendChild(ta);
        ta.focus(); ta.select();
        try {{ document.execCommand('copy'); done(); }} catch (e) {{}}
        document.body.removeChild(ta);
      }}
      function copyText(){{
        if (navigator.clipboard && window.isSecureContext) {{
          navigator.clipboard.writeText(TXT).then(done, legacy);
        }} else {{ legacy(); }}
      }}
    </script>
    """
    components.html(html, height=38, scrolling=False)


THEME_VARS = {
    "light": """
    --bg: oklch(0.96 0.014 42);
    --fg: oklch(0.25 0.014 42);
    --card: oklch(0.98 0.014 42);
    --primary: oklch(0.55 0.16 42);
    --primary-fg: oklch(0.98 0.014 42);
    --secondary: oklch(0.92 0.014 42);
    --muted-fg: oklch(0.50 0.014 42);
    --border: oklch(0.90 0.014 42);
    --destructive: oklch(0.55 0.16 22);
    --shadow: 0 1px 2px oklch(0.25 0.014 42 / 0.05);
  """,
    "dark": """
    --bg: oklch(0.20 0.014 42);
    --fg: oklch(0.96 0.014 42);
    --card: oklch(0.25 0.014 42);
    --primary: oklch(0.65 0.16 42);
    --primary-fg: oklch(0.20 0.014 42);
    --secondary: oklch(0.30 0.014 42);
    --muted-fg: oklch(0.70 0.014 42);
    --border: oklch(0.35 0.014 42);
    --destructive: oklch(0.60 0.16 22);
    --shadow: 0 1px 2px oklch(0.15 0.014 42 / 0.35);
  """,
}

BASE_CSS = """
  .stApp{ background: var(--bg); color: var(--fg); }
  #MainMenu, footer{ visibility: hidden; }
  header[data-testid="stHeader"]{
    background: transparent !important;
    visibility: visible;
  }
  [data-testid="stSidebarCollapsedControl"]{
    visibility: visible !important;
  }
  .block-container{ padding-top: 2.2rem; max-width: 52rem; }
  .stApp, .stMarkdown, p, li, span, label, h1, h2, h3{ color: var(--fg); }
  code{
    background: var(--secondary) !important; color: var(--fg) !important;
    padding:.05rem .3rem; border-radius:.25rem; font-size:.85em;
  }
  hr{ border-color: var(--border) !important; }

  [data-testid="stSidebar"]{
    background: var(--card); border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .block-container{ padding-top: 1.4rem; }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{
    color: var(--muted-fg) !important; font-size:.78rem;
  }

  .stButton > button{
    background: var(--card); color: var(--fg);
    border:1px solid var(--border); border-radius: var(--radius);
    font-weight:550; transition: background .12s ease, border-color .12s ease;
    box-shadow: var(--shadow);
  }
  .stButton > button:hover{
    border-color: var(--primary); color: var(--primary);
    background: color-mix(in oklch, var(--primary) 7%, var(--card));
  }
  .stButton > button[kind="primary"]{
    background: var(--primary); color: var(--primary-fg);
    border-color: var(--primary);
  }

  [data-testid="stExpander"]{
    background: var(--card); border:1px solid var(--border);
    border-radius: var(--radius);
  }
  [data-testid="stExpander"] summary{ color: var(--fg); font-size:.87rem; }

  [data-testid="stChatInput"]{
    background: var(--card); border:1px solid var(--border);
    border-radius: var(--radius);
  }
  [data-testid="stChatInput"] textarea{
    color: var(--fg) !important; font-size:.95rem;
  }
  [data-testid="stChatInput"] textarea::placeholder{ color: var(--muted-fg); }

  .brand{ display:flex; align-items:center; gap:.55rem; margin-bottom:.15rem; }
  .brand-mark{
    width:1.75rem; height:1.75rem; border-radius:.5rem;
    background: var(--primary); color: var(--primary-fg);
    display:flex; align-items:center; justify-content:center;
    font-size:.95rem; flex:0 0 auto;
  }
  .brand-name{ font-weight:700; font-size:1rem; letter-spacing:-.01em; }
  .brand-sub{ color:var(--muted-fg); font-size:.78rem; margin:0 0 .2rem 2.3rem; }

  .hero{ text-align:center; padding: 1.8rem 0 1.5rem; }
  .hero h1{
    font-size: 2.3rem; font-weight:700; letter-spacing:-.025em;
    margin:0 0 .5rem; color: var(--fg);
  }
  .hero p{ color: var(--muted-fg); font-size:1rem; margin:0 auto; max-width:34rem; }

  .sec{
    font-size:.68rem; font-weight:700; letter-spacing:.1em;
    color:var(--muted-fg); text-transform:uppercase; margin:.1rem 0 .55rem;
  }
  .ucard{
    border:1px solid var(--border); border-radius:var(--radius);
    background:var(--card); padding:.85rem .95rem; box-shadow: var(--shadow);
  }
  .ucard .nm{ font-weight:650; font-size:1rem; }
  .ucard .rl{ color:var(--primary); font-size:.83rem; font-weight:600; margin-top:.1rem; }
  .ucard .dp{ color:var(--muted-fg); font-size:.79rem; margin-top:.05rem; }
  .ucard .sc{ margin-top:.55rem; }

  .idbox{
    border:1px solid var(--border); border-radius:var(--radius);
    background: var(--bg); padding:.7rem .8rem;
  }
  .idbox .nm{ font-weight:650; font-size:.94rem; }
  .idbox .rl{ color:var(--primary); font-size:.8rem; font-weight:600; }
  .idbox .dp{ color:var(--muted-fg); font-size:.76rem; margin-top:.1rem; }

  [data-testid="stChatMessage"]{
    background: var(--card); border:1px solid var(--border);
    border-radius: var(--radius); padding:.9rem 1rem; margin-bottom:.7rem;
    box-shadow: var(--shadow);
  }
  [data-testid="stChatMessage"] p{ line-height:1.62; }
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3{
    font-size:1.02rem; margin:.7rem 0 .3rem; font-weight:650;
  }
  [data-testid="stChatMessage"] ul{ margin:.3rem 0; padding-left:1.2rem; }
  [data-testid="stChatMessage"] li{ margin:.15rem 0; }
  [data-testid="stChatMessage"] table{
    border-collapse:collapse; width:100%; font-size:.85rem; margin:.5rem 0;
  }
  [data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td{
    border:1px solid var(--border); padding:.3rem .55rem; text-align:left;
  }
  [data-testid="stChatMessage"] th{ background: var(--secondary); }

  .refusal{
    border:1px solid color-mix(in oklch, var(--destructive) 34%, transparent);
    background: color-mix(in oklch, var(--destructive) 9%, transparent);
    border-radius: var(--radius); padding:.85rem 1rem;
  }
  .refusal b{ color: var(--destructive); display:block; margin-bottom:.2rem; }

  .src{
    border:1px solid var(--border); border-radius:var(--radius);
    background:var(--bg); padding:.5rem .7rem; margin-bottom:.4rem;
  }
  .src .t{ font-weight:620; font-size:.87rem; }
  .src .m{ color:var(--muted-fg); font-size:.76rem; margin-top:.1rem; }
  .stats{ color:var(--muted-fg); font-size:.74rem; margin-top:.55rem; }

  .guard{
    border:1px solid var(--border); border-left:3px solid var(--primary);
    border-radius:.4rem; padding:.5rem .7rem; font-size:.76rem;
    color:var(--muted-fg); background:var(--bg);
  }
  .dot-row{
    display:flex; align-items:center; gap:.4rem;
    font-size:.78rem; color:var(--muted-fg);
  }
  .dot{ width:.5rem; height:.5rem; border-radius:999px; display:inline-block; }
  .dot-ok{ background:#22c55e; }
  .dot-bad{ background:#ef4444; }

  [data-testid="stStatusWidget"]{ display:none; }
"""


def _css(theme: str) -> str:
    return (f"<style>:root{{{THEME_VARS.get(theme, THEME_VARS['light'])}"
            f"--radius:0.5rem;}}{BASE_CSS}</style>")


def _base_url() -> str:
    try:
        return st.secrets.get("API_BASE_URL", api_client.DEFAULT_BASE_URL)
    except Exception:
        return os.getenv("API_BASE_URL", api_client.DEFAULT_BASE_URL)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("theme", "light")
    st.session_state.setdefault("seller_choice", 0)
    st.session_state.setdefault("history_turns", MAX_UI_TURNS)


def _toggle_theme() -> None:
    st.session_state.theme = (
        "dark" if st.session_state.theme == "light" else "light")


def _theme_button(key: str, use_container_width: bool = True) -> None:
    dark = st.session_state.theme == "dark"
    label = "\N{BLACK SUN WITH RAYS} Light" if dark else "\N{LAST QUARTER MOON} Dark"
    if st.button(label, key=key, use_container_width=use_container_width):
        _toggle_theme()
        st.rerun()


def _reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending = None


def _sign_out() -> None:
    st.session_state.user = None
    st.session_state.seller_choice = 0
    st.session_state.pop("seller_select", None)
    _reset_conversation()


def _chips(scope: list[str]) -> str:
    if not scope:
        return '<span class="chip-muted">No seller portfolio</span>'
    return "".join(f'<span class="chip">&check; {s}</span>' for s in scope)


def _brand(sub: str) -> str:
    return (f'<div class="brand"><div class="brand-mark">&#128274;</div>'
            f'<div class="brand-name">{PAGE_TITLE}</div></div>'
            f'<div class="brand-sub">{sub}</div>')


def _seller_options(scope: list[str]) -> list[str]:
    return ["All authorized sellers"] + [
        f"{sid} - {name}" for sid, name in SELLER_LABELS.items()
        if sid in scope]


def _seller_filter(options: list[str]) -> list[str] | None:
    index = min(st.session_state.seller_choice, len(options) - 1)
    if index <= 0:
        return None
    return [options[index].split(" - ")[0]]


def _load_users(base_url: str) -> list[dict]:
    try:
        users = api_client.list_users(base_url)
    except api_client.ApiError as exc:
        st.error(str(exc))
        st.caption("Start the service with `python answering/serve.py`")
        st.stop()
    if not users:
        st.error("No selectable identities were returned.")
        st.stop()
    return users


def _login_sidebar(base_url: str) -> None:
    with st.sidebar:
        st.markdown(_brand("Permission-aware assistant"),
                    unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sec">How it works</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="ucard"><div class="rl">1 &middot; Pick an identity</div>'
            '<div class="dp">Each one maps to a role, department and seller '
            'portfolio.</div></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ucard"><div class="rl">2 &middot; Ask a question</div>'
            '<div class="dp">Retrieval runs only over documents that identity '
            'is authorized to read.</div></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ucard"><div class="rl">3 &middot; Get a cited answer</div>'
            '<div class="dp">Every claim is grounded in verified sources with '
            'page-level citations.</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="sec">Service status</div>',
                    unsafe_allow_html=True)
        if api_client.health(base_url):
            st.markdown('<div class="dot-row"><span class="dot dot-ok"></span>'
                        'API online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dot-row"><span class="dot dot-bad"></span>'
                        'API unreachable</div>', unsafe_allow_html=True)
            st.caption("Start it with `python answering/serve.py`")

        st.divider()
        _theme_button("theme_login")
        st.markdown(
            '<div class="guard">&#128274; In production this identity '
            'selector would be an authenticated SSO session. The server '
            'derives permissions from the identity and never trusts a '
            'client-supplied scope.</div>', unsafe_allow_html=True)


def _login_screen(base_url: str) -> None:
    st.markdown(
        '<div class="hero"><h1>Enterprise Knowledge AI</h1>'
        '<p>A permission-aware assistant. Every answer is drawn only from the '
        'documents your identity is authorized to read.</p></div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sec">Choose an identity to continue</div>',
                unsafe_allow_html=True)

    users = _load_users(base_url)
    for row_start in range(0, len(users), 2):
        columns = st.columns(2, gap="medium")
        for column, user in zip(columns, users[row_start:row_start + 2]):
            with column:
                st.markdown(
                    f'<div class="ucard"><div class="nm">{user["name"]}</div>'
                    f'<div class="rl">{user["role"]}</div>'
                    f'<div class="dp">{user["department"]} &middot; '
                    f'{user["user_id"]}</div>'
                    f'<div class="sc">{_chips(user.get("seller_scope") or [])}'
                    f'</div></div>', unsafe_allow_html=True)
                if st.button(f"Continue as {user['name'].split()[0]}",
                             key=f"login_{user['user_id']}",
                             use_container_width=True):
                    st.session_state.user = user
                    st.session_state.seller_choice = 0
                    st.session_state.pop("seller_select", None)
                    _reset_conversation()
                    st.rerun()
        st.write("")


def _sidebar(user: dict, base_url: str) -> None:
    with st.sidebar:
        st.markdown(_brand("Permission-aware assistant"),
                    unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sec">Signed in</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="idbox"><div class="nm">{user["name"]}</div>'
            f'<div class="rl">{user["role"]}</div>'
            f'<div class="dp">{user["department"]} &middot; '
            f'{user["user_id"]}</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="sec">Authorized sellers</div>',
                    unsafe_allow_html=True)
        scope = user.get("seller_scope") or []
        st.markdown(_chips(scope), unsafe_allow_html=True)
        st.caption("Organization-wide policies and runbooks are governed by "
                   "classification, not seller scope.")

        st.write("")
        st.markdown('<div class="sec">Focus (optional)</div>',
                    unsafe_allow_html=True)
        options = _seller_options(scope)
        index = min(st.session_state.seller_choice, len(options) - 1)
        st.selectbox("Seller filter", options, index=index,
                     key="seller_select", label_visibility="collapsed")
        st.session_state.seller_choice = options.index(
            st.session_state.seller_select)

        st.write("")
        st.markdown('<div class="sec">Conversation</div>',
                    unsafe_allow_html=True)
        st.slider("History turns sent to the model", 2, 10,
                  value=st.session_state.history_turns, key="turn_slider")
        st.session_state.history_turns = st.session_state.turn_slider

        st.write("")
        if st.button("New conversation", use_container_width=True,
                     type="primary"):
            _reset_conversation()
            st.rerun()
        column_switch, column_theme = st.columns(2)
        with column_switch:
            if st.button("Switch user", use_container_width=True):
                _sign_out()
                st.rerun()
        with column_theme:
            _theme_button("theme_sidebar")

        st.divider()
        if api_client.health(base_url):
            st.markdown('<div class="dot-row"><span class="dot dot-ok"></span>'
                        'Service online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dot-row"><span class="dot dot-bad"></span>'
                        'Service unreachable</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="guard">&#128274; Authorization is enforced '
            'server-side by the permission engine before any document reaches '
            'the model.</div>', unsafe_allow_html=True)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"\N{PAGE FACING UP} Sources ({len(sources)})"):
        for source in sources:
            meta = [f"<code>{source['document_id']}</code>"]
            if source.get("seller_id"):
                meta.append(f"Seller {source['seller_id']}")
            if source.get("pages"):
                pages = ", ".join(str(p) for p in source["pages"])
                label = "Page" if len(source["pages"]) == 1 else "Pages"
                meta.append(f"{label} {pages}")
            if source.get("relevance") is not None:
                meta.append(f"relevance {source['relevance']:.2f}")
            st.markdown(
                f'<div class="src"><div class="t">&#128196; '
                f'{_safe(source.get("label", ""))}</div>'
                f'<div class="m">{" &middot; ".join(meta)}</div></div>',
                unsafe_allow_html=True)


def _render_stats(stats: dict, effective: list[str]) -> None:
    if not stats:
        return
    scope = ", ".join(effective) if effective else "none"
    st.markdown(
        f'<div class="stats">scope {scope} &middot; '
        f'candidates {stats.get("candidates", 0)} &middot; '
        f'authorized {stats.get("authorized", 0)} &middot; '
        f'used {stats.get("selected", 0)} &middot; '
        f'withheld {stats.get("withheld", 0)} &middot; '
        f'{stats.get("latency_ms", 0)} ms</div>', unsafe_allow_html=True)


def _render_refusal() -> None:
    st.markdown(
        '<div class="refusal"><b>&#128274; No authorized information found</b>'
        "I couldn't find sufficient authorized information to answer this "
        "request.</div>", unsafe_allow_html=True)


def _example_prompts(scope: list[str]) -> list[str]:
    return ([SELLER_PROMPTS[s] for s in scope if s in SELLER_PROMPTS]
            + ORG_PROMPTS)[:4]


def _history_payload() -> list[dict]:
    return [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if not m.get("failed")][-st.session_state.history_turns:]


def _strip_inline_sources(text: str) -> str:
    marker = "\nSources:\n"
    index = text.find(marker)
    while index != -1:
        after = text[index + len(marker):]
        if not after.strip() or after.lstrip().startswith("- "):
            return text[:index].rstrip()
        index = text.find(marker, index + 1)
    if text.rstrip().endswith("Sources:"):
        return text.rstrip()[:-len("Sources:")].rstrip()
    return text


def _deltas(events, captured: dict):
    for event in events:
        kind = event.get("type")
        if kind == "delta":
            yield event.get("text", "")
        elif kind == "meta":
            captured["meta"] = event
        elif kind == "done":
            captured["done"] = event
        elif kind == "error":
            captured["error"] = event.get("message", "Generation failed.")
            return


def _stream_answer(user: dict, base_url: str, question: str,
                   seller_filter: list[str] | None) -> dict:
    events = api_client.ask_stream(user["user_id"], question,
                                   conversation=_history_payload(),
                                   seller_filter=seller_filter,
                                   base_url=base_url)
    captured: dict = {}
    meta = None
    for event in events:
        if event.get("type") == "meta":
            meta = event
            captured["meta"] = event
            break

    if meta is not None and not meta.get("answered", True):
        for _ in events:
            pass
        return {"answer": "", "meta": meta, "refused": True}

    answer = st.write_stream(_deltas(events, captured))
    answer = _strip_inline_sources(_safe(answer))
    if captured.get("error"):
        st.error(_safe(captured["error"]))
        return {"answer": answer, "meta": captured.get("meta", {}),
                "failed": True}
    if not answer.strip():
        st.error("The assistant returned an empty answer. Please try again.")
        return {"answer": "", "meta": captured.get("meta", {}), "failed": True}
    return {"answer": answer,
            "meta": {**(captured.get("meta") or {}),
                     **(captured.get("done") or {})}}


def _chat_screen(user: dict, base_url: str) -> None:
    _sidebar(user, base_url)

    scope = user.get("seller_scope") or []
    options = _seller_options(scope)
    seller_filter = _seller_filter(options)

    if st.session_state.messages:
        for index, message in enumerate(st.session_state.messages):
            message["id"] = message.get("id") or f"m{index}"
            with st.chat_message(message["role"],
                                 avatar=AVATARS.get(message["role"])):
                if message.get("refused"):
                    _render_refusal()
                elif message.get("failed"):
                    if message.get("content"):
                        st.markdown(message["content"])
                    st.warning("The previous answer failed to generate and is "
                               "not part of the conversation memory.")
                else:
                    st.markdown(message["content"])
                if message["role"] == "assistant":
                    if (not message.get("failed")
                            and not message.get("refused")
                            and message.get("content")):
                        _copy_button(message["content"],
                                     key=f"copy-{message['id']}")
                    _render_sources(message.get("sources", []))
                    _render_stats(message.get("stats", {}),
                                  message.get("effective_sellers", []))
    else:
        st.markdown(
            f'<div class="hero"><h1>Hello, {user["name"].split()[0]}</h1>'
            f'<p>Ask about the sellers, tickets, incidents, runbooks and '
            f'policies you are authorized to read.</p></div>',
            unsafe_allow_html=True)
        st.markdown('<div class="sec">Suggested questions</div>',
                    unsafe_allow_html=True)
        for index, prompt in enumerate(_example_prompts(scope)):
            if st.button(prompt, key=f"ex_{index}", use_container_width=True):
                st.session_state.pending = prompt
                st.rerun()

    typed = st.chat_input("Ask about sellers, tickets, incidents or policies",
                          max_chars=MAX_QUESTION_CHARS)
    question = typed or st.session_state.pending
    st.session_state.pending = None
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS.get("user")):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS.get("assistant")):
        stage = st.status("Working on your request...", state="running")
        try:
            result = _stream_answer(user, base_url, question, seller_filter)
        except api_client.ApiError as exc:
            stage.update(label="Service error", state="error")
            st.error(str(exc))
            return
        except Exception:
            stage.update(label="Something went wrong", state="error")
            st.error("Something went wrong while answering. Please try again.")
            return

        meta = result.get("meta") or {}
        sources = meta.get("sources", []) or []
        stats = meta.get("stats", {}) or {}
        effective = meta.get("effective_sellers", []) or []

        if result.get("failed"):
            stage.update(label="Generation failed", state="error")
            st.session_state.messages.append({
                "role": "assistant", "content": result.get("answer", ""),
                "failed": True})
            return

        if not result.get("refused"):
            stage.update(label="Answer ready", state="complete")
            _copy_button(result["answer"], key="copy-live")
            _render_sources(sources)
            _render_stats(stats, effective)
        else:
            stage.update(label="Access filtered", state="complete")
            _render_refusal()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "refused": bool(result.get("refused")),
        "sources": sources,
        "stats": stats,
        "effective_sellers": effective,
    })
    st.rerun()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON,
                       layout="centered",
                       initial_sidebar_state="expanded")
    _init_state()
    st.markdown(_css(st.session_state.theme), unsafe_allow_html=True)
    base_url = _base_url()

    if st.session_state.user is None:
        _login_sidebar(base_url)
        _login_screen(base_url)
    else:
        _chat_screen(st.session_state.user, base_url)


if __name__ == "__main__":
    main()

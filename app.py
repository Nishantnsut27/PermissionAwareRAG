"""Streamlit application layer for the permission-aware knowledge system.

Thin client over the Phase 5 API. This process holds no retrieval, ranking or
permission logic: it selects a prototype identity and renders what the server
returns. The UI is not the security boundary.
"""
from __future__ import annotations

import html
import hmac
import json
import math
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import api_client
from ui_theme import brand_mark, css as _css, icon

PAGE_TITLE = "Enterprise Knowledge AI"
PAGE_ICON = str(Path(__file__).parent / "images" / "knowledge-mark.png")
MAX_UI_TURNS = 6
MAX_QUESTION_CHARS = 1000
DEMO_LOGIN_VERSION = 1

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
DEPARTMENT_PROMPTS = {
    "Business": [
        "What does the payment settlement policy say about adjustments?",
        "What are the current priorities for my seller accounts?",
    ],
    "Support": [
        "What steps should Support follow when escalating a seller issue?",
        "Which checks should I complete before closing a support ticket?",
    ],
    "Engineering": [
        "What are the required steps for incident recovery and verification?",
        "Which engineering changes are awaiting operational validation?",
    ],
    "Operations": [
        "What are the steps in the payment recovery runbook?",
        "What checks are required before releasing a corrected payout?",
    ],
    "IT": [
        "What is the process for reviewing access and security incidents?",
        "Which controls are required before restoring a service?",
    ],
}

AVATARS = {
    "user": '''<svg viewBox="0 0 64 64" role="img" aria-label="You">
      <rect x="2" y="2" width="60" height="60" rx="18" fill="#EEF0FF"/>
      <circle cx="32" cy="25" r="11" fill="#6257D9"/>
      <path d="M13 55c1-11 8-18 19-18s18 7 19 18" fill="#6257D9"/>
      <path d="M22 38c2 5 6 8 10 8s8-3 10-8" fill="#5147C4"/>
    </svg>''',
    "assistant": '''<svg viewBox="0 0 64 64" role="img" aria-label="Assistant">
      <rect x="2" y="2" width="60" height="60" rx="18" fill="#E6F7F5"/>
      <path d="M32 11v7" stroke="#138A83" stroke-width="4" stroke-linecap="round"/>
      <circle cx="32" cy="10" r="3" fill="#20A69D"/>
      <rect x="13" y="19" width="38" height="31" rx="11" fill="#168F88"/>
      <rect x="19" y="26" width="26" height="14" rx="6" fill="#F7FFFE"/>
      <circle cx="26" cy="33" r="2.5" fill="#176C68"/>
      <circle cx="38" cy="33" r="2.5" fill="#176C68"/>
      <path d="M26 45h12" stroke="#B9F0E9" stroke-width="3" stroke-linecap="round"/>
    </svg>''',
}


def _safe(text: str | None) -> str:
    return text or ""


def _copy_button(text: str, key: str) -> None:
    if not text:
        return
    payload = json.dumps(text).replace("</", "<\\/")
    html = f"""
    <style>
      html, body {{ width:max-content; height:36px; margin:0; padding:0;
        overflow:hidden; }}
      body {{ display:flex; align-items:center; }}
    </style>
    <style>
      #{key}{{
        display:inline-flex; align-items:center; gap:.35rem;
        font: 600 12px/1 'Segoe UI', sans-serif;
        color:var(--muted-fg, #64748b); background:transparent; cursor:pointer;
        border:1px solid var(--border, #e2e8f0); border-radius:8px;
        box-sizing:border-box; height:30px; padding:5px 11px;
        transition:all .12s ease;
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
    st.iframe(html, height=40, width="content")


def _base_url() -> str:
    try:
        return st.secrets.get("API_BASE_URL", api_client.DEFAULT_BASE_URL)
    except Exception:
        return os.getenv("API_BASE_URL", api_client.DEFAULT_BASE_URL)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("active_question", None)
    st.session_state.setdefault("active_history", None)
    st.session_state.setdefault("theme", "light")
    st.session_state.setdefault("seller_choice", 0)
    st.session_state.setdefault("history_turns", MAX_UI_TURNS)
    st.session_state.setdefault("view", "chat")
    if st.session_state.get("demo_login_version") != DEMO_LOGIN_VERSION:
        # Require existing demo sessions to pass through the new login screen.
        st.session_state.user = None
        st.session_state.messages = []
        st.session_state.demo_login_version = DEMO_LOGIN_VERSION


def _theme_toggle(key: str) -> None:
    """Light | switch | Dark sliding toggle (native st.toggle, no JS bridge).

    `st.toggle` value True means dark. Only one instance is ever visible
    (login sidebar vs app sidebar), so per-key state stays in sync with
    `st.session_state.theme`.
    """
    dark = st.session_state.theme == "dark"
    col_light, col_switch, col_dark = st.columns(
        [1.2, 0.75, 1.2], gap="small", vertical_alignment="center")
    with col_light:
        st.markdown(
            f'<div class="ld-label{" on" if not dark else ""}">Light</div>',
            unsafe_allow_html=True)
    with col_switch:
        st.toggle("Theme", value=dark, key=key,
                  label_visibility="collapsed")
        want_dark = bool(st.session_state.get(key, dark))
        if want_dark != dark:
            st.session_state.theme = "dark" if want_dark else "light"
            st.rerun()
    with col_dark:
        st.markdown(
            f'<div class="ld-label right{" on" if dark else ""}">Dark</div>',
            unsafe_allow_html=True)


def _reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.active_question = None
    st.session_state.active_history = None


def _sign_out() -> None:
    st.session_state.user = None
    st.session_state.demo_login_version = DEMO_LOGIN_VERSION
    st.session_state.view = "chat"
    st.session_state.seller_choice = 0
    st.session_state.pop("seller_select", None)
    _reset_conversation()


def _chips(scope: list[str]) -> str:
    if not scope:
        return '<span class="chip-muted">No seller portfolio</span>'
    return "".join(f'<span class="chip">&check; {s}</span>' for s in scope)


def _id_chips(items) -> str:
    items = [str(i) for i in (items or [])]
    if not items:
        return '<div class="idnone">(none)</div>'
    return ('<div class="idlist">'
            + "".join(f"<code>{html.escape(i)}</code>" for i in items)
            + "</div>")


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
        users = api_client.list_demo_users(base_url)
    except api_client.ApiError as exc:
        st.error(str(exc))
        st.caption("Start the service with `python answering/serve.py`")
        st.caption("For a local synthetic-data demo, set `API_DEMO_MODE=true`. "
                   "The selector is not authentication.")
        st.stop()
    if not users:
        st.error("No selectable identities were returned.")
        st.stop()
    return users


def _demo_password(user: dict) -> str:
    """Public password pattern for the explicitly loopback-only demo."""
    first_name = str(user.get("name") or "").strip().split()
    return f"{first_name[0].casefold()}@login" if first_name else ""


def _valid_demo_password(user: dict, password: str) -> bool:
    expected = _demo_password(user)
    supplied = str(password or "").strip().casefold()
    return bool(expected) and hmac.compare_digest(expected, supplied)


def _login_sidebar(base_url: str) -> None:
    with st.sidebar:
        st.markdown(_brand("Permission-aware assistant"),
                    unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sec">How it works</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="ucard"><div class="rl">1. Choose your account</div>'
            '<div class="dp">Select a sample account and enter its demo password.</div></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ucard"><div class="rl">2. Ask a question</div>'
            '<div class="dp">Ask about sellers, policies, incidents, or operations.</div></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ucard"><div class="rl">3. Review the answer</div>'
            '<div class="dp">Open the cited sources to check supporting pages.</div></div>',
            unsafe_allow_html=True)

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
        _theme_toggle("theme_toggle_login")
        st.markdown(
            '<div class="guard">Local demo accounts use sample identities '
            'and published passwords.</div>',
            unsafe_allow_html=True)


def _login_screen(base_url: str) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Permission-aware knowledge assistant</div>'
        '<h1>Knowledge answers, scoped to your access.</h1>'
        '<p>Choose an account to explore the knowledge workspace.</p></div>',
        unsafe_allow_html=True)

    users = _load_users(base_url)
    user_by_id = {user["user_id"]: user for user in users}
    _, center_column, _ = st.columns([1, 1.5, 1], gap="large")
    with center_column:
        with st.container(border=True):
            st.markdown(
                '<div class="login-kicker">Local demo access</div>'
                '<div class="login-title">Welcome back</div>'
                '<div class="login-copy">Select your sample account and enter '
                'the demo password.</div>', unsafe_allow_html=True)
            with st.form("demo_login_form", clear_on_submit=False):
                selected_id = st.selectbox(
                    "Demo account", list(user_by_id),
                    format_func=lambda user_id: (
                        f"{user_by_id[user_id]['name']} · "
                        f"{user_by_id[user_id]['role']}"))
                password = st.text_input("Password", type="password",
                                         placeholder="firstname@login")
                submitted = st.form_submit_button(
                    "Continue", type="primary", use_container_width=True)
            if submitted:
                selected_user = user_by_id[selected_id]
                if _valid_demo_password(selected_user, password):
                    st.session_state.user = selected_user
                    st.session_state.seller_choice = 0
                    st.session_state.pop("seller_select", None)
                    _reset_conversation()
                    st.rerun()
                st.error("That demo password does not match the selected account.")
            with st.expander("Show demo passwords"):
                for user in users:
                    st.text(f"{user['name']}  ·  {_demo_password(user)}")


def _sidebar(user: dict, base_url: str) -> None:
    with st.sidebar:
        st.markdown(_brand("Permission-aware assistant"),
                    unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sec">Demo identity</div>', unsafe_allow_html=True)
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
        st.caption("Organization-wide documents also follow role, clearance "
                   "and department rules.")

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
        st.slider("Conversation memory", 2, MAX_UI_TURNS,
                  value=min(st.session_state.history_turns, MAX_UI_TURNS), key="turn_slider")
        st.session_state.history_turns = st.session_state.turn_slider

        st.write("")
        if st.button("New conversation", use_container_width=True,
                     type="primary"):
            _reset_conversation()
            st.session_state.view = "chat"
            st.rerun()
        if st.button("Switch user", key="switch_user",
                         use_container_width=True):
                _sign_out()
                st.rerun()
        _theme_toggle("theme_toggle_sidebar")

        st.divider()
        _evaluation_sidebar(base_url)

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
            if not isinstance(source, dict):
                continue
            # These values come from document metadata. Render them as plain
            # text so hostile IDs, labels, and page strings cannot become HTML
            # or Markdown links.
            label = html.escape(str(source.get("label") or "Untitled source"))
            details = [f"Document ID: {source.get('document_id', 'unknown')}"]
            if source.get("seller_id"):
                details.append(f"Seller: {source['seller_id']}")
            pages = source.get("pages")
            if isinstance(pages, (list, tuple)) and pages:
                page_label = "Page" if len(pages) == 1 else "Pages"
                details.append(f"{page_label}: {', '.join(map(str, pages))}")
            relevance = source.get("relevance")
            if isinstance(relevance, (int, float)) and math.isfinite(relevance):
                details.append(f"Relevance: {relevance:.2f}")
            details_text = html.escape(" · ".join(map(str, details)))
            st.markdown(
                f'<div class="source-item"><div class="source-title">'
                f'<span aria-hidden="true">📄</span> {label}</div>'
                f'<div class="source-meta">{details_text}</div></div>',
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


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.{digits}f}%"


def _num(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _metric_rows(pairs: list[tuple[str, str]]) -> str:
    return "".join(f'<div class="evrow"><span>{label}</span><b>{value}</b>'
                   f'</div>' for label, value in pairs)


def _eval_summary(base_url: str, report: str | None = None) -> dict | None:
    """Read the latest saved evaluation. Never triggers a run."""
    try:
        return api_client.evaluation_summary(base_url, report=report)
    except api_client.ApiError:
        return None


def _evaluation_sidebar(base_url: str) -> None:
    st.markdown('<div class="sec">Evaluation</div>', unsafe_allow_html=True)
    payload = _eval_summary(base_url)
    if payload is None:
        st.markdown('<div class="evempty">Evaluation data is unavailable '
                    'while the service is unreachable.</div>',
                    unsafe_allow_html=True)
        return

    if not payload.get("has_results"):
        st.markdown(
            '<div class="evempty">No evaluation results available. Run the '
            'evaluation to generate results.</div>', unsafe_allow_html=True)
    else:
        summary = payload.get("summary") or {}
        total = summary.get("total_cases", 0)
        errors = summary.get("errors", 0)
        dataset_total = (payload.get("dataset") or {}).get("total_cases", 0)
        selected = payload.get("selected_cases") or total
        st.markdown(
            f'<div class="evtally">'
            f'<div class="evpill"><span class="n">{total}</span>'
            f'<span class="l">Cases</span></div>'
            f'<div class="evpill ok"><span class="n">'
            f'{summary.get("passed", 0)}</span><span class="l">Passed</span>'
            f'</div>'
            f'<div class="evpill bad"><span class="n">'
            f'{summary.get("failed", 0)}</span><span class="l">Failed</span>'
            f'</div>'
            + (f'<div class="evpill warn"><span class="n">{errors}</span>'
               f'<span class="l">Errors</span></div>' if errors else "")
            + '</div>', unsafe_allow_html=True)

        retrieval = summary.get("retrieval") or {}
        generation = summary.get("generation") or {}
        security = summary.get("security") or {}
        st.markdown(
            '<div class="evhead">Retrieval</div>'
            + _metric_rows([
                ("Recall@K", _pct(retrieval.get("recall_at_k"))),
                ("Precision@K", _pct(retrieval.get("precision_at_k"))),
                ("MRR", _num(retrieval.get("mrr"))),
            ])
            + '<div class="evhead">Generation</div>'
            + _metric_rows([
                ("Groundedness", _pct(generation.get("groundedness"))),
                ("Relevancy", _pct(generation.get("answer_relevancy"))),
                ("Correctness", _pct(generation.get("answer_correctness"))),
            ])
            + '<div class="evhead">Security</div>'
            + _metric_rows([
                ("Authorization", _pct(security.get("authorization_accuracy"))),
                ("Leakage", _pct(security.get("answer_leakage_rate"))),
                ("Injection blocked",
                 _pct(security.get("prompt_injection_blocked"))),
            ]), unsafe_allow_html=True)
        st.caption(f"Last evaluation: {payload.get('generated_at') or 'n/a'}")
        if dataset_total:
            st.caption(f"Coverage: {total} results recorded from {selected} selected "
                       f"of {dataset_total} golden cases.")
        if payload.get("partial") or errors:
            st.caption("Incomplete evidence: review coverage and errors before interpreting scores.")

    label = ("Back to chat" if st.session_state.view == "evaluation"
             else "View evaluation matrix")
    if st.button(label, key="eval_toggle", use_container_width=True):
        st.session_state.view = ("chat"
                                 if st.session_state.view == "evaluation"
                                 else "evaluation")
        st.rerun()


def _matrix_frame(matrix: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix)
    if frame.empty:
        return frame
    return frame.rename(columns={
        "eval_id": "ID",
        "user": "User",
        "category": "Category",
        "expected_behavior": "Expected",
        "actual_behavior": "Actual",
        "recall_at_k": "Recall@K",
        "precision_at_k": "Precision@K",
        "mrr": "MRR",
        "groundedness": "Groundedness",
        "relevancy": "Relevancy",
        "correctness": "Correctness",
        "security": "Security",
        "status": "Status",
    })


def _filter_matrix(frame: pd.DataFrame, outcome: str, category: str,
                   user: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if outcome == "Passed":
        frame = frame[frame["Status"] == "PASS"]
    elif outcome == "Failed":
        frame = frame[frame["Status"] == "FAIL"]
    elif outcome == "Errors":
        frame = frame[frame["Status"] == "ERROR"]
    elif outcome == "Security relevant":
        frame = frame[frame["Category"].isin(
            ["unauthorized_access", "prompt_injection", "mixed_access",
             "cross_seller"])]
    elif outcome == "Retrieval scored":
        frame = frame[frame["Recall@K"].notna()]
    elif outcome == "Generation scored":
        frame = frame[frame["Groundedness"].notna()]
    if category != "All categories":
        frame = frame[frame["Category"] == category]
    if user != "All users":
        frame = frame[frame["User"] == user]
    return frame


def _render_case_detail(results: list[dict], eval_id: str) -> None:
    case = next((r for r in results if r["eval_id"] == eval_id), None)
    if case is None:
        return
    st.markdown(f"**{case['eval_id']} &middot; {case['user']} &middot; "
                f"{case['category']}**", unsafe_allow_html=True)
    st.caption(case["question"])

    if case.get("error"):
        st.error(f"ERROR: {case['error']}")
    if case.get("failure_reasons"):
        for reason in case["failure_reasons"]:
            st.warning(reason)

    security = case.get("security") or {}
    retrieval_verdict = ("YES" if security.get("unauthorized_retrieval") else "NO") if security else "N/A"
    leakage_verdict = (("YES" if security.get("answer_leakage") else "NO")
                       if security and security.get("leakage_checked", True) else "N/A")
    columns = st.columns(3)
    columns[0].markdown(
        f'<div class="kpi"><div class="l">Status</div>'
        f'<div class="v">{case["status"]}</div>'
        f'<div class="s">expected {case["expected_behavior"]} / '
        f'got {case.get("actual_behavior") or "n/a"}</div></div>',
        unsafe_allow_html=True)
    columns[1].markdown(
        f'<div class="kpi"><div class="l">Unauthorized retrieval</div>'
        f'<div class="v">'
        f'{retrieval_verdict}</div>'
        f'<div class="s">documents outside the identity scope</div></div>',
        unsafe_allow_html=True)
    columns[2].markdown(
        f'<div class="kpi"><div class="l">Answer leakage</div>'
        f'<div class="v">'
        f'{leakage_verdict}</div>'
        f'<div class="s">restricted content in the reply</div></div>',
        unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("**Expected sources**")
        st.markdown(_id_chips(case.get("expected_sources")),
                    unsafe_allow_html=True)
        st.markdown("**Retrieved into context**")
        st.markdown(_id_chips(case.get("retrieved_sources")),
                    unsafe_allow_html=True)
    with right:
        st.markdown("**Relevant retrieved**")
        st.markdown(_id_chips(case.get("relevant_sources")),
                    unsafe_allow_html=True)
        st.markdown("**Authorized candidate pool**")
        st.markdown(_id_chips(case.get("candidate_sources")),
                    unsafe_allow_html=True)

    if security.get("unauthorized_documents"):
        st.error("Unauthorized documents: "
                 + ", ".join(security["unauthorized_documents"]))
    if security.get("leaked_evidence"):
        st.error("Leak evidence: " + " | ".join(security["leaked_evidence"]))

    generation = case.get("generation") or {}
    if generation.get("judge_notes"):
        st.caption(f"Judge: {generation['judge_notes']}")
    if generation.get("unsupported_claims"):
        st.caption("Unsupported claims: "
                   + " | ".join(generation["unsupported_claims"]))

    with st.expander("Actual answer"):
        st.markdown(case.get("answer") or "(no answer)")


def _evaluation_screen(user: dict, base_url: str) -> None:
    _sidebar(user, base_url)

    st.markdown(
        '<div class="hero"><div class="eyebrow">Saved results &middot; No automatic runs</div>'
        '<h1>Evaluation overview</h1><p>Every case is executed '
        'through the same permission-aware pipeline as the chat interface. '
        'Retrieval, generation and security are scored separately.</p></div>',
        unsafe_allow_html=True)

    selected_report = st.session_state.get("evaluation_report")
    payload = _eval_summary(base_url, selected_report)
    if payload is None:
        st.error("The evaluation service is unreachable.")
        return

    available_reports = payload.get("available_reports") or []
    report_ids = [row.get("report_id") for row in available_reports if row.get("report_id")]
    if report_ids:
        current_report = payload.get("report_id") or report_ids[0]
        if current_report not in report_ids:
            current_report = report_ids[0]
        selected_report = st.selectbox(
            "Saved evaluation run", report_ids,
            index=report_ids.index(current_report), key="evaluation_report_select",
            format_func=lambda value: next(
                (f"{value} - {row.get('run_id') or 'checkpoint'} - "
                 f"{(row.get('summary') or {}).get('total_cases', 0)} cases"
                 for row in available_reports if row.get("report_id") == value), value))
        if selected_report != st.session_state.get("evaluation_report"):
            st.session_state.evaluation_report = selected_report
            st.rerun()

    controls, spacer = st.columns([1, 2])
    with controls:
        if st.button("Refresh saved results", use_container_width=True, type="primary"):
            st.rerun()
    with spacer:
        dataset = payload.get("dataset") or {}
        st.caption(
            f"Golden dataset {dataset.get('dataset_id', '')} v"
            f"{dataset.get('version', '')} - {dataset.get('total_cases', 0)} "
            f"cases, {dataset.get('security_cases', 0)} security relevant. "
            "The dataset is ground truth and is never modified by a run.")

    if not payload.get("has_results"):
        st.info("No evaluation results are available.")
        return
    if not payload.get("summary"):
        st.info(f"Saved report {payload.get('run_id') or '(checkpoint)'} contains "
                f"{payload.get('total_cases', 0)} recorded cases. Detailed metrics, "
                "answers and evidence require an authenticated operator account.")
        return

    with st.expander("Testing instructions"):
        st.markdown("Start with the offline checks in **guide.md**. Preview a selection with:")
        st.code(r".\.venv\Scripts\python.exe -m evaluation.runner --plan --only E001,E033,E056", language="powershell")
        st.caption("Live runs are started from your terminal and use provider quota. "
                   "Refreshing this page only reads saved results.")

    summary = payload.get("summary") or {}
    retrieval = summary.get("retrieval") or {}
    generation = summary.get("generation") or {}
    security = summary.get("security") or {}
    total_cases = dataset.get("total_cases", 0)
    recorded = summary.get("total_cases", 0)
    selected_cases = payload.get("selected_cases") or recorded
    completed = summary.get("passed", 0) + summary.get("failed", 0)
    st.progress(min(recorded / total_cases, 1.0) if total_cases else 0.0,
                text=f"{recorded} of {total_cases} golden cases recorded - "
                     f"{completed} fully scored, {summary.get('errors', 0)} errors")
    if selected_cases != total_cases:
        st.warning(f"Limited run: {selected_cases} of {total_cases} golden cases were "
                   "selected. Its metrics are not a full-system score.")
    if payload.get("schema_version") != 3:
        st.warning("Legacy report: these results predate the evaluation fixes. "
                   "Keep them as a baseline and start a new run when quota is available.")
    elif payload.get("partial") or summary.get("errors"):
        st.warning("Partial evaluation. Unscored cases and API errors do not count as passes.")
    if payload.get("stop_reason"):
        st.caption(f"Run stopped: {payload['stop_reason']}")
    progress = payload.get("progress") or {}
    if progress.get("phase") in {"waiting", "evaluating"}:
        st.info(f"Checkpoint is active: {progress['phase']}"
                + (f" case {progress['case_id']}" if progress.get("case_id") else ""))

    st.write("")
    cards = st.columns(4)
    cards[0].markdown(
        f'<div class="kpi"><div class="l">Cases</div>'
        f'<div class="v">{recorded} / {selected_cases}</div>'
        f'<div class="s">{summary.get("passed", 0)} passed &middot; '
        f'{summary.get("failed", 0)} failed &middot; '
        f'{summary.get("errors", 0)} errors; {total_cases} golden total</div></div>',
        unsafe_allow_html=True)
    cards[1].markdown(
        f'<div class="kpi"><div class="l">Recall@K</div>'
        f'<div class="v">{_pct(retrieval.get("recall_at_k"))}</div>'
        f'<div class="s">precision '
        f'{_pct(retrieval.get("precision_at_k"))} &middot; MRR '
        f'{_num(retrieval.get("mrr"))}</div></div>', unsafe_allow_html=True)
    cards[2].markdown(
        f'<div class="kpi"><div class="l">Groundedness</div>'
        f'<div class="v">{_pct(generation.get("groundedness"))}</div>'
        f'<div class="s">relevancy '
        f'{_pct(generation.get("answer_relevancy"))} &middot; correctness '
        f'{_pct(generation.get("answer_correctness"))}</div></div>',
        unsafe_allow_html=True)
    cards[3].markdown(
        f'<div class="kpi"><div class="l">Authorization</div>'
        f'<div class="v">'
        f'{_pct(security.get("authorization_accuracy"))}</div>'
        f'<div class="s">leakage '
        f'{_pct(security.get("answer_leakage_rate"))} &middot; unauthorized '
        f'retrieval {_pct(security.get("unauthorized_retrieval_rate"))}'
        f'</div></div>', unsafe_allow_html=True)

    st.write("")
    st.caption(
        f"Run {payload.get('run_id', '')} &middot; "
        f"{payload.get('generated_at', '')} &middot; "
        f"{payload.get('duration_s', 0)}s &middot; model "
        f"{payload.get('model', '')} &middot; prompt injection blocked "
        f"{_pct(security.get('prompt_injection_blocked'))} "
        f"({security.get('prompt_injection_cases', 0)} cases)")

    st.caption(
        f"Run selection: {selected_cases} of {total_cases} golden cases; "
        f"results recorded: {recorded}. Retrieval {retrieval.get('scored_cases', 0)} cases; "
        f"generation {generation.get('scored_cases', 0)}; "
        f"security {security.get('scored_cases', 0)}. "
        "Metrics use only measured cases; N/A means unmeasured or not applicable.")
    st.info("Detailed evaluation answers and evidence require an authenticated "
            "operator account. The local demo identity selector does not grant access.")
    return


def _example_prompts(user: dict) -> list[str]:
    scope = user.get("seller_scope") or []
    prompts = ([SELLER_PROMPTS[s] for s in scope if s in SELLER_PROMPTS]
               + DEPARTMENT_PROMPTS.get(user.get("department"), ORG_PROMPTS))
    return list(dict.fromkeys(prompts))[:4]


def _history_payload() -> list[dict]:
    return [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if not m.get("failed") and not m.get("partial")
            and not m.get("refused")][-st.session_state.history_turns:]


def _strip_inline_sources(text: str) -> str:
    # Streamlit's Markdown renderer treats raw HTML in table cells as text.
    # Convert model-produced line breaks to readable inline separators first.
    text = re.sub(r"<br\s*/?>", " · ", text, flags=re.IGNORECASE)
    # The model occasionally emits this drafting label despite the prompt.
    # Source details are already shown in the dedicated Sources control.
    text = re.sub(r"\s*\[Evidence\]", "", text, flags=re.IGNORECASE)
    for marker in ("\nSources:\n", "\nEvidence available in context:\n"):
        index = text.find(marker)
        while index != -1:
            after = text[index + len(marker):]
            if not after.strip() or after.lstrip().startswith("- "):
                return text[:index].rstrip()
            index = text.find(marker, index + 1)
    if text.rstrip().endswith(("Sources:", "Evidence available in context:")):
        return text.rstrip().rsplit("\n", 1)[0].rstrip()
    return text


NO_EVIDENCE_MESSAGE = (
    "I couldn't find relevant information in the sources available for this "
    "account. If you expected an answer, ask your administrator to review "
    "which sources are available to your role."
)
NO_EVIDENCE_PATTERN = re.compile(
    r"(?is)^\s*(?:the\s+)?(?:provided\s+)?(?:passages|context|sources)\b"
    r".{0,180}\b(?:do not|don't|does not|doesn't)\b"
    r".{0,80}\b(?:contain|include|mention|describe)\b"
)


def _completion_label(outcome: str | None) -> str:
    return {
        "answer": "Answer completed.",
        "partial": "Partial answer: generation ended before completion; this answer is not in conversation memory.",
        "refusal": "Request refused by the authorization policy.",
        "interrupted": "Answer interrupted: the connection ended before completion.",
        "error": "Answer failed and is not part of the conversation memory.",
    }.get(outcome, "Answer failed and is not part of the conversation memory.")


def _stream_answer(user: dict, base_url: str, question: str,
                   seller_filter: list[str] | None, history: list[dict]) -> dict:
    events = api_client.ask_demo_stream(user["user_id"], question,
                                   conversation=history,
                                   seller_filter=seller_filter,
                                   base_url=base_url)
    captured: dict = {}
    for event in events:
        kind = event.get("type")
        if kind == "meta":
            captured["meta"] = event
        elif kind == "delta":
            captured.setdefault("deltas", []).append(event.get("text", ""))
        elif kind == "done":
            captured["done"] = event
        elif kind == "error":
            captured["error"] = event.get("message", "Generation failed.")
            break
        else:
            captured["error"] = "The service returned an invalid stream event."
            break
    if captured.get("error"):
        st.error(_safe(captured["error"]))
        return {"answer": "", "meta": captured.get("meta", {}),
                "outcome": "error", "failed": True,
                "message": captured["error"]}
    if not captured.get("done"):
        st.error("The connection ended before the answer completed. Please try again.")
        return {"answer": "", "meta": captured.get("meta", {}),
                "outcome": "interrupted", "failed": True,
                "message": "The connection ended before completion."}
    done = captured["done"]
    answer = _strip_inline_sources(_safe(done.get("answer", "")))
    no_evidence = bool(NO_EVIDENCE_PATTERN.search(answer))
    if no_evidence:
        answer = NO_EVIDENCE_MESSAGE
    outcome = done.get("outcome")
    if outcome not in ("answer", "partial", "refusal"):
        return {"answer": "", "meta": done, "outcome": "error",
                "failed": True, "message": "Invalid terminal outcome."}
    if outcome in ("answer", "partial") and not answer.strip():
        return {"answer": "", "meta": done, "outcome": "error",
                "failed": True, "message": "The service returned an empty answer."}
    meta = dict(done)
    if no_evidence:
        meta["sources"] = []
    return {"answer": answer, "outcome": outcome,
            "refused": outcome == "refusal", "partial": outcome == "partial",
            "meta": meta}


def _chat_screen(user: dict, base_url: str) -> None:
    _sidebar(user, base_url)

    scope = user.get("seller_scope") or []
    options = _seller_options(scope)
    seller_filter = _seller_filter(options)
    typed = st.chat_input("Ask about sellers, tickets, incidents or policies",
                          max_chars=MAX_QUESTION_CHARS)
    question = st.session_state.active_question
    if question:
        history = st.session_state.active_history or []
        st.session_state.active_question = None
        st.session_state.active_history = None
    else:
        question = typed or st.session_state.pending
        st.session_state.pending = None
        if question:
            # Commit the user turn and remove the welcome suggestions in a
            # quick rerun before starting any retrieval/model work.
            st.session_state.active_history = _history_payload()
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.active_question = question
            st.rerun()

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
                    st.warning(_completion_label(message.get("outcome")))
                else:
                    st.markdown(message["content"])
                    if message.get("partial"):
                        st.warning("This answer was cut off by the output limit and is not in conversation memory.")
                if message["role"] == "assistant":
                    st.caption(_completion_label(message.get("outcome")))
                if message["role"] == "assistant":
                    if (not message.get("failed")
                            and not message.get("refused")
                            and not message.get("partial")
                            and message.get("content")):
                        _copy_button(message["content"],
                                     key=f"copy-{message['id']}")
                    _render_sources(message.get("sources", []))
                    _render_stats(message.get("stats", {}),
                                  message.get("effective_sellers", []))
    elif not question:
        st.markdown(
            f'<div class="hero"><div class="eyebrow">Permission-aware knowledge assistant</div>'
            f'<h1>How can I help, {html.escape(user["name"].split()[0])}?</h1>'
            f'<p>Ask about the information you are authorized to access.</p></div>',
            unsafe_allow_html=True)
        st.markdown('<div class="sec">Suggested questions</div>',
                    unsafe_allow_html=True)
        prompt_columns = st.columns(2)
        for index, prompt in enumerate(_example_prompts(user)):
            with prompt_columns[index % 2]:
                if st.button(prompt, key=f"ex_{index}", use_container_width=True):
                    st.session_state.pending = prompt
                    st.rerun()

    if not question:
        return

    with st.chat_message("assistant", avatar=AVATARS.get("assistant")):
        stage = st.status("Working on your request...", state="running")
        try:
            result = _stream_answer(user, base_url, question, seller_filter, history)
        except api_client.ApiError as exc:
            stage.update(label="Service error", state="error")
            st.error(str(exc))
            st.session_state.messages.append({
                "role": "assistant", "content": "", "failed": True,
                "outcome": "error"})
            st.rerun()
        except Exception:
            stage.update(label="Something went wrong", state="error")
            st.error("Something went wrong while answering. Please try again.")
            st.session_state.messages.append({
                "role": "assistant", "content": "", "failed": True,
                "outcome": "error"})
            st.rerun()

        meta = result.get("meta") or {}
        sources = meta.get("sources", []) or []
        stats = meta.get("stats", {}) or {}
        effective = meta.get("effective_sellers", []) or []

        if result.get("failed"):
            outcome = result.get("outcome", "error")
            stage.update(label=_completion_label(outcome), state="error")
            st.session_state.messages.append({
                "role": "assistant", "content": result.get("answer", ""),
                "failed": True, "outcome": outcome})
            st.rerun()

        if result.get("partial"):
            stage.update(label="Partial answer", state="error")
            st.markdown(result["answer"])
            st.warning(_completion_label("partial"))
            _render_sources(sources)
            _render_stats(stats, effective)
        elif not result.get("refused"):
            stage.update(label="Answer ready", state="complete")
            st.markdown(result["answer"])
            _copy_button(result["answer"], key="copy-live")
            _render_sources(sources)
            _render_stats(stats, effective)
        else:
            stage.update(label="Access filtered", state="complete")
            _render_refusal()
        st.caption(_completion_label(result.get("outcome")))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "outcome": result.get("outcome"),
        "refused": bool(result.get("refused")),
        "partial": bool(result.get("partial")),
        "sources": sources,
        "stats": stats,
        "effective_sellers": effective,
    })
    st.rerun()


def main() -> None:
    # The matrix needs room, so the evaluation view runs in the wide layout
    # while the chat stays in the narrow reading column.
    evaluation_view = st.session_state.get("view") == "evaluation"
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON,
                       layout="wide" if evaluation_view else "centered",
                       initial_sidebar_state="expanded")
    _init_state()
    st.markdown(_css(st.session_state.theme), unsafe_allow_html=True)
    if evaluation_view:
        st.markdown("<style>.block-container{ max-width: 88rem; }</style>",
                    unsafe_allow_html=True)
    base_url = _base_url()

    if st.session_state.user is None:
        _login_sidebar(base_url)
        _login_screen(base_url)
    elif st.session_state.view == "evaluation":
        _evaluation_screen(st.session_state.user, base_url)
    else:
        _chat_screen(st.session_state.user, base_url)


if __name__ == "__main__":
    main()

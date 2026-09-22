"""Streamlit application layer for the permission-aware knowledge system.

Thin client over the Phase 5 API. This process holds no retrieval, ranking or
permission logic: it selects a prototype identity and renders what the server
returns. The UI is not the security boundary.
"""
from __future__ import annotations

import html
import json
import os

import pandas as pd
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

AVATARS = {"user": "\N{ADULT}\u200d\N{BRIEFCASE}", "assistant": "\N{ROBOT FACE}"}


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
  @keyframes rise-in{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
  @keyframes pulse-soft{0%,100%{box-shadow:0 0 0 0 color-mix(in oklch,var(--primary) 0%,transparent)}50%{box-shadow:0 0 0 5px color-mix(in oklch,var(--primary) 12%,transparent)}}
  .stApp{ background: var(--bg); color: var(--fg); }
  #MainMenu, footer{ visibility: hidden; }
  header[data-testid="stHeader"]{
    background: transparent !important;
    visibility: visible;
  }
  /* Sidebar open/close control: force a high-contrast black arrow.
     Streamlit has used both collapsedControl and stSidebarCollapsedControl
     test ids across releases, so cover both variants. */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"]{
    visibility: visible !important;
  }
  [data-testid="collapsedControl"] button,
  [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="collapsedControl"] button *,
  [data-testid="stSidebarCollapsedControl"] button *{
    color:#000 !important;
    fill:#000 !important;
    stroke:#000 !important;
  }
  [data-testid="collapsedControl"] button svg,
  [data-testid="stSidebarCollapsedControl"] button svg{
    color:#000 !important;
    fill:none !important;
    stroke:#000 !important;
    stroke-width:2.5 !important;
  }
  .block-container{ padding-top: 2.2rem; max-width: 52rem; }
  .stApp, .stMarkdown, p, li, label, h1, h2, h3{ color: var(--fg); }
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
    font-weight:600; font-size:.92rem; letter-spacing:.005em;
    padding:.6rem 1.15rem; min-height:2.85rem; line-height:1.2;
    transition:transform .16s ease, background .16s ease, border-color .16s ease, box-shadow .16s ease;
    box-shadow: var(--shadow);
  }
  .stButton > button:hover{
    border-color: var(--primary); color: var(--primary);
    background: color-mix(in oklch, var(--primary) 7%, var(--card));
    transform:translateY(-1px); box-shadow:0 5px 14px color-mix(in oklch,var(--fg) 10%,transparent);
  }
  .stButton > button:active{ transform:translateY(0) scale(.98); }
  .stButton > button[kind="primary"]{
    background: var(--primary); color: var(--primary-fg);
    border-color: var(--primary);
    box-shadow:0 4px 12px color-mix(in oklch,var(--primary) 22%,transparent);
  }

  .stApp .st-key-switch_user button{
    height:2.5rem !important; min-height:2.5rem !important;
    max-height:2.5rem !important; width:100% !important;
    padding:0 .6rem !important; font-size:.85rem !important; line-height:1 !important;
    font-weight:600; border-radius:999px;
    display:inline-flex; align-items:center; justify-content:center;
    white-space:nowrap; overflow:hidden;
    background: var(--card); color: var(--fg);
    border:1px solid var(--border);
  }
  .stApp .st-key-switch_user button div{ flex:0 0 auto !important; }
  .stApp .st-key-switch_user button p{ margin:0 !important; }

  /* Light / Dark sliding toggle: bold active label, blue pill track,
     white knob, crater dots on the empty side of the track.
     NOTE: st.toggle renders as data-testid="stCheckbox" (not stToggle). */
  .ld-label{
    font-weight:800; font-size:.92rem; letter-spacing:-.01em;
    color:var(--muted-fg); text-align:right; line-height:1.2;
    white-space:nowrap; margin:0; padding:.2rem 0;
  }
  .ld-label.right{ text-align:left; }
  .ld-label.on{ color:var(--fg); }
  /* tight toggle row: minimal column gaps */
  [data-testid="stHorizontalBlock"]:has(.st-key-theme_toggle_sidebar),
  [data-testid="stHorizontalBlock"]:has(.st-key-theme_toggle_login){
    gap:.3rem !important;
  }
  [data-testid="stCheckbox"]{
    display:flex !important; justify-content:center !important;
    margin:0 !important; padding:0 !important;
  }
  [data-testid="stCheckbox"] label{
    margin:0 !important; padding:0 !important;
    display:flex !important; align-items:center !important;
  }
  /* track = the label's direct div (excluding the hidden widget label) */
  [data-testid="stCheckbox"] label > div:not([data-testid="stWidgetLabel"]){
    background:#5b9dff !important; border:none !important;
    width:3rem !important; min-width:3rem !important;
    height:1.65rem !important; min-height:1.65rem !important;
    margin:0 !important; padding:3px !important;
    border-radius:999px !important; position:relative !important;
    display:flex !important; align-items:center !important;
    box-shadow:inset 0 1px 3px rgba(15,23,42,.28) !important;
  }
  /* knob */
  [data-testid="stCheckbox"] label > div:not([data-testid="stWidgetLabel"]) > div{
    background:#fff !important;
    width:calc(1.65rem - 6px) !important; height:calc(1.65rem - 6px) !important;
    min-width:calc(1.65rem - 6px) !important;
    border-radius:999px !important; margin:0 !important; padding:0 !important;
    transform:translateX(0) !important;
    transition:transform 150ms ease !important;
    box-shadow:0 1px 2px rgba(15,23,42,.35) !important;
  }
  [data-testid="stCheckbox"] label[data-selected] > div:not([data-testid="stWidgetLabel"]) > div{
    transform:translateX(calc(3rem - 1.65rem)) !important;
  }
  /* crater dots on the empty side of the track */
  [data-testid="stCheckbox"] label > div:not([data-testid="stWidgetLabel"])::after{
    content:''; position:absolute; top:0; bottom:0; width:1.5rem;
    right:.2rem; pointer-events:none;
    background-image:
      radial-gradient(circle, rgba(255,255,255,.95) 2.4px, transparent 2.9px),
      radial-gradient(circle, rgba(255,255,255,.9) 1.4px, transparent 1.9px);
    background-repeat:no-repeat;
    background-position:78% 32%, 55% 68%;
  }
  [data-testid="stCheckbox"] label[data-selected] > div:not([data-testid="stWidgetLabel"])::after{
    right:auto; left:.2rem;
    background-position:22% 32%, 45% 68%;
  }

  [data-testid^="stChatMessageAvatar"], .stChatMessageAvatar{
    background: var(--secondary) !important;
    border:1px solid var(--border) !important;
    font-size:1.12rem;
  }

  .idlist{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.1rem 0 .4rem; }
  .idlist code{
    background: var(--secondary) !important; color: var(--fg) !important;
    border:1px solid var(--border); border-radius:.35rem;
    padding:.14rem .45rem; font-size:.78rem;
  }
  .idnone{ color: var(--muted-fg); font-size:.82rem; }

  [data-testid="stExpander"]{
    background: var(--card); border:1px solid var(--border);
    border-radius: var(--radius);
  }
  [data-testid="stExpander"] summary{ color: var(--fg) !important; font-size:.87rem; }
  [data-testid="stExpander"] summary p,
  [data-testid="stExpander"] summary span,
  [data-testid="stExpander"] summary svg{ color: var(--fg) !important; fill: currentColor; }

  /* Chat input: force every layer (bottom bar, wrapper, textarea,
     send button) onto theme vars. Streamlit paints the bottom container
     and inner wrappers white by default, which leaks through in dark mode. */
  [data-testid="stBottom"],
  [data-testid="stBottomBlockContainer"],
  [data-testid="stChatInput"]{
    background: var(--bg) !important; border-color: transparent !important;
  }
  [data-testid="stBottom"] > div,
  [data-testid="stBottomBlockContainer"] > div{
    background: var(--bg) !important;
  }
  [data-testid="stChatInput"] > div,
  [data-testid="stChatInput"] form,
  [data-testid="chatInput"]{
    background: var(--card) !important; border-color: var(--border) !important;
  }
  [data-testid="stChatInput"]{
    background: var(--card) !important;
    border:1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
  }
  [data-testid="stChatInput"]:focus-within{
    border-color: var(--primary) !important;
    box-shadow:0 0 0 1px var(--primary);
  }
  [data-testid="stChatInput"] textarea{
    background: transparent !important;
    color: var(--fg) !important; font-size:.95rem;
    caret-color: var(--primary);
  }
  [data-testid="stChatInput"] textarea::placeholder{ color: var(--muted-fg) !important; opacity:1; }
  /* send (arrow) button inside the chat input: fixed circle, centred icon.
     The stray inner square in dark mode comes from inner spans/svg
     inheriting button borders/backgrounds, so strip them explicitly. */
  [data-testid="stChatInput"] > div{
    display:flex !important; align-items:center !important; gap:.5rem;
  }
  [data-testid="stChatInput"] button{
    flex:0 0 2rem !important; width:2rem !important; height:2rem !important;
    min-height:0 !important; min-width:2rem !important;
    padding:0 !important; margin:0 !important; align-self:center !important;
    display:inline-flex !important; align-items:center !important;
    justify-content:center !important;
    background: var(--secondary) !important; color: var(--muted-fg) !important;
    border:1px solid var(--border) !important; border-radius:50% !important;
    box-shadow:none !important; outline:none !important;
  }
  [data-testid="stChatInput"] button:hover:not(:disabled){
    background: var(--primary) !important; color: var(--primary-fg) !important;
    border-color: var(--primary) !important; transform:none;
  }
  [data-testid="stChatInput"] button:disabled{
    opacity:.55;
  }
  [data-testid="stChatInput"] button span,
  [data-testid="stChatInput"] button div,
  [data-testid="stChatInput"] button svg,
  [data-testid="stChatInput"] button svg *{
    background:transparent !important; border:none !important;
    box-shadow:none !important; outline:none !important;
  }
  [data-testid="stChatInput"] button svg{
    width:1rem !important; height:1rem !important; display:block !important;
    flex:0 0 auto !important;
    fill:none !important; stroke:currentColor !important;
    stroke-width:2.2 !important; stroke-linecap:round !important;
    stroke-linejoin:round !important;
  }
  /* Chrome autofill / webkit background fix inside chat input */
  [data-testid="stChatInput"] textarea:-webkit-autofill,
  [data-testid="stChatInput"] textarea:-webkit-autofill:hover,
  [data-testid="stChatInput"] textarea:-webkit-autofill:focus{
    -webkit-text-fill-color: var(--fg) !important;
    -webkit-box-shadow:0 0 0 100px var(--card) inset !important;
  }

  .brand{ display:flex; align-items:center; gap:.55rem; margin-bottom:.15rem; }
  .brand-mark{
    width:1.75rem; height:1.75rem; border-radius:.5rem;
    background: var(--primary); color: var(--primary-fg);
    display:flex; align-items:center; justify-content:center;
    font-size:.95rem; flex:0 0 auto;
    animation:pulse-soft 3.2s ease-in-out infinite;
  }
  .brand-name{ font-weight:700; font-size:1rem; letter-spacing:-.01em; }
  .brand-sub{ color:var(--muted-fg); font-size:.78rem; margin:0 0 .2rem 2.3rem; }

  .hero{ text-align:center; padding: 1.8rem 0 1.5rem; }
  .hero, .kpi, .ucard, .src, [data-testid="stChatMessage"]{ animation:rise-in .35s ease both; }
  .hero h1{
    font-size: 2.3rem; font-weight:700; letter-spacing:-.025em;
    margin:0 0 .5rem; color: var(--fg);
  }
  .hero p{ color: var(--muted-fg); font-size:1rem; margin:0 auto; max-width:34rem; }
  .eyebrow{
    color:var(--primary); font-size:.7rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.12em; margin-bottom:.65rem;
  }
  .chip, .chip-muted{
    display:inline-flex; align-items:center; border:1px solid var(--border);
    border-radius:999px; padding:.25rem .55rem; margin:.15rem .2rem .15rem 0;
    font-size:.73rem; font-weight:600; background:var(--secondary);
  }
  .chip{ color:var(--primary); }
  .chip-muted{ color:var(--muted-fg); }
  .stButton > button:focus-visible{
    outline:2px solid var(--primary); outline-offset:2px;
  }
  .kpi{ min-height:7.2rem; }
  @media (max-width:640px){
    .block-container{ padding:1.4rem 1rem; }
    .hero{ padding:1.2rem 0; }
    .hero h1{ font-size:1.85rem; }
    .kpi .v{ font-size:1.25rem; }
  }

  .sec{
    font-size:.68rem; font-weight:700; letter-spacing:.1em;
    color:var(--muted-fg); text-transform:uppercase; margin:.1rem 0 .55rem;
  }
  .ucard{
    border:1px solid var(--border); border-radius:var(--radius);
    background:var(--card); padding:.85rem .95rem; box-shadow: var(--shadow);
    transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  }
  .ucard:hover{ transform:translateY(-2px); border-color:var(--primary); box-shadow:0 8px 20px color-mix(in oklch,var(--fg) 10%,transparent); }
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
    box-shadow: var(--shadow); transition:transform .18s ease, box-shadow .18s ease;
  }
  [data-testid="stChatMessage"]:hover{ transform:translateY(-1px); box-shadow:0 7px 18px color-mix(in oklch,var(--fg) 9%,transparent); }
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

  .evrow{
    display:flex; justify-content:space-between; align-items:baseline;
    font-size:.79rem; padding:.16rem 0; color:var(--muted-fg);
  }
  .evrow b{ color:var(--fg); font-weight:620; font-variant-numeric:tabular-nums; }
  .evhead{
    font-size:.68rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:var(--muted-fg);
    margin:.6rem 0 .2rem; padding-top:.45rem;
    border-top:1px solid var(--border);
  }
  .evtally{ display:flex; gap:.4rem; margin:.15rem 0 .5rem; }
  .evpill{
    flex:1; text-align:center; border:1px solid var(--border);
    border-radius:.4rem; padding:.35rem .2rem; background:var(--bg);
  }
  .evpill .n{ font-size:1.05rem; font-weight:700; display:block;
    font-variant-numeric:tabular-nums; }
  .evpill .l{ font-size:.62rem; text-transform:uppercase;
    letter-spacing:.06em; color:var(--muted-fg); }
  .evpill.ok .n{ color:#16a34a; }
  .evpill.bad .n{ color: var(--destructive); }
  .evpill.warn .n{ color:#d97706; }
  .evempty{
    border:1px dashed var(--border); border-radius:var(--radius);
    padding:.7rem .8rem; font-size:.78rem; color:var(--muted-fg);
    background:var(--bg);
  }
  .kpi{
    border:1px solid var(--border); border-radius:var(--radius);
    background:var(--card); padding:.7rem .85rem; box-shadow:var(--shadow);
  }
  .kpi .l{ font-size:.68rem; text-transform:uppercase; letter-spacing:.07em;
    color:var(--muted-fg); }
  .kpi .v{ font-size:1.5rem; font-weight:700; letter-spacing:-.02em;
    font-variant-numeric:tabular-nums; }
  .kpi .s{ font-size:.72rem; color:var(--muted-fg); }

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
    st.session_state.setdefault("view", "chat")


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


def _sign_out() -> None:
    st.session_state.user = None
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
        _theme_toggle("theme_toggle_login")
        st.markdown(
            '<div class="guard">&#128274; In production this identity '
            'selector would be an authenticated SSO session. The server '
            'derives permissions from the identity and never trusts a '
            'client-supplied scope.</div>', unsafe_allow_html=True)


def _login_screen(base_url: str) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Your enterprise knowledge workspace</div>'
        '<h1>Find answers. Know the source.</h1>'
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
                f'{html.escape(_safe(source.get("label", "")))}</div>'
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

    with st.expander("Testing instructions"):
        st.markdown("Start with the offline checks in **guide.md**. Preview a selection with:")
        st.code(r".\.venv\Scripts\python.exe -m evaluation.runner --plan --only E001,E033,E056", language="powershell")
        st.caption("Live runs are started from your terminal and use provider quota. "
                   "Refreshing this page only reads saved results.")

    if not payload.get("has_results"):
        st.markdown(
            '<div class="evempty">No evaluation results available. Run the '
            'evaluation to generate results.</div>', unsafe_allow_html=True)
        return

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
    if payload.get("schema_version") != 2:
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
    if not payload.get("reports_enabled"):
        st.info("Case answers span multiple identities. Detailed reports are disabled "
                "by default; see guide.md for trusted local operator access.")
        return
    try:
        detail = api_client.evaluation_matrix(base_url, report=payload.get("report_id"))
    except api_client.ApiError as exc:
        st.error(str(exc))
        return
    frame = _matrix_frame(detail.get("matrix", []))
    if frame.empty:
        st.info("The run produced no rows.")
        return

    st.markdown('<div class="sec">Evaluation matrix</div>',
                unsafe_allow_html=True)
    filters = st.columns(3)
    outcome = filters[0].selectbox(
        "Show", ["All", "Passed", "Failed", "Errors", "Security relevant",
                 "Retrieval scored", "Generation scored"],
        key="eval_outcome")
    category = filters[1].selectbox(
        "Category", ["All categories"] + sorted(frame["Category"].unique()),
        key="eval_category")
    who = filters[2].selectbox(
        "User", ["All users"] + sorted(frame["User"].unique()),
        key="eval_user")

    filtered = _filter_matrix(frame, outcome, category, who)
    st.caption(f"{len(filtered)} of {len(frame)} cases")
    if filtered.empty:
        st.info("No cases match these filters. Try a different outcome, category or user.")
        return
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Recall@K": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Precision@K": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "MRR": st.column_config.NumberColumn(format="%.2f"),
            "Groundedness": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Relevancy": st.column_config.NumberColumn(format="%.2f"),
            "Correctness": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    by_category = summary.get("by_category") or {}
    if by_category:
        with st.expander("Results by category"):
            st.dataframe(
                pd.DataFrame([
                    {"Category": name, "Total": stats["total"],
                     "Passed": stats["passed"], "Failed": stats["failed"],
                     "Errors": stats["errors"]}
                    for name, stats in sorted(by_category.items())
                ]), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">Case detail</div>', unsafe_allow_html=True)
    options = list(filtered["ID"]) if not filtered.empty else list(frame["ID"])
    chosen = st.selectbox("Inspect a case", options, key="eval_case")
    _render_case_detail(detail.get("results", []), chosen)


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
                   seller_filter: list[str] | None, history: list[dict]) -> dict:
    events = api_client.ask_stream(user["user_id"], question,
                                   conversation=history,
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
    if not captured.get("done"):
        st.error("The connection ended before the answer completed. Please try again.")
        return {"answer": answer, "meta": captured.get("meta", {}), "failed": True}
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
            f'<div class="hero"><div class="eyebrow">Ask &middot; Explore &middot; Verify</div>'
            f'<h1>Hello, {html.escape(user["name"].split()[0])}</h1>'
            f'<p>Ask about the sellers, tickets, incidents, runbooks and '
            f'policies you are authorized to read.</p></div>',
            unsafe_allow_html=True)
        st.markdown('<div class="sec">Suggested questions</div>',
                    unsafe_allow_html=True)
        prompt_columns = st.columns(2)
        for index, prompt in enumerate(_example_prompts(scope)):
            with prompt_columns[index % 2]:
                if st.button(prompt, key=f"ex_{index}", use_container_width=True):
                    st.session_state.pending = prompt
                    st.rerun()

    typed = st.chat_input("Ask about sellers, tickets, incidents or policies",
                          max_chars=MAX_QUESTION_CHARS)
    question = typed or st.session_state.pending
    st.session_state.pending = None
    if not question:
        return

    history = _history_payload()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS.get("user")):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS.get("assistant")):
        stage = st.status("Working on your request...", state="running")
        try:
            result = _stream_answer(user, base_url, question, seller_filter, history)
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

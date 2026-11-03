"""Streamlit application layer for the permission-aware knowledge system.

Thin client over the Phase 5 API. This process holds no retrieval, ranking or
permission logic: it selects a prototype identity and renders what the server
returns. The UI is not the security boundary.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import streamlit as st

import api_client

# All presentation code, including themes and icons, lives in this file.
_PATHS = {
    "book": '<path d="M12 7c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1V6c-3-1-6-1-9 1Z"/><path d="M12 7v14M6 10h3m-3 4h3m6-4h3m-3 4h3"/>',
    "shield": '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8 12 3 3 5-6"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    "arrow": '<path d="M5 12h14m-5-5 5 5-5 5"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
    "moon": '<path d="M20 15.2A8.5 8.5 0 0 1 8.8 4 8.5 8.5 0 1 0 20 15.2Z"/>',
    "check": '<path d="m5 12 4 4L19 6"/>',
}


def icon(name: str, size: int = 20) -> str:
    """Decorative SVG; its adjacent text supplies the accessible label."""
    return (f'<svg class="line-icon" width="{size}" height="{size}" '
            'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_PATHS[name]}</svg>')


def brand_mark(extra_class: str = "") -> str:
    return f'<span class="brand-mark {html.escape(extra_class)}">{icon("book", 25)}</span>'


THEMES = {
    "light": """
        --bg:#f7f5f0; --fg:#292c28; --card:#fffefa; --primary:#a64b32;
        --primary-fg:#ffffff; --secondary:#eeeae2; --muted-fg:#6f736a;
        --border:#dedfd6; --destructive:#a83b3b; --success:#42755b;
        --success-soft:#eaf4ed; --warning:#875418; --warning-soft:#fff3da;
        --error-soft:#fbecee; --info:#315f87; --info-soft:#ebf2f8;
        --disabled-bg:#eeeae2; --disabled-fg:#74786e; --sidebar-icon:#111111;
        --accent-soft:#f3e5d9; --sidebar:#efeee7; --shadow:0 3px 14px #393a2810;
        color-scheme:light;
    """,
    "dark": """
        --bg:#1c211f; --fg:#edece4; --card:#252c28; --primary:#edb498;
        --primary-fg:#26201c; --secondary:#303a33; --muted-fg:#b0b9ae;
        --border:#404a42; --destructive:#f0a2a2; --success:#9ccaad;
        --success-soft:#263c2e; --warning:#edc184; --warning-soft:#3b3325;
        --error-soft:#402c2d; --info:#a3c7e5; --info-soft:#283740;
        --disabled-bg:#303a33; --disabled-fg:#a2aa9f; --sidebar-icon:#edece4;
        --accent-soft:#39352b; --sidebar:#202722; --shadow:0 3px 14px #00000018;
        color-scheme:dark;
    """,
}

CSS = """
  @keyframes rise-in { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:none } }
  .stApp { background:var(--bg); color:var(--fg); }
  #MainMenu, footer { visibility:hidden; }
  header[data-testid="stHeader"] { background:transparent; }
  .block-container { max-width:55rem; padding-top:2.3rem; padding-bottom:3rem; }
  .stApp, .stMarkdown, p, li, label, h1, h2, h3 { color:var(--fg); }
  .stApp { font-family:"Segoe UI",sans-serif; }
  h1, h2, h3 { text-wrap:balance; }
  hr { border-color:var(--border)!important; margin:1rem 0!important; }
  code { background:var(--secondary)!important; color:var(--fg)!important;
    padding:.1rem .3rem; border-radius:4px; font-size:.85em; }
  .line-icon { display:inline-block; flex-shrink:0; vertical-align:middle; }
  [data-testid="stSidebar"] { background:var(--sidebar); color:var(--fg); color-scheme:inherit;
    border-right:1px solid var(--border);
    transition:transform 200ms ease,min-width 200ms ease,max-width 200ms ease; }
  [data-testid="stSidebar"][aria-expanded="true"] {
    width:310px!important; min-width:min(310px,100vw)!important;
    max-width:min(310px,100vw)!important; }
  [data-testid="stSidebarContent"] { scrollbar-color:var(--border) var(--sidebar); }
  [data-testid="stSidebarUserContent"] { padding:1.7rem 1.25rem; }
  [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapseButton"] button {
    color:var(--sidebar-icon)!important; background:transparent;
    min-width:44px; min-height:44px; border-radius:9px;
    display:inline-flex; align-items:center; justify-content:center; }
  [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
  [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
  [data-testid="stExpandSidebarButton"] svg,
  [data-testid="stSidebarCollapseButton"] svg {
    color:var(--sidebar-icon)!important; -webkit-text-fill-color:var(--sidebar-icon); }
  [data-testid="stExpandSidebarButton"]:hover,
  [data-testid="stSidebarCollapseButton"] button:hover { background:var(--accent-soft); }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color:var(--muted-fg)!important; font-size:.78rem; line-height:1.6; }
  [data-testid="stCaptionContainer"] { opacity:1!important; }
  [data-testid="InputInstructions"] { color:var(--muted-fg)!important; }
  .stButton > button, .stFormSubmitButton > button {
    background:var(--card); color:var(--fg); border:1px solid var(--border);
    border-radius:10px; min-height:2.8rem; padding:.6rem 1rem;
    font-size:.88rem; font-weight:600; box-shadow:none;
    transition:background .16s,border-color .16s,box-shadow .16s; }
  .stButton > button *, .stFormSubmitButton > button * { color:inherit!important; }
  .stButton > button:hover, .stFormSubmitButton > button:hover {
    background:var(--accent-soft); color:var(--primary); border-color:var(--primary); }
  .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background:var(--primary); color:var(--primary-fg); border-color:var(--primary); }
  .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    filter:brightness(.94); }
  .stButton > button:disabled, .stFormSubmitButton > button:disabled,
  .stButton > button:disabled:hover, .stFormSubmitButton > button:disabled:hover {
    background:var(--disabled-bg); color:var(--disabled-fg); border-color:var(--border);
    cursor:not-allowed; filter:none; }
  html:has(body.copy-frame) { width:max-content; background:var(--bg); }
  body.copy-frame {
    width:max-content; height:40px; margin:0; padding:0 4px; overflow:hidden;
    display:flex; align-items:center; background:var(--bg);
  }
  body.copy-frame .copy-button {
    display:inline-flex; align-items:center; gap:.35rem;
    font:600 12px/1 'Segoe UI',sans-serif; color:var(--muted-fg);
    background:transparent; cursor:pointer; border:1px solid var(--border);
    border-radius:8px; box-sizing:border-box; height:30px; padding:5px 11px;
    transition:all .12s ease;
  }
  body.copy-frame .copy-button:hover { color:var(--primary); border-color:var(--primary); }
  body.copy-frame .copy-button.done {
    color:var(--success); border-color:var(--success); background:var(--success-soft); }
  body.copy-frame .copy-button.error {
    color:var(--destructive); border-color:var(--destructive); background:var(--error-soft); }
  body.copy-frame .copy-button:focus-visible { outline-offset:2px; }
  body.copy-frame .copy-button svg { flex:0 0 auto; }
  body.copy-frame .copy-buffer { position:fixed; opacity:0; }
  button:focus-visible, [data-testid="stSelectbox"] [role="group"]:focus-within {
    outline:2px solid var(--primary)!important; outline-offset:3px; border-radius:10px; }
  [data-testid="stSelectbox"] [role="group"],
  [data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"] {
    background:var(--card)!important; color:var(--fg)!important;
    border-color:var(--border)!important; border-radius:9px; min-height:3rem; }
  [data-testid="stSelectbox"] input,
  [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    color:var(--fg)!important; -webkit-text-fill-color:var(--fg)!important;
    background:transparent!important; caret-color:var(--primary); }
  [data-testid="stSelectbox"] input::placeholder,
  [data-testid="stTextInput"] input::placeholder,
  [data-testid="stTextArea"] textarea::placeholder {
    color:var(--muted-fg)!important; -webkit-text-fill-color:var(--muted-fg)!important; opacity:1; }
  [data-testid="stSelectbox"] [role="group"]:focus-within,
  [data-testid="stTextInputRootElement"]:focus-within,
  [data-testid="stTextAreaRootElement"]:focus-within {
    border-color:var(--primary)!important; box-shadow:0 0 0 2px var(--accent-soft); }
  [data-testid="stSelectbox"] button, [data-testid="stSelectbox"] svg,
  [data-testid="stTextInput"] button {
    color:var(--muted-fg)!important; -webkit-text-fill-color:var(--muted-fg)!important;
    background:transparent!important; }
  [data-testid="stSelectbox"] [role="group"][data-disabled="true"],
  [data-testid="stTextInputRootElement"]:has(input:disabled),
  [data-testid="stTextAreaRootElement"]:has(textarea:disabled) {
    background:var(--disabled-bg)!important; border-color:var(--border)!important; }
  [data-testid="stSelectbox"] [role="group"][data-disabled="true"] *,
  [data-testid="stTextInput"] input:disabled, [data-testid="stTextInput"] button:disabled,
  [data-testid="stTextArea"] textarea:disabled, [data-testid="stWidgetLabel"][disabled] * {
    color:var(--disabled-fg)!important; -webkit-text-fill-color:var(--disabled-fg)!important; }
  [data-testid="stSelectboxVirtualDropdown"] {
    background:var(--card)!important; color:var(--fg)!important;
    border-color:var(--border)!important; box-shadow:var(--shadow); }
  [role="listbox"], [role="option"] {
    background:var(--card)!important; color:var(--fg)!important; }
  [role="listbox"] *, [role="option"] * {
    color:var(--fg)!important; -webkit-text-fill-color:var(--fg)!important; }
  [role="option"]:hover, [role="option"]:focus,
  [role="option"][data-hovered="true"], [role="option"][data-focused="true"] {
    background:var(--secondary)!important; }
  [role="option"][aria-selected="true"],
  [role="option"][aria-selected="true"]:hover { background:var(--accent-soft)!important; }
  [role="option"][aria-disabled="true"], [role="option"][aria-disabled="true"] * {
    color:var(--disabled-fg)!important; -webkit-text-fill-color:var(--disabled-fg)!important; }
  [data-testid="stSlider"] { color:var(--fg); }
  [data-testid="stSliderTickBar"] { color:var(--muted-fg)!important; }
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    background:var(--success-soft); color:var(--success); }
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    background:var(--warning-soft); color:var(--warning); }
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
    background:var(--error-soft); color:var(--destructive); }
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
    background:var(--info-soft); color:var(--info); }
  [data-testid="stAlertContainer"] { border:1px solid currentColor; }
  [data-testid="stAlertContainer"] * { color:inherit!important; }
  [data-testid="stTable"], [data-testid="stTable"] table {
    background:var(--card)!important; color:var(--fg)!important; border-color:var(--border)!important; }
  [data-testid="stTable"] th { background:var(--secondary)!important; }
  [data-testid="stTable"] th, [data-testid="stTable"] td,
  [data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {
    color:var(--fg)!important; border-color:var(--border)!important; }
  [data-testid="stCode"], [data-testid="stCode"] pre {
    background:var(--secondary)!important; color:var(--fg)!important; }
  .brand { display:flex; gap:.7rem; align-items:center; }
  .brand-mark { width:42px; height:42px; display:inline-flex; align-items:center;
    justify-content:center; flex-shrink:0; border-radius:13px; color:#fff9ef;
    background:#a64b32; box-shadow:inset 0 0 0 1px #ffffff18; }
  .brand-name { font-size:.92rem; font-weight:700; letter-spacing:-.03em; }
  .brand-sub { font-size:.72rem; color:var(--muted-fg); margin-top:.1rem; }
  .brand-mark.hero-mark { width:64px; height:64px; border-radius:20px; margin-bottom:1.5rem;
    transform:rotate(-6deg); box-shadow:0 8px 0 -3px var(--accent-soft); }
  .hero-mark .line-icon { width:34px; height:34px; transform:rotate(6deg); }
  .sec, .eyebrow, .login-kicker { color:var(--muted-fg); font-size:.65rem;
    font-weight:700; text-transform:uppercase; letter-spacing:.14em; }
  .sec { margin:.35rem 0 .5rem; }
  .eyebrow { color:var(--primary); margin-bottom:.8rem; }
  .hero { text-align:center; padding:2.8rem 0 2.3rem; animation:rise-in .35s ease both; }
  .hero h1 { font-size:2.65rem; font-weight:600; letter-spacing:-.06em; line-height:1.16;
    color:var(--fg); margin:0 0 1rem; padding:0; }
  .hero h1 em { font-family:Georgia,serif; color:var(--primary); font-weight:400; }
  .hero p { font-size:.95rem; line-height:1.75; color:var(--muted-fg); max-width:33rem; margin:0 auto; }
  .login-hero { text-align:left; padding:2rem 1.5rem 1rem 0; }
  .login-hero h1 { font-size:clamp(2.4rem,3.8vw,3.6rem); }
  .login-hero p { margin:0; max-width:26rem; }
  .feature-list { display:grid; gap:.85rem; margin-top:2rem; }
  .feature { display:flex; align-items:center; gap:.7rem; font-size:.83rem; color:var(--muted-fg); }
  .feature .line-icon { color:var(--primary); }
  .workspace-bar { display:flex; justify-content:space-between; align-items:center;
    gap:1rem; padding:.25rem 0 1rem; border-bottom:1px solid var(--border);
    font-size:.79rem; color:var(--muted-fg); }
  .workspace-label { display:flex; align-items:center; gap:.5rem; }
  .demo-badge { display:inline-flex; align-items:center; gap:.4rem; white-space:nowrap;
    font-size:.65rem; font-weight:600; border:1px solid var(--border); border-radius:6px;
    padding:.3rem .5rem; color:var(--muted-fg); background:var(--card); }
  .demo-badge::before { content:""; width:5px; height:5px; background:var(--primary); border-radius:50%; }
  .st-key-login_card { background:var(--card); border:1px solid var(--border)!important;
    max-width:100%;
    border-radius:18px; padding:1.8rem!important; box-shadow:var(--shadow); margin-top:1.7rem; }
  [data-testid="stForm"] { border-color:var(--border); }
  .st-key-login_card [data-testid="stForm"] { border:0; padding:0; background:transparent; }
  .login-kicker { color:var(--primary); margin-bottom:.7rem; }
  .login-title { font-size:1.65rem; font-weight:600; letter-spacing:-.04em; margin-bottom:.35rem; }
  .login-copy { font-size:.86rem; line-height:1.65; color:var(--muted-fg); margin-bottom:.7rem; }
  .login-footnote { display:flex; gap:.55rem; font-size:.75rem; line-height:1.65;
    padding-top:.8rem; color:var(--muted-fg); border-top:1px solid var(--border); }
  .login-footnote .line-icon { margin-top:.15rem; color:var(--primary); }
  .ucard { display:grid; grid-template-columns:minmax(0,1fr); gap:.12rem;
    margin:.35rem 0 .9rem; padding-left:.7rem; border-left:2px solid var(--border); }
  .step-icon { display:flex; align-items:center; justify-content:center; flex-shrink:0;
    width:30px; height:30px; color:var(--primary); border:1px solid var(--border); border-radius:9px; }
  .ucard .rl { font-weight:650; font-size:.8rem; line-height:1.45; margin:0; }
  .ucard .dp { color:var(--muted-fg); font-size:.75rem; line-height:1.5;
    overflow-wrap:anywhere; }
  .idbox { display:flex; flex-direction:column; gap:.2rem; align-items:flex-start; border:1px solid var(--border);
    border-radius:12px; background:var(--card); padding:.9rem .75rem; }
  .identity-avatar { display:flex; align-items:center; justify-content:center; flex-shrink:0;
    width:36px; height:36px; border-radius:50%; background:var(--accent-soft);
    color:var(--primary); font-size:.76rem; font-weight:700; }
  .idbox .nm { font-weight:600; font-size:.89rem; }
  .idbox .rl { color:var(--primary); font-size:.75rem; }
  .idbox .dp { color:var(--muted-fg); font-size:.69rem; margin-top:.2rem; }
  .chip, .chip-muted { display:inline-flex; gap:.3rem; align-items:center; padding:.25rem .5rem;
    margin:.1rem .2rem .1rem 0; border:1px solid var(--border); border-radius:6px;
    font-size:.71rem; color:var(--primary); background:var(--card); }
  .chip-muted { color:var(--muted-fg); }
  .ld-label { display:flex; align-items:center; justify-content:flex-end; gap:.35rem;
    color:var(--muted-fg); font-size:.75rem; }
  .ld-label.right { justify-content:flex-start; }
  .ld-label.on { color:var(--fg); font-weight:600; }
  [class*="st-key-theme_toggle"] [data-testid="stCheckbox"] { display:flex; justify-content:center; }
  [class*="st-key-theme_toggle"] [data-testid="stCheckbox"] label > div:not([data-testid]) {
    background:var(--border)!important; }
  [class*="st-key-theme_toggle"] [data-testid="stCheckbox"][data-selected="true"] label > div:not([data-testid]) {
    background:var(--primary)!important; }
  [class*="st-key-theme_toggle"] [data-testid="stCheckbox"] label > div:not([data-testid]) > div {
    background:var(--primary-fg)!important; }
  [class*="st-key-theme_toggle"] [data-testid="stCheckbox"]:has(input:focus-visible) {
    outline:2px solid var(--primary); outline-offset:3px; border-radius:6px; }
  .guard { display:flex; gap:.5rem; color:var(--muted-fg); font-size:.72rem; line-height:1.65; padding:.75rem 0; }
  .guard .line-icon { color:var(--primary); margin-top:.15rem; }
  .dot-row { display:flex; align-items:center; gap:.4rem; font-size:.73rem; color:var(--muted-fg); }
  .dot { width:6px; height:6px; border-radius:50%; display:inline-block; }
  .dot-ok { background:var(--success); } .dot-bad { background:var(--destructive); }
  .dot-warn { background:var(--warning); }
  [class*="st-key-ex_"] button { min-height:6.2rem; text-align:left; justify-content:flex-start;
    padding:1.15rem; border-radius:12px; background:var(--card); }
  [class*="st-key-ex_"] button p { font-size:.85rem; line-height:1.55; font-weight:400; }
  [class*="st-key-ex_"] button [data-testid="stIconMaterial"] { color:var(--primary); margin-right:.5rem; }
  .suggestion-footer { display:flex; align-items:center; justify-content:center; gap:.4rem;
    font-size:.72rem; color:var(--muted-fg); margin:1.25rem 0; }
  [data-testid="stChatMessage"] { background:transparent; border:0;
    border-radius:0; padding:.8rem .4rem; margin-bottom:.55rem; animation:rise-in .25s ease both; }
  [data-testid^="stChatMessageAvatar"] { color:var(--primary)!important; background:transparent!important; }
  [data-testid="stChatMessage"] p { line-height:1.75; }
  [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 { font-size:1.05rem; }
  [data-testid="stChatMessage"] th { background:var(--secondary); }
  [data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td { border-color:var(--border); }
  [data-testid="stBottom"], [data-testid="stBottom"] > div,
  [data-testid="stBottomBlockContainer"] { background:var(--bg)!important; }
  [data-testid="stChatInput"] { background:var(--card)!important; border:1px solid var(--border)!important;
    border-radius:16px!important; box-shadow:0 6px 25px #252a2010; padding:.3rem; }
  [data-testid="stBottomBlockContainer"] { max-width:52rem; }
  [data-testid="stChatInput"]:focus-within { border-color:var(--primary)!important; box-shadow:0 0 0 2px var(--accent-soft); }
  [data-testid="stChatInput"] > div { background:transparent!important;
    border:0!important; outline:0!important; box-shadow:none!important; }
  [data-testid="stChatInput"] textarea { color:var(--fg)!important;
    -webkit-text-fill-color:var(--fg); background:transparent!important; caret-color:var(--primary); }
  [data-testid="stChatInput"] textarea::placeholder {
    color:var(--muted-fg)!important; -webkit-text-fill-color:var(--muted-fg); opacity:1; }
  [data-testid="stChatInputSubmitButton"] { background:var(--primary)!important; color:var(--primary-fg)!important; border-radius:10px; }
  [data-testid="stChatInputSubmitButton"]:disabled {
    background:var(--disabled-bg)!important; color:var(--disabled-fg)!important; opacity:1; }
  [data-testid="stChatInputSubmitButton"] svg { fill:currentColor; }
  [data-testid="stExpander"] { border:1px solid var(--border); border-radius:10px; background:var(--card); }
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] summary *,
  [data-testid="stExpander"] [data-testid="stText"],
  [data-testid="stExpander"] [data-testid="stText"] * {
    color:var(--fg)!important; -webkit-text-fill-color:var(--fg)!important; }
  [data-testid="stExpander"] summary { background:var(--card)!important; border-radius:9px; }
  [data-testid="stExpander"] summary p { font-size:.83rem; }
  [data-testid="stExpander"] summary:hover { background:var(--secondary)!important; }
  [data-testid="stExpander"] summary:focus-visible {
    outline:2px solid var(--primary); outline-offset:2px; border-radius:9px; }
  [data-testid="stExpander"] [data-testid="stText"] { white-space:pre-wrap; overflow-wrap:anywhere; }
  [data-testid="stExpander"] [data-testid="stText"] pre { font-family:inherit; color:var(--fg); font-size:.8rem; white-space:pre-wrap; }
  .source-item { padding:.5rem .2rem; border-bottom:1px solid var(--border); }
  .source-item:last-child { border-bottom:0; padding-bottom:.15rem; }
  .source-title { font-size:.84rem; font-weight:600; line-height:1.4; }
  .source-title span { font-size:.78rem; margin-right:.2rem; }
  .source-meta { color:var(--muted-fg); font-size:.7rem; line-height:1.45;
    margin:.16rem 0 0 1.35rem; overflow-wrap:anywhere; }
  .refusal { background:var(--accent-soft); border:1px solid var(--border); border-radius:10px; padding:1rem; font-size:.86rem; }
  .refusal b { display:flex; align-items:center; gap:.5rem; color:var(--primary); margin-bottom:.4rem; }
  .stats { font-size:.7rem; color:var(--muted-fg); margin-top:.65rem; }
  .evrow { display:flex; justify-content:space-between; font-size:.78rem; padding:.2rem 0; color:var(--muted-fg); }
  .evrow b { color:var(--fg); font-variant-numeric:tabular-nums; }
  .evhead { font-size:.65rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted-fg); margin:.6rem 0 .3rem; }
  .evtally { display:flex; gap:.4rem; }
  .evpill { flex:1; text-align:center; padding:.4rem; border:1px solid var(--border); border-radius:8px; }
  .evpill .n { display:block; font-size:1.1rem; font-weight:600; }
  .evpill .l { font-size:.6rem; color:var(--muted-fg); text-transform:uppercase; }
  .evpill.ok .n { color:var(--success); } .evpill.bad .n { color:var(--destructive); }
  .evpill.warn .n { color:var(--warning); }
  .evempty { font-size:.76rem; color:var(--muted-fg); line-height:1.6; padding:.6rem 0; }
  .kpi { border:1px solid var(--border); border-radius:12px; background:var(--card); padding:1.1rem; min-height:8rem; }
  .kpi .l { font-size:.65rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted-fg); }
  .kpi .v { font-size:1.8rem; font-weight:600; font-variant-numeric:tabular-nums; }
  .kpi .s { font-size:.73rem; color:var(--muted-fg); }
  .idlist { display:flex; flex-wrap:wrap; gap:.3rem; } .idnone { color:var(--muted-fg); font-size:.8rem; }
  @media(min-width:768px) {
    [data-testid="stSidebarResizeHandle"] { display:none; }
  }
  @media(max-width:767.98px) {
    [data-testid="stSidebarCollapseButton"] { opacity:1; }
    [data-testid="stSidebarResizeHandle"] { display:none; }
    .block-container { padding:3.75rem 1rem 2rem; }
    [data-testid="stBottomBlockContainer"] { padding-inline:1rem; }
    .hero { padding:1.8rem 0; } .hero h1 { font-size:2.2rem; }
    .login-hero { padding:1rem 0; } .login-hero h1 { font-size:2.7rem; }
    .st-key-login_card { margin-top:0; padding:1.25rem!important; }
    .feature-list { gap:.6rem; margin-top:1.25rem; }
    [class*="st-key-ex_"] button { min-height:5rem; }
  }
  @media(prefers-reduced-motion:reduce) {
    *, *::before, *::after { animation:none!important; transition:none!important; scroll-behavior:auto!important; }
  }
"""


def _css(theme: str, *, wide: bool = False) -> str:
    layout = ".block-container{max-width:88rem!important;}" if wide else ""
    return f"<style>:root{{{THEMES.get(theme, THEMES['light'])}}}{CSS}{layout}</style>"


def _copy_button_css(theme: str) -> str:
    """Give the isolated copy iframe the shared palette and component styles."""
    return _css(theme)

PAGE_TITLE = "Enterprise Knowledge AI"
PAGE_ICON = str(Path(__file__).parent / "images" / "knowledge-mark.png")
MAX_UI_TURNS = 6
MAX_QUESTION_CHARS = 1000
DEMO_LOGIN_VERSION = 2

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


def _copy_button(text: str) -> None:
    if not text:
        return
    payload = json.dumps(text).replace("</", "<\\/")
    styles = _copy_button_css(st.session_state.get("theme", "light"))
    html = f"""
    {styles}
    <body class="copy-frame">
    <button class="copy-button" type="button" onclick="copyText()">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      <span aria-live="polite">Copy</span>
    </button>
    <script>
      const TXT = {payload};
      const BTN = document.querySelector('.copy-button');
      function feedback(ok){{
        BTN.classList.add(ok ? 'done' : 'error');
        BTN.querySelector('span').textContent = ok ? 'Copied!' : 'Copy failed';
        setTimeout(() => {{
          BTN.classList.remove('done', 'error');
          BTN.querySelector('span').textContent = 'Copy';
        }}, 1600);
      }}
      function legacy(){{
        const ta = document.createElement('textarea');
        ta.value = TXT;
        ta.className = 'copy-buffer';
        document.body.appendChild(ta);
        ta.focus(); ta.select();
        try {{ feedback(document.execCommand('copy')); }}
        catch (e) {{ feedback(false); }}
        document.body.removeChild(ta);
        BTN.focus({{preventScroll:true}});
      }}
      function copyText(){{
        if (navigator.clipboard && window.isSecureContext) {{
          navigator.clipboard.writeText(TXT).then(() => feedback(true), legacy);
        }} else {{ legacy(); }}
      }}
    </script>
    </body>
    """
    st.iframe(html, height=40, width="content")


def _base_url() -> str:
    # The local launcher selects a free API port, overriding deployment secrets.
    if base_url := os.getenv("API_BASE_URL"):
        return base_url
    try:
        return st.secrets.get("API_BASE_URL", api_client.DEFAULT_BASE_URL)
    except Exception:
        return api_client.DEFAULT_BASE_URL


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("demo_access_token", None)
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
        st.session_state.demo_access_token = None
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
    st.session_state.demo_access_token = None
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


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def _demo_users(base_url: str) -> list[dict]:
    """Cache only the public directory, never credentials or login sessions."""
    users = api_client.list_demo_users(base_url)
    if not users:
        raise api_client.ApiError("No selectable identities were returned.")
    return users


def _load_users(base_url: str) -> list[dict] | None:
    try:
        with st.spinner("Connecting to the demo service..."):
            return _demo_users(base_url)
    except api_client.ApiError as exc:
        st.error(str(exc))
        st.caption("The service may be waking up. Try connecting again in a moment.")
        if st.button("Retry connection", key="retry_demo_users"):
            st.rerun()
        return None


@st.fragment
def _service_status(base_url: str) -> None:
    """Run the index diagnostic only on request, independently of chat reruns."""
    st.markdown('<div class="sec">Service status</div>', unsafe_allow_html=True)
    if st.button("Check service", key="check_service"):
        with st.spinner("Checking service..."):
            status = api_client.readiness(base_url, timeout=5)
        st.session_state.service_check = (base_url, status)
    previous = st.session_state.get("service_check")
    if not previous or previous[0] != base_url:
        st.caption("Check the connection and knowledge index when needed.")
        return
    status = previous[1]
    if status["ready"]:
        tone, label = "ok", "Backend and knowledge index ready"
    elif status["reachable"]:
        tone, label = "warn", "Backend online; knowledge index not ready"
    else:
        tone, label = "bad", "Service unreachable; it may be waking up"
    st.markdown(f'<div class="dot-row"><span class="dot dot-{tone}"></span>'
                f'{label}</div>', unsafe_allow_html=True)
    st.caption("Last requested check. Check again to refresh.")
    for problem in status.get("problems", []):
        st.text(str(problem))


def _login_sidebar(base_url: str) -> None:
    with st.sidebar:
        st.markdown(_brand("Permission-aware assistant"),
                    unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sec">How it works</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="ucard"><div class="rl">1. Choose your account</div>'
            '<div class="dp">Select a sample identity; the backend applies its permissions.</div></div>',
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
        _service_status(base_url)

        st.divider()
        _theme_toggle("theme_toggle_login")
        st.markdown(
            '<div class="guard">The backend applies each sample identity’s '
            'permissions after checking its demo password. Use synthetic data only.</div>',
            unsafe_allow_html=True)


def _login_screen(base_url: str) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Permission-aware knowledge assistant</div>'
        '<h1>Knowledge answers, scoped to your access.</h1>'
        '<p>Choose an account to explore the knowledge workspace.</p></div>',
        unsafe_allow_html=True)

    with st.container(horizontal_alignment="center"):
        with st.container(border=True, key="login_card", width=480):
            st.markdown(
                '<div class="login-kicker">Demo identity</div>'
                '<div class="login-title">Explore a sample account</div>'
                '<div class="login-copy">Select a sample identity. The backend '
                'will apply its permissions to your questions.</div>',
                unsafe_allow_html=True)
            users = _load_users(base_url)
            if users is None:
                return
            user_by_id = {user["user_id"]: user for user in users}
            with st.form("demo_login_form", clear_on_submit=False):
                selected_id = st.selectbox(
                    "Demo account", list(user_by_id),
                    format_func=lambda user_id: (
                        f"{user_by_id[user_id]['name']} · "
                        f"{user_by_id[user_id]['role']}"))
                password = st.text_input(
                    "Demo password", type="password",
                    placeholder="firstname@login")
                st.caption("Password example: aditya@login. Use the selected account's first name.")
                submitted = st.form_submit_button(
                    "Continue", type="primary", use_container_width=True)
            with st.expander("Demo IDs and password examples"):
                for user in users:
                    first_name = str(user.get("name") or "").split()
                    password_example = (f"{first_name[0].casefold()}@login"
                                        if first_name else "")
                    st.text(f"{user['user_id']} · {user['name']} · {password_example}")
            if submitted:
                try:
                    session = api_client.demo_login(
                        selected_id, password, base_url=base_url)
                except api_client.ApiError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.user = session["user"]
                    st.session_state.demo_access_token = session["access_token"]
                    st.session_state.seller_choice = 0
                    st.session_state.pop("seller_select", None)
                    _reset_conversation()
                    st.rerun()


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
                     key="seller_select", label_visibility="collapsed",
                     disabled=len(options) == 1)
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
        if st.button("New User", key="switch_user",
                         use_container_width=True):
                _sign_out()
                st.rerun()
        _theme_toggle("theme_toggle_sidebar")

        st.divider()
        _evaluation_sidebar(base_url)

        st.divider()
        _service_status(base_url)

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


@st.cache_data(ttl=60, max_entries=32, show_spinner=False)
def _fetch_eval_summary(base_url: str, report: str | None = None) -> dict:
    return api_client.evaluation_summary(base_url, report=report)


def _eval_summary(base_url: str, report: str | None = None) -> dict | None:
    """Read the latest saved evaluation. Never triggers a run."""
    try:
        return _fetch_eval_summary(base_url, report)
    except api_client.ApiError:
        return None


def _evaluation_sidebar(base_url: str) -> None:
    st.markdown('<div class="sec">Evaluation</div>', unsafe_allow_html=True)
    st.caption("Open saved evaluation results when needed.")

    label = ("Back to chat" if st.session_state.view == "evaluation"
             else "View evaluation matrix")
    if st.button(label, key="eval_toggle", use_container_width=True):
        st.session_state.view = ("chat"
                                 if st.session_state.view == "evaluation"
                                 else "evaluation")
        st.rerun()


def _matrix_frame(matrix: list[dict]) -> pd.DataFrame:
    import pandas as pd

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
            _fetch_eval_summary.clear(base_url, selected_report)
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
                   seller_filter: list[str] | None, history: list[dict],
                   access_token: str | None = None) -> dict:
    events = api_client.ask_demo_stream(access_token or "", question,
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
                        _copy_button(message["content"])
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
            result = _stream_answer(
                user, base_url, question, seller_filter, history,
                st.session_state.demo_access_token)
        except api_client.ApiError as exc:
            stage.update(label="Service error", state="error")
            st.error(str(exc))
            if exc.status == 401:
                st.session_state.user = None
                st.session_state.demo_access_token = None
                _reset_conversation()
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
            _copy_button(result["answer"])
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
    # Streamlit keeps desktop navigation fixed and owns the mobile drawer.
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON,
                       layout="wide" if evaluation_view else "centered",
                       initial_sidebar_state="locked")
    _init_state()
    st.markdown(_css(st.session_state.theme, wide=evaluation_view),
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

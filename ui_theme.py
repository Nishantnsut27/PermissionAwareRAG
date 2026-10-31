"""Presentation tokens and local line icons for the Streamlit workspace."""
from __future__ import annotations

import html


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
        --accent-soft:#f3e5d9; --sidebar:#efeee7; --shadow:0 3px 14px #393a2810;
        color-scheme:light;
    """,
    "dark": """
        --bg:#1c211f; --fg:#edece4; --card:#252c28; --primary:#edb498;
        --primary-fg:#26201c; --secondary:#303a33; --muted-fg:#b0b9ae;
        --border:#404a42; --destructive:#f0a2a2; --success:#9ccaad;
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
  [data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--border); }
  [data-testid="stSidebarUserContent"] { padding:1.7rem 1.25rem; }
  [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="stSidebarCollapseButton"] button { color:var(--fg)!important; }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color:var(--muted-fg)!important; font-size:.78rem; line-height:1.6; }
  .stButton > button, .stFormSubmitButton > button {
    background:var(--card); color:var(--fg); border:1px solid var(--border);
    border-radius:10px; min-height:2.8rem; padding:.6rem 1rem;
    font-size:.88rem; font-weight:600; box-shadow:none;
    transition:background .16s,border-color .16s,box-shadow .16s; }
  .stButton > button p, .stFormSubmitButton > button p { color:inherit; }
  .stButton > button:hover, .stFormSubmitButton > button:hover {
    background:var(--accent-soft); color:var(--primary); border-color:var(--primary); }
  .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background:var(--primary); color:var(--primary-fg); border-color:var(--primary); }
  .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    filter:brightness(.94); }
  button:focus-visible, [data-baseweb="select"]:focus-within {
    outline:2px solid var(--primary)!important; outline-offset:3px; border-radius:10px; }
  [data-testid="stTextInput"]:focus-within { outline:0!important; }
  [data-testid="stSelectbox"] [data-baseweb="select"] > div,
  [data-testid="stTextInput"] [data-baseweb="input"] {
    background:var(--card)!important; color:var(--fg)!important;
    border-color:var(--border)!important; border-radius:9px; }
  [data-testid="stSelectbox"] input, [data-testid="stTextInput"] input {
    color:var(--fg)!important; caret-color:var(--primary); }
  [data-testid="stTextInput"] input::placeholder { color:var(--muted-fg)!important; }
  [data-testid="stTextInput"] [data-baseweb="input"] {
    min-height:3rem; }
  [data-testid="stSelectbox"] svg, [data-testid="stTextInput"] button { color:var(--muted-fg)!important; }
  [role="listbox"], [role="option"], [data-baseweb="popover"] > div {
    background:var(--card)!important; color:var(--fg)!important; }
  [role="option"][aria-selected="true"] { background:var(--accent-soft)!important; }
  [data-testid="stSlider"] { color:var(--fg); }
  [data-testid="stSliderTickBar"] { color:var(--muted-fg)!important; }
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
    border-radius:18px; padding:1.8rem!important; box-shadow:var(--shadow); margin-top:1.7rem; }
  .st-key-login_card [data-testid="stForm"] { border:0; padding:0; }
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
  .idbox { display:flex; gap:.65rem; align-items:center; border:1px solid var(--border);
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
  [class*="st-key-theme_toggle"] label > div:first-child { background:var(--primary)!important; }
  .guard { display:flex; gap:.5rem; color:var(--muted-fg); font-size:.72rem; line-height:1.65; padding:.75rem 0; }
  .guard .line-icon { color:var(--primary); margin-top:.15rem; }
  .dot-row { display:flex; align-items:center; gap:.4rem; font-size:.73rem; color:var(--muted-fg); }
  .dot { width:6px; height:6px; border-radius:50%; display:inline-block; }
  .dot-ok { background:var(--success); } .dot-bad { background:var(--destructive); }
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
  [data-testid="stChatInput"] textarea { color:var(--fg)!important; background:transparent!important; caret-color:var(--primary); }
  [data-testid="stChatInput"] textarea::placeholder { color:var(--muted-fg)!important; }
  [data-testid="stChatInputSubmitButton"] { background:var(--primary)!important; color:var(--primary-fg)!important; border-radius:10px; }
  [data-testid="stChatInputSubmitButton"]:disabled { opacity:.4; }
  [data-testid="stExpander"] { border:1px solid var(--border); border-radius:10px; background:var(--card); }
  [data-testid="stExpander"] summary { color:var(--fg)!important; }
  [data-testid="stExpander"] summary p { font-size:.83rem; }
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
  .evpill.warn .n { color:var(--primary); }
  .evempty { font-size:.76rem; color:var(--muted-fg); line-height:1.6; padding:.6rem 0; }
  .kpi { border:1px solid var(--border); border-radius:12px; background:var(--card); padding:1.1rem; min-height:8rem; }
  .kpi .l { font-size:.65rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted-fg); }
  .kpi .v { font-size:1.8rem; font-weight:600; font-variant-numeric:tabular-nums; }
  .kpi .s { font-size:.73rem; color:var(--muted-fg); }
  .idlist { display:flex; flex-wrap:wrap; gap:.3rem; } .idnone { color:var(--muted-fg); font-size:.8rem; }
  @media(max-width:760px) {
    .block-container { padding:1.5rem 1rem 2rem; }
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


def css(theme: str) -> str:
    return f"<style>:root{{{THEMES.get(theme, THEMES['light'])}}}{CSS}</style>"

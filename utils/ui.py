"""UI helpers for Streamlit pages (theme, header, style).

This module centralizes cosmetic utilities so each page stays lean.
"""
from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st


THEME_KEY = "app_theme"  # retained for backward compatibility with pages referencing it

LIGHT_PALETTE = {
    "bg": "#ffffff",
    "text": "#111827",
    "primary": "#2563eb",
    "secondary": "#f3f4f6",
    "border": "#e5e7eb",
    "code_bg": "#f9fafb",
}


def get_palette():
    """Return the (now fixed) light palette.

    Dark mode removed per requirement; keep function for existing imports.
    """
    # Ensure legacy key exists (some pages may still read it for display logic)
    st.session_state[THEME_KEY] = 'light'
    return LIGHT_PALETTE


def inject_global_css():
    """Inject base CSS according to current theme.

    Keeps it idempotent: each rerun rewrites the style block.
    """
    p = get_palette()
    css = f"""
    <style id='__global_theme'>
    html, body, [data-testid=stAppViewContainer] {{
        background: {p['bg']} !important;
        color: {p['text']} !important;
    }}
    .stButton button, .stDownloadButton button {{
        background: linear-gradient(90deg,{p['primary']} 0%, {p['primary']}cc 100%) !important;
        color: #fff !important;
        border: 1px solid {p['primary']} !important;
        border-radius: 6px !important;
        font-weight: 500;
        transition: all .15s ease;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        filter: brightness(1.08);
        box-shadow: 0 2px 6px rgba(0,0,0,.18);
    }}
    table {{ border-collapse: collapse; }}
    thead tr {{ background:{p['secondary']}; }}
    tbody tr:nth-child(odd) {{ background: {p['secondary']}55; }}
    code, pre {{ background: {p['code_bg']} !important; border:1px solid {p['border']}33; }}
    /* Logo positioning */
    .logo {{ position:absolute; top:0; left:600px; width:90px; height:90px; }}
    .app-header-bar {{ display:flex; gap:.75rem; align-items:center; margin-bottom:.75rem; }}
    .app-header-spacer {{ flex:1; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header(show_logo_checkbox: bool = True, show_theme_toggle: bool = False, title: str | None = None):
    """Render a shared header section.

    Parameters
    ----------
    show_logo_checkbox: bool
        Whether to include the checkbox to toggle logo visibility.
    show_theme_toggle: bool
        Whether to show light/dark theme radio buttons.
    title: str | None
        Optional page title (if not using st.title elsewhere).
    """
    # Session defaults
    if 'show_logo' not in st.session_state:
        st.session_state.show_logo = True
    # Force light theme only
    st.session_state[THEME_KEY] = 'light'

    with st.sidebar:
        if show_logo_checkbox:
            st.session_state.show_logo = st.checkbox('Show Logo', value=st.session_state.show_logo)
        # Theme toggle removed (fixed light mode)

    inject_global_css()

    # Logo
    if st.session_state.show_logo:
        logo_path = Path('./assets/logo.jpg')
        if logo_path.exists():
            img_base64 = base64.b64encode(logo_path.read_bytes()).decode()
            st.markdown(f"<img class='logo' src='data:image/png;base64,{img_base64}' alt='Logo' />", unsafe_allow_html=True)

    if title:
        # Removed theme badge (only one mode now)
        st.markdown(f"<div class='app-header-bar'><h1 style='margin:0;'>{title}</h1><div class='app-header-spacer'></div></div>", unsafe_allow_html=True)


__all__ = [
    'render_header',
    'inject_global_css',
    'get_palette'
]

"""Streamlit entry point:  streamlit run app/main.py"""

import logging
import sys
from pathlib import Path

# `streamlit run` puts app/ (not the project root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.ui import chat, connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Local Text-to-SQL", page_icon="🗄️", layout="wide")

if "session" in st.session_state:
    chat.render()
else:
    connection.render()

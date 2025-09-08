import streamlit as st
import subprocess
import sys
import os
from streamlit_option_menu import streamlit_option_menu
import importlib.util
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import traceback
import numpy as np

# Set page config
st.set_page_config(
    page_title="AChE Prediction Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    try:
        with open("style.css") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styling.")

# Load and run an app module
def load_and_run_app(app_file):
    """Dynamically load an app module"""
    try:
        spec = importlib.util.spec_from_file_location("app_module", app_file)
        app_module = importlib.util.module_from_spec(spec)
        
        # Create a clean namespace for the app
        app_globals = {
            '__name__': '__main__',
            '__file__': app_file,
            'st': st,
            'pd': pd,
            'np': np,
            'os': os,
            'sys': sys
        }
        
        # Read and execute the app file
        with open(app_file, 'r') as f:
            code = f.read()
        
        exec(code, app_globals)
        return True
        
    except Exception as e:
        st.error(f"Error loading {app_file}: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_chemberta_app():
    load_and_run_app("app_chemberta.py")

def run_rdkit_app():
    load_and_run_app("app_rdkit.py")

def run_circular_app():
    load_and_run_app("app_circular.py")

def run_graph_app():
    load_and_run_app("app_graph_combined.py")

def main():
    # Load CSS
    load_css()
    
    # Compact header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.8), rgba(88, 86, 214, 0.8));
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 122, 255, 0.2);
    ">
        <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">🧬 AChE Prediction Suite</div>
        <div style="font-size: 1rem; opacity: 0.9;">AI-powered molecular activity prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact navigation
    selected = streamlit_option_menu(
        menu_title=None,
        options=["🧬 ChemBERTa", "💊 RDKit", "🔄 Circular", "📊 Graph"],
        icons=["cpu", "flask", "arrow-repeat", "diagram-3"],
        menu_icon=None,
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0", "background-color": "transparent", "margin-bottom": "1rem"},
            "icon": {"color": "#007AFF", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "padding": "8px 16px",
                "background-color": "rgba(0, 122, 255, 0.1)",
                "color": "#007AFF",
                "border-radius": "12px",
                "margin": "0 4px",
                "backdrop-filter": "blur(10px)",
                "border": "1px solid rgba(0, 122, 255, 0.2)",
                "transition": "all 0.2s ease"
            },
            "nav-link-selected": {
                "background-color": "#007AFF",
                "color": "white",
                "font-weight": "600",
                "box-shadow": "0 4px 12px rgba(0, 122, 255, 0.3)"
            },
        }
    )
    
    # App routing
    if selected == "🧬 ChemBERTa":
        run_chemberta_app()
    elif selected == "💊 RDKit":
        run_rdkit_app()
    elif selected == "🔄 Circular":
        run_circular_app()
    elif selected == "📊 Graph":
        run_graph_app()

if __name__ == "__main__":
    main()

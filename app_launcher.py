import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Set page config as the very first command
st.set_page_config(
    page_title="Molecular Prediction Suite",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced iOS minimalistic interface with beautiful styling
st.markdown("""
<style>
/* Import Inter font for authentic iOS look */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main > div {
    padding-top: 0;
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    min-height: 100vh;
}

/* Navigation header with iOS glass effect */
.nav-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 20px 24px;
    border-radius: 0 0 20px 20px;
    margin: -1rem -1rem 2rem -1rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.nav-title {
    color: #34C759;
    font-size: 28px;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #34C759 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.nav-subtitle {
    color: #86868B;
    font-size: 16px;
    font-weight: 400;
    margin-top: 8px;
    margin-bottom: 0;
}

/* Enhanced app cards with iOS depth */
.app-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #FAFBFC 100%);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(0, 0, 0, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.app-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 50%, #34C759 100%);
}

.app-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
}

.app-card-icon {
    font-size: 32px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.app-card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
    color: #1D1D1F;
    letter-spacing: -0.01em;
}

.app-card-description {
    font-size: 15px;
    color: #515154;
    line-height: 1.5;
    margin-bottom: 16px;
}

.app-card-features {
    list-style: none;
    padding: 0;
    margin: 0 0 20px 0;
}

.app-card-features li {
    font-size: 14px;
    color: #86868B;
    margin-bottom: 6px;
    padding-left: 20px;
    position: relative;
}

.app-card-features li:before {
    content: "•";
    color: #007AFF;
    font-weight: 600;
    position: absolute;
    left: 0;
}

/* Enhanced button styling */
.stButton > button {
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    box-shadow: 0 4px 16px rgba(0, 122, 255, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0056CC 0%, #4C46A8 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 122, 255, 0.4);
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, #F2F2F7 0%, #E5E5EA 100%);
    color: #007AFF;
    border: 1px solid #D1D1D6;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #E5E5EA 0%, #D1D1D6 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

/* iOS-style metric cards */
.ios-card {
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 16px;
    border: none;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    position: relative;
    overflow: hidden;
}

.ios-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.05) 100%);
    pointer-events: none;
}

.card-value {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 4px;
    position: relative;
    z-index: 1;
}

.card-label {
    font-size: 13px;
    opacity: 0.95;
    font-weight: 500;
    position: relative;
    z-index: 1;
}

/* Enhanced input styling */
.stTextInput > div > div > input {
    border-radius: 12px;
    border: 2px solid #E5E5EA;
    font-size: 16px;
    padding: 12px 16px;
    transition: all 0.3s ease;
    background: #FFFFFF;
}

.stTextInput > div > div > input:focus {
    border-color: #007AFF;
    box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
}

/* Beautiful progress bars */
.stProgress > div > div {
    background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
    border-radius: 8px;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_app' not in st.session_state:
    st.session_state.selected_app = 'home'

def create_app_cards():
    """Create beautiful iOS-style app cards"""
    
    apps = [
        {
            "name": "ChemBERTa Transformer",
            "icon": "🧪",
            "description": "State-of-the-art transformer model with attention visualization for molecular property prediction.",
            "features": ["Attention weight analysis", "Interactive drawing", "Batch processing", "SMILES & SDF support"],
            "key": "chemberta"
        },
        {
            "name": "RDKit Descriptors",
            "icon": "⚛️",
            "description": "AutoML TPOT-based molecular property prediction using RDKit descriptors and automated machine learning.",
            "features": ["AutoML TPOT pipeline", "RDKit descriptors", "Automated feature selection", "Optimized ML models"],
            "key": "rdkit"
        },
        {
            "name": "Circular Fingerprints",
            "icon": "🔄",
            "description": "Ensemble activity prediction using circular fingerprints with multiple machine learning algorithms.",
            "features": ["Circular fingerprints", "Ensemble modeling", "Activity prediction", "ECFP features"],
            "key": "circular"
        },
        {
            "name": "Graph Neural Networks",
            "icon": "🕸️",
            "description": "Graph convolutional network-based activity prediction for molecular graph analysis.",
            "features": ["Graph convolution", "Activity prediction", "Molecular graphs", "Deep learning"],
            "key": "graph"
        }
    ]
    
    # Create responsive grid
    for i in range(0, len(apps), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j < len(apps):
                app = apps[i + j]
                with col:
                    st.markdown(f"""
                    <div class="app-card">
                        <div class="app-card-icon">{app['icon']}</div>
                        <div class="app-card-title">{app['name']}</div>
                        <div class="app-card-description">{app['description']}</div>
                        <ul class="app-card-features">
                            {''.join([f"<li>{feature}</li>" for feature in app['features']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Launch {app['name']}", key=f"btn_{app['key']}", use_container_width=True):
                        st.session_state.selected_app = app['key']
                        st.rerun()

def load_chemberta_app():
    """Load and run ChemBERTa app with full functionality"""
    try:
        # Import all necessary modules that ChemBERTa app needs
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Draw
        import traceback
        import numpy as np
        from simpletransformers.classification import ClassificationModel
        import streamlit.components.v1 as components
        from streamlit_ketcher import st_ketcher
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import seaborn as sns
        from transformers import AutoTokenizer, AutoModel
        import re
        from PIL import Image
        import io
        
        # Make these available globally for the exec
        globals().update({
            'pd': pd,
            'Chem': Chem,
            'Draw': Draw,
            'traceback': traceback,
            'np': np,
            'ClassificationModel': ClassificationModel,
            'components': components,
            'st_ketcher': st_ketcher,
            'torch': torch,
            'F': F,
            'plt': plt,
            'sns': sns,
            'AutoTokenizer': AutoTokenizer,
            'AutoModel': AutoModel,
            're': re,
            'Image': Image,
            'io': io
        })
        
        # Read the ChemBERTa app file and exclude the page config
        with open('app_chemberta.py', 'r') as f:
            content = f.read()
        
        # Remove the set_page_config call to avoid conflict
        lines = content.split('\n')
        filtered_lines = []
        skip_block = False
        
        for line in lines:
            if 'st.set_page_config(' in line:
                skip_block = True
                continue
            elif skip_block and ')' in line and not line.strip().startswith('#'):
                skip_block = False
                continue
            elif not skip_block:
                filtered_lines.append(line)
        
        # Execute the modified content
        exec('\n'.join(filtered_lines), globals())
        
    except Exception as e:
        st.error(f"Error loading ChemBERTa app: {e}")
        st.info("Make sure app_chemberta.py is in the same directory and all dependencies are installed.")

def load_rdkit_app():
    """Load and run RDKit app with full functionality"""
    try:
        # Import all necessary modules that RDKit app needs
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Draw, Descriptors
        import pickle
        import numpy as np
        import streamlit.components.v1 as components
        
        # Make these available globally for the exec
        globals().update({
            'pd': pd,
            'Chem': Chem,
            'Draw': Draw,
            'Descriptors': Descriptors,
            'pickle': pickle,
            'np': np,
            'components': components
        })
        
        # Read the RDKit app file and exclude the page config
        with open('app_rdkit.py', 'r') as f:
            content = f.read()
        
        # Remove the set_page_config call to avoid conflict
        lines = content.split('\n')
        filtered_lines = []
        skip_block = False
        
        for line in lines:
            if 'st.set_page_config(' in line:
                skip_block = True
                continue
            elif skip_block and ')' in line and not line.strip().startswith('#'):
                skip_block = False
                continue
            elif not skip_block:
                filtered_lines.append(line)
        
        # Execute the modified content
        exec('\n'.join(filtered_lines), globals())
        
    except Exception as e:
        st.error(f"Error loading RDKit app: {e}")
        st.info("Make sure app_rdkit.py is in the same directory and all dependencies are installed.")

def load_circular_app():
    """Load and run Circular Fingerprints app with full functionality"""
    try:
        # Import all necessary modules that Circular app needs
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Draw
        import numpy as np
        import streamlit.components.v1 as components
        
        # Make these available globally for the exec
        globals().update({
            'pd': pd,
            'Chem': Chem,
            'Draw': Draw,
            'np': np,
            'components': components
        })
        
        # Read the Circular app file and exclude the page config
        with open('app_circular.py', 'r') as f:
            content = f.read()
        
        # Remove the set_page_config call to avoid conflict
        lines = content.split('\n')
        filtered_lines = []
        skip_block = False
        
        for line in lines:
            if 'st.set_page_config(' in line:
                skip_block = True
                continue
            elif skip_block and ')' in line and not line.strip().startswith('#'):
                skip_block = False
                continue
            elif not skip_block:
                filtered_lines.append(line)
        
        # Execute the modified content
        exec('\n'.join(filtered_lines), globals())
        
    except Exception as e:
        st.error(f"Error loading Circular Fingerprints app: {e}")
        st.info("Make sure app_circular.py is in the same directory and all dependencies are installed.")

def load_graph_app():
    """Load and run Graph Neural Networks app with full functionality"""
    try:
        # Import all necessary modules that Graph app needs
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import Draw
        import numpy as np
        import streamlit.components.v1 as components
        
        # Make these available globally for the exec
        globals().update({
            'pd': pd,
            'Chem': Chem,
            'Draw': Draw,
            'np': np,
            'components': components
        })
        
        # Read the Graph app file and exclude the page config
        with open('app_graph_combined.py', 'r') as f:
            content = f.read()
        
        # Remove the set_page_config call to avoid conflict
        lines = content.split('\n')
        filtered_lines = []
        skip_block = False
        
        for line in lines:
            if 'st.set_page_config(' in line:
                skip_block = True
                continue
            elif skip_block and ')' in line and not line.strip().startswith('#'):
                skip_block = False
                continue
            elif not skip_block:
                filtered_lines.append(line)
        
        # Execute the modified content
        exec('\n'.join(filtered_lines), globals())
        
    except Exception as e:
        st.error(f"Error loading Graph Neural Networks app: {e}")
        st.info("Make sure app_graph_combined.py is in the same directory and all dependencies are installed.")

def run_selected_app():
    """Load and run the selected app directly in the interface"""
    app_names = {
        'chemberta': '🧪 ChemBERTa Transformer',
        'rdkit': '⚛️ RDKit Descriptors',
        'circular': '🔄 Circular Fingerprints', 
        'graph': '🕸️ Graph Neural Networks'
    }
    
    selected_name = app_names.get(st.session_state.selected_app)
    
    # Navigation back to home at top left corner
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home", type="secondary", use_container_width=True):
            st.session_state.selected_app = 'home'
            st.rerun()
    
    # Load the appropriate app directly
    if st.session_state.selected_app == 'chemberta':
        load_chemberta_app()
    elif st.session_state.selected_app == 'rdkit':
        load_rdkit_app()
    elif st.session_state.selected_app == 'circular':
        load_circular_app()
    elif st.session_state.selected_app == 'graph':
        load_graph_app()

def main():
    """Main application logic"""
    
    if st.session_state.selected_app == 'home':
        # Beautiful app selection cards
        create_app_cards()
        
        # Footer with enhanced styling
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #86868B; font-size: 14px; margin-top: 2rem; 
                    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%); border-radius: 16px; margin: 2rem 0;">
            <p style="margin: 0;"><strong>© 2025 AChE Inhibitor Prediction Suite</strong></p>
            <p style="margin: 4px 0 0 0;">Powered by Streamlit, RDKit & Advanced Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Load and run the selected app directly
        run_selected_app()

if __name__ == "__main__":
    main()

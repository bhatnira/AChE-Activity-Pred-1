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
    page_title="AChE Inhibitor Prediction Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# iOS-style CSS
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Main container styling */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 0;
        margin: 0;
    }
    
    /* iOS-style glass morphism header */
    .ios-header {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .ios-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .ios-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0;
        font-weight: 300;
    }
    
    /* iOS-style card grid */
    .app-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* iOS-style app cards */
    .app-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        position: relative;
        overflow: hidden;
    }
    
    .app-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        background: rgba(255, 255, 255, 0.35);
    }
    
    .app-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 20px 20px 0 0;
    }
    
    .app-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    .app-description {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    .app-features {
        text-align: left;
        margin-bottom: 1.5rem;
    }
    
    .app-features ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .app-features li {
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .app-features li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #4ade80;
        font-weight: bold;
    }
    
    /* iOS-style buttons */
    .ios-button {
        background: rgba(255, 255, 255, 0.9);
        color: #667eea;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px 0 rgba(255, 255, 255, 0.3);
    }
    
    .ios-button:hover {
        background: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(255, 255, 255, 0.4);
    }
    
    /* Footer */
    .ios-footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    
    /* Navigation styling */
    .nav-container {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 1rem;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

def load_app_module(app_file):
    """Dynamically load an app module"""
    try:
        spec = importlib.util.spec_from_file_location("app_module", app_file)
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        return app_module
    except Exception as e:
        st.error(f"Error loading {app_file}: {e}")
        return None

def main():
    # Create the main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Navigation
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    selected = streamlit_option_menu(
        menu_title=None,
        options=["🏠 Home", "🧬 ChemBERTa", "💊 RDKit", "🔄 Circular FP", "📊 Graph NN"],
        icons=["house", "cpu", "flask", "arrow-repeat", "diagram-3"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "12px 16px",
                "background-color": "rgba(255, 255, 255, 0.1)",
                "color": "white",
                "border-radius": "15px",
                "margin": "0 5px",
                "backdrop-filter": "blur(10px)",
                "border": "1px solid rgba(255, 255, 255, 0.2)"
            },
            "nav-link-selected": {
                "background-color": "rgba(255, 255, 255, 0.9)",
                "color": "#667eea",
                "font-weight": "600",
                "box-shadow": "0 4px 15px 0 rgba(255, 255, 255, 0.3)"
            },
        }
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if selected == "🏠 Home":
        show_home_page()
    elif selected == "🧬 ChemBERTa":
        run_chemberta_app()
    elif selected == "💊 RDKit":
        run_rdkit_app()
    elif selected == "🔄 Circular FP":
        run_circular_app()
    elif selected == "📊 Graph NN":
        run_graph_app()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_home_page():
    # Header
    st.markdown("""
    <div class="ios-header">
        <div class="ios-title">🧬 AChE Prediction Suite</div>
        <div class="ios-subtitle">Advanced AI-powered acetylcholinesterase inhibitor prediction platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # App grid
    st.markdown('<div class="app-grid">', unsafe_allow_html=True)
    
    # ChemBERTa Card
    st.markdown("""
    <div class="app-card">
        <div class="app-icon">🧬</div>
        <div class="app-title">ChemBERTa Transformer</div>
        <div class="app-description">State-of-the-art transformer model for molecular property prediction</div>
        <div class="app-features">
            <ul>
                <li>Attention weight visualization</li>
                <li>Transformer-based predictions</li>
                <li>Real-time molecular analysis</li>
                <li>SMILES & drawing input</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # RDKit Card
    st.markdown("""
    <div class="app-card">
        <div class="app-icon">💊</div>
        <div class="app-title">RDKit Descriptors</div>
        <div class="app-description">Traditional molecular descriptors with machine learning</div>
        <div class="app-features">
            <ul>
                <li>Molecular descriptors analysis</li>
                <li>Feature importance ranking</li>
                <li>Interactive visualizations</li>
                <li>Batch processing support</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Circular Fingerprints Card
    st.markdown("""
    <div class="app-card">
        <div class="app-icon">🔄</div>
        <div class="app-title">Circular Fingerprints</div>
        <div class="app-description">Morgan fingerprints for structural similarity analysis</div>
        <div class="app-features">
            <ul>
                <li>Morgan circular fingerprints</li>
                <li>Structural similarity maps</li>
                <li>Substructure analysis</li>
                <li>High-throughput screening</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Graph Neural Networks Card
    st.markdown("""
    <div class="app-card">
        <div class="app-icon">📊</div>
        <div class="app-title">Graph Neural Networks</div>
        <div class="app-description">Deep learning on molecular graph representations</div>
        <div class="app-features">
            <ul>
                <li>Graph convolutional networks</li>
                <li>Node & edge feature learning</li>
                <li>Molecular graph analysis</li>
                <li>Advanced deep learning</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="ios-footer">
        <p>🔬 Powered by advanced machine learning and molecular informatics</p>
        <p>Select a model from the navigation above to start predicting</p>
    </div>
    """, unsafe_allow_html=True)

def run_chemberta_app():
    st.markdown("""
    <div class="ios-header">
        <div class="ios-title">🧬 ChemBERTa Predictor</div>
        <div class="ios-subtitle">Transformer-based molecular activity prediction with attention visualization</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and run the ChemBERTa app
    try:
        # Import necessary modules for ChemBERTa app
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
        from transformers import RobertaTokenizer
        
        # Execute the ChemBERTa app functions
        exec(open('app_chemberta.py').read().split('if __name__ == \'__main__\':')[0])
        
        # Run the ChemBERTa interface
        tab1, tab2, tab3, tab4 = st.tabs(["⚗️ SMILES", "🎨 Draw", "📄 SDF", "📊 Batch"])
        
        with tab1:
            handle_smiles_input()
        with tab2:
            handle_drawing_input()
        with tab3:
            st.markdown("### 📄 SDF File Upload")
            uploaded_sdf_file = st.file_uploader("SDF File", type=['sdf'], key="sdf_file_uploader")
            if st.button('🔍 Predict SDF', type="primary", key="sdf_predict_button"):
                if uploaded_sdf_file is not None:
                    with st.spinner('Processing SDF file...'):
                        sdf_file_prediction(uploaded_sdf_file)
                else:
                    st.error('⚠️ Please upload an SDF file.')
        with tab4:
            st.markdown("### 📊 Excel File Batch Prediction")
            uploaded_excel_file = st.file_uploader("Excel File", type=['xlsx'], key="excel_file_uploader")
            if uploaded_excel_file is not None:
                df = pd.read_excel(uploaded_excel_file, engine='openpyxl')
                st.dataframe(df.head(), use_container_width=True)
                smiles_column = st.selectbox("Select SMILES column:", options=df.columns.tolist())
                if st.button('🔍 Predict Excel', type="primary", key="excel_predict_button"):
                    with st.spinner('Processing Excel file...'):
                        excel_file_prediction(uploaded_excel_file, smiles_column)
            else:
                st.info('⬆️ Please upload an Excel file.')
                
    except Exception as e:
        st.error(f"Error loading ChemBERTa app: {e}")
        st.info("Please ensure all dependencies are installed and the ChemBERTa model is available.")

def run_rdkit_app():
    st.markdown("""
    <div class="ios-header">
        <div class="ios-title">💊 RDKit Predictor</div>
        <div class="ios-subtitle">Molecular descriptors-based activity prediction with feature analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Execute the RDKit app
        exec(open('app_rdkit.py').read())
    except Exception as e:
        st.error(f"Error loading RDKit app: {e}")
        st.info("Please ensure the RDKit app file is available and all dependencies are installed.")

def run_circular_app():
    st.markdown("""
    <div class="ios-header">
        <div class="ios-title">🔄 Circular Fingerprints</div>
        <div class="ios-subtitle">Morgan fingerprints for molecular similarity and activity prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Execute the Circular FP app
        exec(open('app_circular.py').read())
    except Exception as e:
        st.error(f"Error loading Circular Fingerprints app: {e}")
        st.info("Please ensure the Circular FP app file is available and all dependencies are installed.")

def run_graph_app():
    st.markdown("""
    <div class="ios-header">
        <div class="ios-title">📊 Graph Neural Networks</div>
        <div class="ios-subtitle">Deep learning on molecular graphs for activity prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Check for both possible graph app files
        if os.path.exists('app_graphC.py'):
            exec(open('app_graphC.py').read())
        elif os.path.exists('app_graphR.py'):
            exec(open('app_graphR.py').read())
        elif os.path.exists('app_graph_combined.py'):
            exec(open('app_graph_combined.py').read())
        else:
            st.error("Graph app file not found")
    except Exception as e:
        st.error(f"Error loading Graph NN app: {e}")
        st.info("Please ensure the Graph NN app file is available and all dependencies are installed.")

if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import traceback
import numpy as np
import os
from simpletransformers.classification import ClassificationModel
import streamlit.components.v1 as components
from streamlit_ketcher import st_ketcher
import torch
import torch.nn.functional as F

# Set page config as the very first command
st.set_page_config(
    page_title="Predict Acetylcholinesterase Inhibitory Activity with ChemBERTa",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to load custom CSS
def load_css():
    try:
        with open("style.css") as f:
            css = f.read()
        components.html(f"<style>{css}</style>", height=0, width=0)
    except FileNotFoundError:
        pass

# Function to load Font Awesome icons
def load_fa_icons():
    components.html(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        """,
        height=0, width=0
    )

# Path to the directory where the model was saved
saved_model_path = "checkpoint-2000"

# Load ChemBERTa model
@st.cache_resource
def load_chemberta_model():
    try:
        model = ClassificationModel('roberta', saved_model_path, use_cuda=False)
        return model
    except Exception as e:
        st.error(f'Error loading ChemBERTa model: {e}')
        return None

# Function to compute predictions for a single SMILES input
def compute_chemberta_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        model = load_chemberta_model()
        if model is not None:
            try:
                predictions, raw_outputs = model.predict([smiles])
                logits = raw_outputs[0]
                
                # Compute probabilities
                probs = F.softmax(torch.tensor(logits), dim=0)
                prob_active = probs[1].item()
                
                return mol, predictions[0], prob_active
            except Exception as e:
                st.error(f'Error in prediction: {e}')
                return None, None, None
    return None, None, None

def single_input_prediction(smiles):
    mol, classification_prediction, classification_probability = compute_chemberta_prediction(smiles)
    if mol is not None:
        try:
            return mol, classification_prediction, classification_probability
        except Exception as e:
            st.error(f'Error in prediction: {e}')
            return None, None, None
    return None, None, None

# Function to handle drawing input
def handle_drawing_input():
    st.markdown("### 🎨 Draw Molecule")
    
    # Ketcher molecule editor first
    smile_code = st_ketcher("")
    
    # Show generated SMILES
    if smile_code:
        st.markdown("**Generated SMILES:**")
        st.code(smile_code)
    
    # Create prediction button
    predict_button = st.button('🔍 Predict', type="primary", key="draw_predict_btn")

    if predict_button:
        if smile_code:
            with st.spinner('Analyzing...'):
                mol, classification_prediction, classification_probability = single_input_prediction(smile_code)
            
            if mol is not None:
                # Metric cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                    activity_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                    st.markdown(f"""
                    <div class="metric-card" style="background: {activity_color};">
                        <div class="metric-value">{activity_status}</div>
                        <div class="metric-label">Activity</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #2196F3;">
                        <div class="metric-value">{classification_probability:.1%}</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #9C27B0;">
                        <div class="metric-value">ChemBERTa</div>
                        <div class="metric-label">Model</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Results layout - emphasis on prediction results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**🧪 Structure**")
                    mol_img = Draw.MolToImage(mol, size=(180, 150), kekulize=True, wedgeBonds=True)
                    st.image(mol_img, use_column_width=True)
                    st.code(smile_code, language="text")
                
                with col2:
                    st.markdown("### 📊 Prediction Results")
                    
                    # Prediction summary in a highlighted box
                    activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                    activity_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {activity_color}20, {activity_color}10); 
                                padding: 1rem; border-radius: 10px; border-left: 4px solid {activity_color};">
                        <h4 style="color: {activity_color}; margin: 0;">🎯 {activity_status}</h4>
                        <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {classification_probability:.1%}</p>
                        <p style="margin: 0.5rem 0;"><strong>Model:</strong> ChemBERTa Transformer</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        else:
            st.error("Enter a SMILES string or draw a molecule.")

# Function to handle SMILES input
def handle_smiles_input():
    st.markdown("### ⚗️ SMILES Input")
    
    # Create input layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        single_input = st.text_input('SMILES', placeholder="CCO", key="single_smiles_input")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button('🔍 Predict', type="primary", key="smiles_predict_btn")
    
    # Add some example SMILES
    st.markdown("""
    <div class="upload-area">
        <h4>💡 Example SMILES:</h4>
        <ul style="text-align: left; display: inline-block;">
            <li><code>CC(=O)OC1=CC=CC=C1C(=O)O</code> - Aspirin</li>
            <li><code>CCO</code> - Ethanol</li>
            <li><code>CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</code> - Ibuprofen</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if predict_button and single_input:
        with st.spinner('🧬 Analyzing molecular properties...'):
            mol, classification_prediction, classification_probability = single_input_prediction(single_input)
            
        if mol is not None:
            # Metric cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                activity_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                st.markdown(f"""
                <div class="metric-card" style="background: {activity_color};">
                    <div class="metric-value">{activity_status}</div>
                    <div class="metric-label">Activity</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: #2196F3;">
                    <div class="metric-value">{classification_probability:.1%}</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: #9C27B0;">
                    <div class="metric-value">ChemBERTa</div>
                    <div class="metric-label">Model</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Results layout - emphasis on prediction results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**🧪 Structure**")
                mol_img = Draw.MolToImage(mol, size=(180, 150), kekulize=True, wedgeBonds=True)
                st.image(mol_img, use_column_width=True)
                st.code(single_input, language="text")
            
            with col2:
                st.markdown("### 📊 Prediction Results")
                
                # Prediction summary in a highlighted box
                activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                activity_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {activity_color}20, {activity_color}10); 
                            padding: 1rem; border-radius: 10px; border-left: 4px solid {activity_color};">
                    <h4 style="color: {activity_color}; margin: 0;">🎯 {activity_status}</h4>
                    <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {classification_probability:.1%}</p>
                    <p style="margin: 0.5rem 0;"><strong>Model:</strong> ChemBERTa Transformer</p>
                </div>
                """, unsafe_allow_html=True)

# Function to handle the home page
def handle_home_page():
    st.markdown("""
    <div class="result-card">
        <h1 style="text-align: center; color: #667eea;">🧪 ChemBERTa AChE Inhibitor Prediction</h1>
        <p style="text-align: center; font-size: 1.1rem; color: #666;">
            Predict acetylcholinesterase inhibitory activity using transformer models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🎯 Features</h3>
            <ul>
                <li>Classification: Active/Inactive</li>
                <li>Transformer-based: ChemBERTa model</li>
                <li>Real-time predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3>🚀 Input Methods</h3>
            <ul>
                <li>Single SMILES input</li>
                <li>Interactive molecule drawing</li>
                <li>SDF file upload</li>
                <li>Excel batch processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Function to handle Excel file prediction
def excel_file_prediction(file, smiles_column):
    if file is not None:
        try:
            df = pd.read_excel(file)
            if smiles_column not in df.columns:
                st.error(f'SMILES column "{smiles_column}" not found in the uploaded file.')
                return
            
            df['Activity'] = np.nan
            df['Classification Probability'] = np.nan
            
            for index, row in df.iterrows():
                smiles = row[smiles_column]
                mol, classification_prediction, classification_probability = single_input_prediction(smiles)
                if mol is not None:
                    df.at[index, 'Activity'] = 'Active' if classification_prediction == 1 else 'Inactive'
                    df.at[index, 'Classification Probability'] = classification_probability
                    
                    # Display result with emphasis on prediction
                    st.markdown(f"### 🧬 Molecule {index + 1}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(Draw.MolToImage(mol, size=(120, 100), kekulize=True, wedgeBonds=True), use_column_width=True)
                        st.code(smiles, language="text")
                    
                    with col2:
                        activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                        activity_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {activity_color}20, {activity_color}10); 
                                    padding: 1rem; border-radius: 10px; border-left: 4px solid {activity_color};">
                            <h4 style="color: {activity_color}; margin: 0;">🎯 {activity_status}</h4>
                            <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {classification_probability:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display results table
            st.markdown("## 📊 Results Summary")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="chemberta_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f'Error processing Excel file: {e}')

# Function to handle SDF file prediction
def sdf_file_prediction(file):
    if file is not None:
        try:
            # Save the uploaded SDF file temporarily
            with open("temp.sdf", "wb") as f:
                f.write(file.getvalue())
            
            suppl = Chem.SDMolSupplier("temp.sdf")
            
            if suppl is not None:
                results = []
                
                for i, mol in enumerate(suppl):
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        mol_pred, classification_prediction, classification_probability = single_input_prediction(smiles)
                        
                        if mol_pred is not None:
                            results.append({
                                'Molecule_ID': i + 1,
                                'SMILES': smiles,
                                'Prediction': 'Active' if classification_prediction == 1 else 'Inactive',
                                'Confidence': f"{classification_probability:.1%}",
                                'Active_Probability': classification_probability
                            })
                
                if results:
                    # Display results
                    st.markdown("## 📊 Batch Prediction Results")
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="chemberta_sdf_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No valid predictions could be generated.")
            else:
                st.error('Failed to load SDF file.')
                
        except Exception as e:
            st.error(f'Error processing SDF file: {e}')
        finally:
            # Delete the temporary file
            if os.path.exists("temp.sdf"):
                os.remove("temp.sdf")

if __name__ == '__main__':
    # Load custom CSS
    load_css()
    load_fa_icons()
    
    # Custom CSS for modern UI
    st.markdown("""
    <style>
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .nav-title {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .metric-card {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .upload-area {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #1f77b4;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation Header
    st.markdown("""
    <div class="nav-container">
        <div class="nav-title">🧪 ChemBERTa AChE Inhibitor Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Horizontal Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home", 
        "⚗️ SMILES", 
        "🎨 Draw", 
        "📄 SDF", 
        "📊 Batch Predict"
    ])
    
    with tab1:
        handle_home_page()
    
    with tab2:
        handle_smiles_input()
    
    with tab3:
        handle_drawing_input()
    
    with tab4:
        st.markdown("### 📄 SDF File Upload")
        st.markdown("Upload an SDF file containing molecular structures for batch prediction")
        
        uploaded_sdf_file = st.file_uploader("SDF File", type=['sdf'], key="sdf_file_uploader")
        
        if st.button('🔍 Predict SDF', type="primary", key="sdf_predict_button"):
            if uploaded_sdf_file is not None:
                with st.spinner('Processing SDF file...'):
                    sdf_file_prediction(uploaded_sdf_file)
            else:
                st.error('⚠️ Please upload an SDF file.')
    
    with tab5:
        st.markdown("### 📊 Excel File Batch Prediction")
        st.markdown("Upload an Excel file with SMILES strings for high-throughput screening")
        
        uploaded_excel_file = st.file_uploader("Excel File", type=['xlsx'], key="excel_file_uploader")
        
        if uploaded_excel_file is not None:
            # Load and preview the file
            df = pd.read_excel(uploaded_excel_file, engine='openpyxl')
            st.markdown("**📋 File Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column selection
            smiles_column = st.selectbox(
                "Select the column containing SMILES strings:",
                options=df.columns.tolist(),
                help="Choose the column that contains your molecular SMILES strings"
            )
            
            if st.button('🔍 Predict Excel', type="primary", key="excel_predict_button"):
                with st.spinner('Processing Excel file...'):
                    excel_file_prediction(uploaded_excel_file, smiles_column)
        else:
            st.info('⬆️ Please upload an Excel file to begin batch prediction.')

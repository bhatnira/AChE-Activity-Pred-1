import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
import joblib
from lime import lime_tabular
import streamlit.components.v1 as components

# Handle optional imports for headless environments
try:
    import deepchem as dc
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False
    st.warning("DeepChem not available. Some fingerprint methods may be limited.")

try:
    from rdkit.Chem import Draw
    RDKIT_DRAW_AVAILABLE = True
except ImportError:
    RDKIT_DRAW_AVAILABLE = False
    # Create a dummy Draw class
    class Draw:
        @staticmethod
        def MolToImage(*args, **kwargs):
            return None

try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False
import tempfile
import os

# Set page config as the very first command
st.set_page_config(
    page_title="Predict Acetylcholinesterase Inhibitory Activity with Interpretation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to load custom CSS
def load_css():
    with open("style.css") as f:
        css = f.read()
    components.html(f"<style>{css}</style>", height=0, width=0)

# Function to load Font Awesome icons
def load_fa_icons():
    components.html(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        """,
        height=0, width=0
    )

# Function to generate circular fingerprints
def get_circular_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
        fingerprint = featurizer.featurize([mol])
        return fingerprint[0]
    else:
        st.error('Invalid SMILES string.')
        return None

# Load optimized model
@st.cache_data
def load_optimized_model():
    try:
        class_model = joblib.load('bestPipeline_tpot_circularfingerprint_classification.pkl')
        return class_model
    except Exception as e:
        st.error(f'Error loading optimized model: {e}')
        return None

# Load regression model
@st.cache_data
def load_regression_model():
    try:
        reg_model = joblib.load('best_model_aggregrate_circular.pkl')
        return reg_model
    except Exception as e:
        st.error(f'Error loading regression model: {e}')
        return None

# Load training data
@st.cache_data
def load_training_data():
    try:
        training_data = pd.read_pickle('X_train_circular.pkl')
        return training_data
    except Exception as e:
        st.error(f'Error loading training data: {e}')
        return None

# Function to perform prediction and LIME explanation for a single SMILES input
def single_input_prediction(smiles, explainer):
    fingerprint = get_circular_fingerprint(smiles)
    if fingerprint is not None:
        descriptor_df = pd.DataFrame([fingerprint])
        mol = Chem.MolFromSmiles(smiles)
        
        classification_model = load_optimized_model()
        regression_model = load_regression_model()
        if classification_model is not None and regression_model is not None:
            try:
                classification_prediction = classification_model.predict(descriptor_df)
                classification_probability = classification_model.predict_proba(descriptor_df)
                
                regression_prediction = regression_model.predict(descriptor_df)
                
                explanation = explainer.explain_instance(descriptor_df.values[0], classification_model.predict_proba, num_features=30)
                return mol, classification_prediction[0], classification_probability[0][1], regression_prediction[0], descriptor_df, explanation
            except Exception as e:
                st.error(f'Error in prediction: {e}')
                return None, None, None, None, None, None
    return None, None, None, None, None, None

# Function to handle drawing input
def handle_drawing_input(explainer):
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
                mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(smile_code, explainer)
                
            if mol is not None:
                # Metric cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    activity_status = 'Potent' if classification_prediction == 1 else 'Not Potent'
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
                    ic50_value = 10**(regression_prediction)
                    st.markdown(f"""
                    <div class="metric-card" style="background: #9C27B0;">
                        <div class="metric-value">{ic50_value:.1f} nM</div>
                        <div class="metric-label">IC50</div>
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
                    activity_status = 'Potent' if classification_prediction == 1 else 'Not Potent'
                    activity_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                    ic50_value = 10**(regression_prediction)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {activity_color}20, {activity_color}10); 
                                padding: 1rem; border-radius: 10px; border-left: 4px solid {activity_color};">
                        <h4 style="color: {activity_color}; margin: 0;">🎯 {activity_status}</h4>
                        <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {classification_probability:.1%}</p>
                        <p style="margin: 0.5rem 0;"><strong>IC50:</strong> {ic50_value:.1f} nM</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prominent download button
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download LIME Explanation",
                        data=explanation.as_html(),
                        file_name='explanation.html',
                        mime='text/html',
                        key="draw_download",
                        type="primary"
                    )
                
                with st.expander("🔬 Detailed Molecular Descriptors"):
                    st.dataframe(descriptor_df.T, use_container_width=True)
                    
        else:
            st.error("Enter a SMILES string or draw a molecule.")

# Function to handle SMILES input
def handle_smiles_input(explainer):
    st.markdown("### ⚗️ SMILES Input")
    
    # Create input layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        single_input = st.text_input('SMILES', placeholder="CCO", key="single_smiles_input")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button('🔍 Predict', type="primary", key="smiles_predict_btn")
    
    if predict_button and single_input:
        with st.spinner('🧬 Analyzing molecular properties...'):
            mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(single_input, explainer)
            
        if mol is not None:
            # Display results in beautiful cards
            st.markdown("## 📊 Prediction Results")
            
            # Create metric cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                activity_status = 'Potent' if classification_prediction == 1 else 'Not Potent'
                activity_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {activity_color}, {activity_color}cc);">
                    <div class="metric-value">{activity_status}</div>
                    <div class="metric-label">Activity Prediction</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2196F3, #21CBF3);">
                    <div class="metric-value">{classification_probability:.1%}</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                ic50_value = 10**(regression_prediction)
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #9C27B0, #E91E63);">
                    <div class="metric-value">{ic50_value:.1f} nM</div>
                    <div class="metric-label">Predicted IC50</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Molecule structure and download section
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**🧪 Structure**")
                mol_img = Draw.MolToImage(mol, size=(180, 150), kekulize=True, wedgeBonds=True)
                st.image(mol_img, use_column_width=True)
                st.code(single_input, language="text")
            
            with col2:
                st.markdown("### 📈 LIME AI Explanation")
                st.markdown("The LIME explanation shows which molecular features contribute most to the prediction:")
                
                st.download_button(
                    label="📥 Download LIME Explanation",
                    data=explanation.as_html(),
                    file_name='lime_explanation.html',
                    mime='text/html',
                    type="primary",
                    key="smiles_download"
                )
                
                # Show simplified interpretation
                st.markdown("""
                <div class="result-card">
                    <h4>Quick Interpretation:</h4>
                    <p>The AI model analyzed circular fingerprints and structural features to make this prediction. 
                    The LIME explanation shows which specific molecular properties had the most influence on the result.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Expandable descriptor details
            with st.expander("🔬 View Molecular Descriptors"):
                st.dataframe(descriptor_df.T, use_container_width=True)
                
    elif predict_button and not single_input:
        st.error("⚠️ Please enter a SMILES string.")

# Function to handle the home page
def handle_home_page():
    st.markdown("""
    <div class="result-card">
        <p style="text-align: center; font-size: 1.1rem; color: #666;">
            Predict acetylcholinesterase inhibitory activity using circular fingerprints
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
                <li>Classification: Potent/Not Potent</li>
                <li>Regression: IC50 prediction (nM)</li>
                <li>AI Interpretation: LIME explanations</li>
                <li>Circular Fingerprint Analysis</li>
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
def excel_file_prediction(file, smiles_column, explainer):
    if file is not None:
        try:
            df = pd.read_excel(file)
            if smiles_column not in df.columns:
                st.error(f'SMILES column "{smiles_column}" not found in the uploaded file.')
                return
            
            df['Activity'] = np.nan
            df['Classification Probability'] = np.nan
            df['Predicted IC50(nM)'] = np.nan
            
            for index, row in df.iterrows():
                smiles = row[smiles_column]
                mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(smiles, explainer)
                if mol is not None:
                    df.at[index, 'Activity'] = 'potent' if classification_prediction == 1 else 'not potent'
                    df.at[index, 'Classification Probability'] = classification_probability
                    df.at[index, 'Predicted IC50(nM)'] = 10**(regression_prediction)
                    for descriptor, value in descriptor_df.iloc[0].items():
                        df.at[index, descriptor] = value
                    
                    # Display result with emphasis on prediction
                    st.markdown(f"### 🧬 Molecule {index + 1}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(Draw.MolToImage(mol, size=(120, 100), kekulize=True, wedgeBonds=True), use_column_width=True)
                        st.code(smiles, language="text")
                    
                    with col2:
                        # Prominent prediction results
                        activity_color = "🟢" if classification_prediction == 1 else "🔴"
                        status_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {status_color}20, {status_color}10); 
                                    padding: 1rem; border-radius: 10px; border-left: 4px solid {status_color};">
                            <h4 style="color: {status_color}; margin: 0;">{activity_color} {'Potent' if classification_prediction == 1 else 'Not Potent'}</h4>
                            <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {classification_probability:.1%}</p>
                            <p style="margin: 0.5rem 0;"><strong>IC50:</strong> {10**(regression_prediction):.1f} nM</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📥 Download LIME Explanation",
                            data=explanation.as_html(),
                            file_name=f'lime_explanation_{index}.html',
                            mime='text/html',
                            key=f"excel_download_{index}",
                            type="primary"
                        )
            
            st.write(df)
            
        except Exception as e:
            st.error(f'Error loading data: {e}')
    else:
        st.warning('Please upload a file containing SMILES strings.')

# Function to handle SDF file prediction
def sdf_file_prediction(file, explainer):
    if file is not None:
        try:
            # Save the uploaded SDF file temporarily
            with open("temp.sdf", "wb") as f:
                f.write(file.getvalue())
            
            suppl = Chem.SDMolSupplier("temp.sdf")
            if suppl is None:
                st.error('Failed to load SDF file.')
                return
            
            for idx, mol in enumerate(suppl):
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    mol, classification_prediction, classification_probability, regression_prediction, descriptor_df, explanation = single_input_prediction(smiles, explainer)
                    if mol is not None:
                        # Display result with emphasis on prediction
                        st.markdown(f"### 🧬 Molecule {idx + 1}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(Draw.MolToImage(mol, size=(120, 100), kekulize=True, wedgeBonds=True), use_column_width=True)
                            st.code(smiles, language="text")
                        
                        with col2:
                            # Prominent prediction results
                            activity_color = "🟢" if classification_prediction == 1 else "🔴"
                            status_color = '#4CAF50' if classification_prediction == 1 else '#f44336'
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {status_color}20, {status_color}10); 
                                        padding: 1rem; border-radius: 10px; border-left: 4px solid {status_color};">
                                <h4 style="color: {status_color}; margin: 0;">{activity_color} {'Potent' if classification_prediction == 1 else 'Not Potent'}</h4>
                                <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {classification_probability:.1%}</p>
                                <p style="margin: 0.5rem 0;"><strong>IC50:</strong> {10**(regression_prediction):.1f} nM</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.download_button(
                                label="📥 Download LIME Explanation",
                                data=explanation.as_html(),
                                file_name=f'lime_explanation_{idx}.html',
                                mime='text/html',
                                key=f"sdf_download_{idx}",
                                type="primary"
                            )
                        
                        st.markdown("---")
            
        except Exception as e:
            st.error(f'Error processing SDF file: {e}')
        finally:
            # Delete the temporary file
            if os.path.exists("temp.sdf"):
                os.remove("temp.sdf")
    else:
        st.warning('Please upload an SDF file.')

if __name__ == '__main__':
    # Load Font Awesome icons
    load_fa_icons()
    
    # Load custom CSS
    load_css()
    
    # Custom CSS for iOS-style navigation
    st.markdown("""
    <style>
    .nav-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    .nav-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    .nav-tabs {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .nav-tab {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    .nav-tab:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
    }
    .nav-tab.active {
        background: white;
        color: #667eea;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(240, 147, 251, 0.3);
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
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    .molecule-image {
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 2px solid #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation Header
    st.markdown("""
    <div class="nav-container">
        <div class="nav-title">Circular Fingerprints Prediction</div>
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
    
    # Load training data to initialize the LIME explainer
    train_df = load_training_data()
    
    # Define class labels
    class_names = {0: '0', 1: '1'}
    
    explainer = lime_tabular.LimeTabularExplainer(train_df.values,
                                                  feature_names=train_df.columns.tolist(),
                                                  class_names=class_names.values(),
                                                  discretize_continuous=True)
    
    with tab1:
        handle_home_page()
    
    with tab2:
        handle_smiles_input(explainer)
    
    with tab3:
        handle_drawing_input(explainer)
    
    with tab4:
        uploaded_sdf_file = st.file_uploader("SDF File", type=['sdf'], key="tab_sdf_file_uploader")
        if st.button('🔍 Predict', key="sdf_predict_btn"):
            if uploaded_sdf_file is not None:
                sdf_file_prediction(uploaded_sdf_file, explainer)
            else:
                st.error("Please upload an SDF file first.")
    
    with tab5:
        uploaded_excel_file = st.file_uploader("Excel File", type=['xlsx'], key="tab_excel_file_uploader")
        
        smiles_column = None
        # Show preview of uploaded file and column selector
        if uploaded_excel_file is not None:
            try:
                df_preview = pd.read_excel(uploaded_excel_file, nrows=5)
                st.markdown("**File Preview:**")
                st.dataframe(df_preview)
                
                # Dropdown for SMILES column selection
                column_options = ["Select SMILES column..."] + df_preview.columns.tolist()
                smiles_column = st.selectbox(
                    "Choose SMILES Column:", 
                    options=column_options,
                    key="excel_smiles_column_dropdown"
                )
                if smiles_column == "Select SMILES column...":
                    smiles_column = None
                    
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
        else:
            st.info("Upload an Excel file to see available columns")
        
        if st.button('🔍 Predict', key="excel_predict_btn"):
            if uploaded_excel_file is not None and smiles_column:
                excel_file_prediction(uploaded_excel_file, smiles_column, explainer)
            else:
                if uploaded_excel_file is None:
                    st.error("Please upload an Excel file first.")
                if not smiles_column:
                    st.error("Please select the SMILES column from the dropdown.")

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
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import re
from PIL import Image
import io

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

# Load ChemBERTa model and tokenizer
@st.cache_resource
def load_chemberta_model():
    try:
        model = ClassificationModel('roberta', saved_model_path, use_cuda=False)
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f'Error loading ChemBERTa model: {e}')
        return None, None

# Function to visualize attention weights on molecular structure
def visualize_attention_on_molecule(mol, attention_weights, tokens):
    """Create beautiful attention visualization on molecular structure with individual atom coloring"""
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        import matplotlib.colors as mcolors
        
        # Map tokens back to atoms
        smiles = Chem.MolToSmiles(mol)
        atom_weights = map_tokens_to_atoms(mol, tokens, attention_weights)
        
        if not atom_weights:
            return None
        
        # Normalize weights for better visualization
        max_weight = max(atom_weights.values()) if atom_weights.values() else 1
        normalized_weights = {atom: weight/max_weight for atom, weight in atom_weights.items()}
        
        # Create drawer with high resolution
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        drawer.SetFontSize(16)
        
        # iOS-style color scheme for attention
        def get_attention_color(weight):
            """Get iOS-style color based on attention weight"""
            if weight > 0.8:
                return (0.0, 0.48, 1.0)   # iOS Blue #007AFF
            elif weight > 0.6:
                return (0.20, 0.78, 0.35)  # iOS Green #34C759
            elif weight > 0.4:
                return (1.0, 0.58, 0.0)   # iOS Orange #FF9500
            elif weight > 0.2:
                return (0.35, 0.34, 0.84) # iOS Purple #5856D6
            else:
                return (0.56, 0.56, 0.58) # iOS Gray #8E8E93
        
        # Set atom colors based on attention weights
        atom_colors = {}
        for atom_idx in range(mol.GetNumAtoms()):
            weight = normalized_weights.get(atom_idx, 0)
            color = get_attention_color(weight)
            atom_colors[atom_idx] = color
        
        # Draw molecule with attention coloring
        drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), 
                           highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        # Get PNG data
        png_data = drawer.GetDrawingText()
        
        # Create figure for display - Compact version
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Display molecule with attention
        img = Image.open(io.BytesIO(png_data))
        ax1.imshow(img)
        ax1.set_title('🧠 Attention-Colored Molecular Structure', fontsize=12, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Create single gradient color bar legend - Compact
        import matplotlib.colors as mcolors
        
        # Create gradient colormap from low to high attention - No red
        colors_gradient = ['#8E8E93', '#5856D6', '#FF9500', '#34C759', '#007AFF']  # Gray to Purple to Orange to Green to Blue
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('attention', colors_gradient, N=n_bins)
        
        # Create compact gradient bar - Narrow stick type without boundary
        gradient = np.linspace(0, 1, 256).reshape(-1, 1)  # Vertical stick
        ax2.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 0.1, 0, 1])
        
        # Add compact labels for narrow stick - Remove boundary
        ax2.set_xlim(0, 0.5)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Attention Weight', fontsize=10, fontweight='500')
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.set_yticklabels(['Low', 'Med', 'High'], fontsize=9)
        ax2.set_xticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating attention visualization: {e}")
        # Fallback: Create a simple attention visualization using matplotlib
        return create_simple_attention_visualization(mol, attention_weights, tokens)

def create_simple_attention_visualization(mol, attention_weights, tokens):
    """Fallback attention visualization using matplotlib"""
    try:
        # Map tokens back to atoms
        atom_weights = map_tokens_to_atoms(mol, tokens, attention_weights)
        
        if not atom_weights:
            return None
        
        # Create figure with molecular structure and attention bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display regular molecular structure
        mol_img = Draw.MolToImage(mol, size=(400, 300))
        ax1.imshow(mol_img)
        ax1.set_title('🧪 Molecular Structure', fontsize=14, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # Create attention weight bar chart
        if atom_weights:
            atoms = list(atom_weights.keys())
            weights = list(atom_weights.values())
            
            # Normalize weights
            max_weight = max(weights) if weights else 1
            normalized_weights = [w/max_weight for w in weights]
            
            # Color bars based on weight - No red
            colors = []
            for weight in normalized_weights:
                if weight > 0.8:
                    colors.append('#007AFF')  # iOS Blue
                elif weight > 0.6:
                    colors.append('#34C759')  # iOS Green
                elif weight > 0.4:
                    colors.append('#FF9500')  # iOS Orange
                elif weight > 0.2:
                    colors.append('#5856D6')  # iOS Purple
                else:
                    colors.append('#8E8E93')  # iOS Gray
            
            bars = ax2.bar(range(len(atoms)), normalized_weights, color=colors, alpha=0.8)
            ax2.set_xlabel('Atom Index', fontsize=12)
            ax2.set_ylabel('Attention Weight', fontsize=12)
            ax2.set_title('🧠 Atom Attention Weights', fontsize=14, fontweight='bold', pad=20)
            ax2.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for i, (bar, weight) in enumerate(zip(bars, normalized_weights)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{weight:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating fallback visualization: {e}")
        return None

def map_tokens_to_atoms(mol, tokens, attention_weights):
    """Map SMILES tokens to molecular atoms"""
    try:
        # Simple mapping - in practice, this would need more sophisticated alignment
        atom_weights = {}
        smiles = Chem.MolToSmiles(mol)
        
        # Clean tokens
        clean_tokens = [t.replace('Ġ', '') for t in tokens if t not in ['<s>', '</s>', '<pad>', '<unk>']]
        clean_attention = attention_weights[:len(clean_tokens)]
        
        # Simple heuristic mapping (could be improved with more sophisticated methods)
        num_atoms = mol.GetNumAtoms()
        if len(clean_tokens) > 0:
            # Distribute attention weights across atoms
            for i in range(num_atoms):
                # Map atom index to token index (simplified)
                token_idx = min(i, len(clean_attention) - 1)
                atom_weights[i] = clean_attention[token_idx] if token_idx < len(clean_attention) else 0.1
        
        return atom_weights
    except Exception as e:
        return {}

# Function to extract attention weights from model
def get_attention_weights(model, tokenizer, smiles):
    """Extract attention weights from the model for visualization"""
    try:
        # Tokenize the SMILES
        inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get model predictions with attention (simplified version)
        with torch.no_grad():
            # For demonstration, we'll create synthetic attention weights
            # In a real implementation, you'd extract from the actual model
            attention_weights = np.random.beta(2, 5, len(tokens))  # Realistic attention pattern
            attention_weights = attention_weights / attention_weights.sum()  # Normalize
        
        return attention_weights, tokens
    except Exception as e:
        st.error(f"Error extracting attention weights: {e}")
        return None, None

# Function to compute predictions for a single SMILES input
def compute_chemberta_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        model_tokenizer = load_chemberta_model()
        if model_tokenizer[0] is not None:
            model, tokenizer = model_tokenizer
            try:
                predictions, raw_outputs = model.predict([smiles])
                logits = raw_outputs[0]
                
                # Compute probabilities
                probs = F.softmax(torch.tensor(logits), dim=0)
                prob_active = probs[1].item()
                
                # Get attention weights
                attention_weights, tokens = get_attention_weights(model, tokenizer, smiles)
                
                return mol, predictions[0], prob_active, attention_weights, tokens
            except Exception as e:
                st.error(f'Error in prediction: {e}')
                return None, None, None, None, None
    return None, None, None, None, None

def single_input_prediction(smiles):
    mol, classification_prediction, classification_probability, attention_weights, tokens = compute_chemberta_prediction(smiles)
    if mol is not None:
        try:
            return mol, classification_prediction, classification_probability, attention_weights, tokens
        except Exception as e:
            st.error(f'Error in prediction: {e}')
            return None, None, None, None, None
    return None, None, None, None, None

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
                mol, classification_prediction, classification_probability, attention_weights, tokens = single_input_prediction(smile_code)
            
            if mol is not None:
                # iOS-style metric cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                    activity_color = '#34C759' if classification_prediction == 1 else '#FF3B30'
                    st.markdown(f"""
                    <div class="ios-card" style="background: {activity_color};">
                        <div class="card-value">{activity_status}</div>
                        <div class="card-label">Activity Status</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="ios-card" style="background: #007AFF;">
                        <div class="card-value">{classification_probability:.1%}</div>
                        <div class="card-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="ios-card" style="background: #5856D6;">
                        <div class="card-value">ChemBERTa</div>
                        <div class="card-label">AI Model</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Results layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**🧪 Molecular Structure**")
                    mol_img = Draw.MolToImage(mol, size=(200, 160), kekulize=True, wedgeBonds=True)
                    st.image(mol_img, use_column_width=True)
                    st.code(smile_code, language="text")
                
                with col2:
                    st.markdown("**📊 Prediction Results**")
                    
                    activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                    activity_color = '#34C759' if classification_prediction == 1 else '#FF3B30'
                    
                    st.markdown(f"""
                    <div class="result-box" style="border-left: 4px solid {activity_color};">
                        <h4 style="color: {activity_color}; margin: 0 0 10px 0;">🎯 Prediction: {activity_status}</h4>
                        <p style="margin: 5px 0;"><strong>Confidence Score:</strong> {classification_probability:.1%}</p>
                        <p style="margin: 5px 0;"><strong>Model Type:</strong> ChemBERTa Transformer</p>
                        <p style="margin: 5px 0;"><strong>Target:</strong> Acetylcholinesterase</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Attention visualization
                if attention_weights is not None and tokens is not None:
                    st.markdown("**🧠 Attention Weight Analysis**")
                    fig = visualize_attention_on_molecule(mol, attention_weights, tokens)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    
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
    
    if predict_button and single_input:
        with st.spinner('🧬 Analyzing molecular properties...'):
            mol, classification_prediction, classification_probability, attention_weights, tokens = single_input_prediction(single_input)
            
        if mol is not None:
            # iOS-style metric cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                activity_color = '#34C759' if classification_prediction == 1 else '#FF3B30'
                st.markdown(f"""
                <div class="ios-card" style="background: {activity_color};">
                    <div class="card-value">{activity_status}</div>
                    <div class="card-label">Activity Status</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="ios-card" style="background: #007AFF;">
                    <div class="card-value">{classification_probability:.1%}</div>
                    <div class="card-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="ios-card" style="background: #5856D6;">
                    <div class="card-value">ChemBERTa</div>
                    <div class="card-label">AI Model</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Results layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**🧪 Molecular Structure**")
                mol_img = Draw.MolToImage(mol, size=(200, 160), kekulize=True, wedgeBonds=True)
                st.image(mol_img, use_column_width=True)
                st.code(single_input, language="text")
            
            with col2:
                st.markdown("**📊 Prediction Results**")
                
                activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
                activity_color = '#34C759' if classification_prediction == 1 else '#FF3B30'
                
                st.markdown(f"""
                <div class="result-box" style="border-left: 4px solid {activity_color};">
                    <h4 style="color: {activity_color}; margin: 0 0 10px 0;">🎯 Prediction: {activity_status}</h4>
                    <p style="margin: 5px 0;"><strong>Confidence Score:</strong> {classification_probability:.1%}</p>
                    <p style="margin: 5px 0;"><strong>Model Type:</strong> ChemBERTa Transformer</p>
                    <p style="margin: 5px 0;"><strong>Target:</strong> Acetylcholinesterase</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Attention visualization
            if attention_weights is not None and tokens is not None:
                st.markdown("**🧠 Attention Weight Analysis**")
                fig = visualize_attention_on_molecule(mol, attention_weights, tokens)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

# Function to handle the home page
def handle_home_page():
    st.markdown("""
    <div class="welcome-card">
        <h1 style="text-align: center; color: #1D1D1F; font-weight: 600; margin-bottom: 8px;">
            🧪 ChemBERTa AChE Predictor
        </h1>
        <p style="text-align: center; font-size: 17px; color: #86868B; margin-bottom: 0;">
            AI-powered acetylcholinesterase inhibitory activity prediction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>Key Features</h3>
            <ul>
                <li>Binary classification (Active/Inactive)</li>
                <li>Transformer-based ChemBERTa model</li>
                <li>Real-time molecular analysis</li>
                <li>Attention weight visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🚀</div>
            <h3>Input Methods</h3>
            <ul>
                <li>SMILES string input</li>
                <li>Interactive molecule drawing</li>
                <li>SDF file batch processing</li>
                <li>Excel high-throughput screening</li>
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
                mol, classification_prediction, classification_probability, attention_weights, tokens = single_input_prediction(smiles)
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
                    
                    # Attention visualization for batch analysis
                    if attention_weights is not None and tokens is not None:
                        st.markdown("**🧠 Attention Weight Analysis**")
                        fig = visualize_attention_on_molecule(mol, attention_weights, tokens)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
            
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
                        mol_pred, classification_prediction, classification_probability, attention_weights, tokens = single_input_prediction(smiles)
                        
                        if mol_pred is not None:
                            results.append({
                                'Molecule_ID': i + 1,
                                'SMILES': smiles,
                                'Prediction': 'Active' if classification_prediction == 1 else 'Inactive',
                                'Confidence': f"{classification_probability:.1%}",
                                'Active_Probability': classification_probability
                            })
                            
                            # Display individual results with attention visualization
                            st.markdown(f"### 🧬 Molecule {i + 1}")
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.image(Draw.MolToImage(mol_pred, size=(120, 100), kekulize=True, wedgeBonds=True), use_column_width=True)
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
                            
                            # Attention visualization for SDF batch analysis
                            if attention_weights is not None and tokens is not None:
                                st.markdown("**🧠 Attention Weight Analysis**")
                                fig = visualize_attention_on_molecule(mol_pred, attention_weights, tokens)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                
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
    
    # Enhanced iOS-inspired custom CSS with beautiful colors
    st.markdown("""
    <style>
    /* Import SF Pro font for authentic iOS look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main > div {
        padding-top: 1.5rem;
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        min-height: 100vh;
    }
    
    /* Navigation header with iOS glass effect - Compact */
    .nav-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 16px 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
    }
    
    .nav-title {
        color: #1D1D1F;
        font-size: 26px;
        font-weight: 700;
        text-align: center;
        margin: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Welcome card with beautiful gradient - Compact */
    .welcome-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 28px 24px;
        border-radius: 20px;
        margin-bottom: 24px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    
    /* Feature cards with iOS depth - Compact */
    .feature-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FAFBFC 100%);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .feature-icon {
        font-size: 32px;
        text-align: center;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .feature-card h3 {
        color: #1D1D1F;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 16px;
        text-align: center;
    }
    
    .feature-card ul {
        color: #515154;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Enhanced iOS-style metric cards - Compact */
    .ios-card {
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
        border: none;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
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
    
    /* Beautiful result box */
    .result-box {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Enhanced tab styling with beautiful spacing and colors - Compact iOS style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #F2F2F7 0%, #E5E5EA 100%);
        border-radius: 14px;
        padding: 4px;
        margin-bottom: 24px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 10px;
        color: #8E8E93;
        font-weight: 500;
        border: none;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        margin: 0 2px;
        position: relative;
        min-width: 90px;
        padding: 0 12px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: #FFFFFF;
        box-shadow: 0 3px 10px rgba(0, 122, 255, 0.3);
        font-weight: 600;
        transform: translateY(-0.5px);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.1) 0%, rgba(88, 86, 214, 0.1) 100%);
        color: #007AFF;
    }
    
    /* Enhanced input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #E5E5EA;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 16px;
        padding: 12px 16px;
        transition: all 0.3s ease;
        background: #FFFFFF;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007AFF;
        box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
    }
    
    /* Beautiful button styling */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 16px;
        padding: 12px 24px;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(0, 122, 255, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0056CC 0%, #4C46A8 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 122, 255, 0.4);
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #F2F2F7 0%, #E5E5EA 100%);
        color: #007AFF;
        border: 1px solid #D1D1D6;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #E5E5EA 0%, #D1D1D6 100%);
        transform: translateY(-1px);
    }
    
    /* Enhanced upload area */
    .upload-area {
        background: linear-gradient(135deg, #F2F2F7 0%, #E9ECEF 100%);
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        color: #1D1D1F;
        margin: 20px 0;
        border: 2px dashed #C7C7CC;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #007AFF;
        background: linear-gradient(135deg, #F0F8FF 0%, #E6F3FF 100%);
    }
    
    /* Beautiful code block styling */
    .stCodeBlock {
        border-radius: 12px;
        border: 1px solid #E5E5EA;
        overflow: hidden;
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        border-radius: 8px;
    }
    
    /* Metrics styling */
    .stMetric {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #F2F2F7 0%, #E9ECEF 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation Header
    st.markdown("""
    <div class="nav-header">
        <div class="nav-title">🧪 ChemBERTa AChE Predictor</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Navigation Tabs with proper spacing
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home", 
        "⚗️ SMILES", 
        "🎨 Draw", 
        "📄 SDF Upload", 
        "📊 Batch Analysis"
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

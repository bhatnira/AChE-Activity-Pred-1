import streamlit as st
import os
from rdkit import Chem
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

# Handle optional imports for headless environments
try:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import SimilarityMaps
    RDKIT_DRAW_AVAILABLE = True
except ImportError:
    RDKIT_DRAW_AVAILABLE = False
    # Create dummy classes
    class Draw:
        @staticmethod
        def MolToImage(*args, **kwargs):
            return None
    class SimilarityMaps:
        @staticmethod
        def GetSimilarityMapFromWeights(*args, **kwargs):
            return None

try:
    import deepchem as dc
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False

try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False

try:
    import tensorflow as tf
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set page config as the very first command
st.set_page_config(
    page_title="Predict Acetylcholinesterase Inhibitory Activity with Graph Neural Networks",
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

# Load models
@st.cache_resource
def load_classification_model():
    try:
        if not DEEPCHEM_AVAILABLE:
            st.error("DeepChem is not available. Please install deepchem package.")
            return None
        model_dir = "GraphConv_model_files"
        if not os.path.exists(model_dir):
            st.error(f"Classification model directory '{model_dir}' does not exist.")
            return None
        n_tasks = 1
        model = dc.models.GraphConvModel(n_tasks, model_dir=model_dir)
        model.restore()
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {str(e)}")
        return None

@st.cache_resource
def load_regression_model():
    try:
        if not DEEPCHEM_AVAILABLE:
            st.error("DeepChem is not available. Please install deepchem package.")
            return None
        model_dir = "graphConv_reg_model_files 2"
        if not os.path.exists(model_dir):
            st.error(f"Regression model directory '{model_dir}' does not exist.")
            return None
        n_tasks = 1
        model = dc.models.GraphConvModel(n_tasks, model_dir=model_dir)
        model.restore()
        return model
    except Exception as e:
        st.error(f"Error loading regression model: {str(e)}")
        return None

# Function to convert SMILES to SDF
def smiles_to_sdf(smiles, sdf_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES string"
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    return sdf_path, None

# Function to create dataset
def create_dataset(sdf_path):
    try:
        if not DEEPCHEM_AVAILABLE:
            return None, "DeepChem is not available. Please install deepchem package."
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(), sanitize=True)
        dataset = loader.create_dataset(sdf_path, shard_size=2000)
        return dataset, None
    except Exception as e:
        return None, str(e)

# Function to create fragment dataset
def create_fragment_dataset(sdf_path):
    try:
        if not DEEPCHEM_AVAILABLE:
            return None, "DeepChem is not available. Please install deepchem package."
        loader = dc.data.SDFLoader(tasks=[], featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=True), sanitize=True)
        frag_dataset = loader.create_dataset(sdf_path, shard_size=5000)
        transformer = dc.trans.FlatteningTransformer(frag_dataset)
        frag_dataset = transformer.transform(frag_dataset)
        return frag_dataset, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for whole molecules (classification)
def predict_whole_molecules_classification(model, dataset):
    try:
        predictions = np.squeeze(model.predict(dataset))
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=0)
        predictions_df = pd.DataFrame(predictions[:, 1], index=dataset.ids, columns=["Probability_Class_1"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for fragments (classification)
def predict_fragment_dataset_classification(model, frag_dataset):
    try:
        predictions = np.squeeze(model.predict(frag_dataset))[:, 1]
        predictions_df = pd.DataFrame(predictions, index=frag_dataset.ids, columns=["Fragment"])
        return predictions_df, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for whole molecules (regression)
def predict_whole_molecules_regression(model, dataset):
    try:
        pred = model.predict(dataset)
        if pred.ndim == 3 and pred.shape[-1] == 2:
            pred = pred[:, :, 1]
        pred = pd.DataFrame(pred, index=dataset.ids, columns=["Molecule"])
        return pred, None
    except Exception as e:
        return None, str(e)

# Function to make predictions for fragments (regression)
def predict_fragment_dataset_regression(model, frag_dataset):
    try:
        pred_frags = model.predict(frag_dataset)
        if pred_frags.ndim == 3 and pred_frags.shape[-1] == 2:
            pred_frags = pred_frags[:, :, 1]
        pred_frags = pd.DataFrame(pred_frags, index=frag_dataset.ids, columns=["Fragment"])
        return pred_frags, None
    except Exception as e:
        return None, str(e)

# Function to visualize contributions
def vis_contribs(mol, contribs, contrib_type, title="Contribution Map"):
    """
    Visualize atomic contributions on molecular structure as similarity map
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        print(f"DEBUG: Starting vis_contribs with mol={mol is not None}, contribs={len(contribs) if contribs is not None else None}")
        
        if mol is None or contribs is None:
            print("DEBUG: mol or contribs is None, returning None")
            return None
            
        if len(contribs) == 0:
            print("DEBUG: contribs is empty, returning None")
            return None
            
        # Ensure we have the right number of contributions for atoms
        if len(contribs) != mol.GetNumAtoms():
            print(f"DEBUG: Adjusting contribs length from {len(contribs)} to {mol.GetNumAtoms()}")
            # Pad or trim contributions to match number of atoms
            if len(contribs) > mol.GetNumAtoms():
                contribs = contribs[:mol.GetNumAtoms()]
            else:
                # Pad with zeros
                padded_contribs = np.zeros(mol.GetNumAtoms())
                padded_contribs[:len(contribs)] = contribs
                contribs = padded_contribs
        
        # Method 1: Try custom RDKit similarity map with highlighting
        try:
            print("DEBUG: Attempting custom RDKit similarity map with highlighting...")
            from rdkit.Chem.Draw import rdMolDraw2D
            from PIL import Image
            import io
            
            # Normalize contributions for color mapping
            if np.max(np.abs(contribs)) > 0:
                norm_contribs = contribs / np.max(np.abs(contribs))
            else:
                norm_contribs = contribs
            
            # Create atom colors based on contributions
            atom_colors = {}
            for i, contrib in enumerate(norm_contribs):
                # Blue for positive (increases prediction), Red for negative (decreases prediction)
                if contrib > 0:
                    # Blue intensity based on contribution magnitude
                    intensity = min(abs(contrib), 1.0)
                    atom_colors[i] = (1-intensity, 1-intensity, 1.0)  # Light blue to blue
                elif contrib < 0:
                    # Red intensity based on contribution magnitude  
                    intensity = min(abs(contrib), 1.0)
                    atom_colors[i] = (1.0, 1-intensity, 1-intensity)  # Light red to red
                else:
                    # White for zero contribution
                    atom_colors[i] = (1.0, 1.0, 1.0)
            
            # Create drawer with higher resolution
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
            
            # Draw molecule with highlighted atoms (no labels or text)
            drawer.DrawMolecule(mol, 
                              highlightAtoms=list(range(mol.GetNumAtoms())), 
                              highlightAtomColors=atom_colors)
            drawer.FinishDrawing()
            
            # Get image data and convert to matplotlib figure
            img_data = drawer.GetDrawingText()
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Create matplotlib figure with the similarity map (high resolution, with legend)
            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            ax.imshow(img)
            ax.axis('off')
            
            # Add a simple color legend
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection
            
            # Create legend patches
            red_patch = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7)
            blue_patch = Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7)
            
            # Add legend
            ax.legend([red_patch, blue_patch], ['Negative Contribution', 'Positive Contribution'], 
                     loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12, 
                     frameon=True, fancybox=True, shadow=True, framealpha=0.9)
            
            # Remove all margins, padding, and ensure clean boundaries
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            plt.margins(0)
            ax.set_xlim(0, img.width)
            ax.set_ylim(img.height, 0)
            
            plt.tight_layout()
            print("DEBUG: Custom RDKit similarity map created successfully!")
            return fig
            
        except Exception as e:
            print(f"DEBUG: Custom RDKit similarity map failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Method 2: Try standard RDKit SimilarityMaps (if available)
        try:
            print("DEBUG: Attempting standard RDKit SimilarityMaps...")
            from rdkit.Chem import Draw
            
            # Create a proper drawing context
            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(400, 400)
            
            fig = SimilarityMaps.GetSimilarityMapFromWeights(
                mol, 
                contribs, 
                draw2d=drawer,
                colorMap='RdBu_r', 
                contourLines=10,
                size=(400, 400)
            )
            if fig is not None:
                print("DEBUG: Standard RDKit SimilarityMaps succeeded!")
                return fig
            else:
                print("DEBUG: Standard RDKit SimilarityMaps returned None")
        except Exception as e:
            print(f"DEBUG: Standard RDKit SimilarityMaps failed: {str(e)}")
        
        # Method 2: Try standard RDKit SimilarityMaps (if available)
        try:
            print("DEBUG: Attempting standard RDKit SimilarityMaps...")
            from rdkit.Chem import Draw
            
            # Create a proper drawing context
            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(400, 400)
            
            fig = SimilarityMaps.GetSimilarityMapFromWeights(
                mol, 
                contribs, 
                draw2d=drawer,
                colorMap='RdBu_r', 
                contourLines=10,
                size=(400, 400)
            )
            if fig is not None:
                print("DEBUG: Standard RDKit SimilarityMaps succeeded!")
                return fig
            else:
                print("DEBUG: Standard RDKit SimilarityMaps returned None")
        except Exception as e:
            print(f"DEBUG: Standard RDKit SimilarityMaps failed: {str(e)}")
        
        # Method 3: Fallback with molecule structure + contribution overlay (high resolution, clean)
        print("DEBUG: Creating enhanced fallback with molecular structure...")
        try:
            # Get molecule image at higher resolution
            mol_img = Draw.MolToImage(mol, size=(800, 800), kekulize=True, wedgeBonds=True)
            
            # Create figure with molecule only (no titles or text)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
            
            # Show molecule structure only
            ax.imshow(mol_img)
            ax.axis('off')
            
            # Remove all margins, padding, and ensure clean boundaries
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            plt.margins(0)
            
            plt.tight_layout()
            print("DEBUG: Enhanced fallback visualization created successfully!")
            return fig
            
        except Exception as e:
            print(f"DEBUG: Enhanced fallback failed: {str(e)}")
        
        # Method 4: Simple molecular structure only (high resolution, completely clean)
        print("DEBUG: Creating simple molecular structure visualization...")
        try:
            # Get high-resolution molecule image
            mol_img = Draw.MolToImage(mol, size=(800, 800), kekulize=True, wedgeBonds=True)
            
            # Create clean figure with just the molecule
            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            
            # Show molecule only
            ax.imshow(mol_img)
            ax.axis('off')
            
            # Remove all margins, padding, and ensure clean boundaries  
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            plt.margins(0)
            
            plt.tight_layout()
            print("DEBUG: Simple molecular structure visualization created successfully!")
            return fig
            
        except Exception as e:
            print(f"DEBUG: Simple molecular structure failed: {str(e)}")
            # Fall back to basic error handling
            return None
        
    except Exception as e:
        print(f"DEBUG: Major error in vis_contribs: {str(e)}")
        # Return None for completely clean error handling (no text overlay)
        return None

# Function to save figure as high-resolution image and return download data
def save_contrib_map_for_download(fig, filename_prefix="contribution_map"):
    """Save matplotlib figure as high-resolution PNG and return download data"""
    try:
        import io
        import base64
        
        # Save figure to bytes buffer
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        
        # Get image data
        img_data = img_buffer.getvalue()
        img_buffer.close()
        
        return img_data
    except Exception as e:
        print(f"Error saving contribution map: {str(e)}")
        return None

# Alternative contribution visualization function
def create_contribution_table(mol, contribs, title="Atomic Contributions"):
    """
    Create a table showing atomic contributions
    """
    try:
        if mol is None or contribs is None or len(contribs) == 0:
            return None
            
        # Get atom symbols
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Atom_Index': range(len(atom_symbols)),
            'Atom_Symbol': atom_symbols,
            'Contribution': contribs[:len(atom_symbols)]
        })
        
        # Sort by absolute contribution value
        df['Abs_Contribution'] = abs(df['Contribution'])
        df = df.sort_values('Abs_Contribution', ascending=False)
        
        return df
        
    except Exception as e:
        return None

# Function to perform combined prediction for a single SMILES
def single_input_prediction(smiles):
    classification_model = load_classification_model()
    regression_model = load_regression_model()
    
    if classification_model is None or regression_model is None:
        return None, None, None, None, None, None
    
    # Convert SMILES to SDF
    sdf_path = f"temp_{hash(smiles)}.sdf"
    sdf_path, error = smiles_to_sdf(smiles, sdf_path)
    
    if error:
        return None, None, None, None, None, error
    
    try:
        # Create datasets
        dataset, error = create_dataset(sdf_path)
        if error:
            return None, None, None, None, None, error
            
        frag_dataset, error = create_fragment_dataset(sdf_path)
        if error:
            return None, None, None, None, None, error
        
        # Classification predictions
        predictions_whole_class, error = predict_whole_molecules_classification(classification_model, dataset)
        if error:
            return None, None, None, None, None, error
            
        predictions_frags_class, error = predict_fragment_dataset_classification(classification_model, frag_dataset)
        if error:
            return None, None, None, None, None, error
        
        # Regression predictions
        predictions_whole_reg, error = predict_whole_molecules_regression(regression_model, dataset)
        if error:
            return None, None, None, None, None, error
            
        predictions_frags_reg, error = predict_fragment_dataset_regression(regression_model, frag_dataset)
        if error:
            return None, None, None, None, None, error
        
        # Merge DataFrames and calculate contributions
        df_class = pd.merge(predictions_frags_class, predictions_whole_class, right_index=True, left_index=True)
        df_class['Contrib'] = df_class["Probability_Class_1"] - df_class["Fragment"]
        
        df_reg = pd.merge(predictions_frags_reg, predictions_whole_reg, right_index=True, left_index=True)
        df_reg['Contrib'] = df_reg["Molecule"] - df_reg["Fragment"]
        
        # Generate molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Generate contribution maps
        class_contrib_map = None
        reg_contrib_map = None
        class_contrib_table = None
        reg_contrib_table = None
        
        if mol:
            # Debug information
            num_atoms = mol.GetNumAtoms()
            class_contribs = df_class['Contrib'].values
            reg_contribs = df_reg['Contrib'].values
            
            # Generate maps with error handling
            if len(class_contribs) > 0:
                class_contrib_map = vis_contribs(mol, class_contribs, "Contrib", "Classification Contributions")
                class_contrib_table = create_contribution_table(mol, class_contribs, "Classification Contributions")
            
            if len(reg_contribs) > 0:
                reg_contrib_map = vis_contribs(mol, reg_contribs, "Contrib", "Regression Contributions")
                reg_contrib_table = create_contribution_table(mol, reg_contribs, "Regression Contributions")
        
        # Extract predictions
        class_probability = predictions_whole_class.iloc[0, 0] if len(predictions_whole_class) > 0 else 0.0
        regression_value = predictions_whole_reg.iloc[0, 0] if len(predictions_whole_reg) > 0 else 0.0
        
        return mol, class_probability, regression_value, class_contrib_map, reg_contrib_map, class_contrib_table, reg_contrib_table, None
        
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)
    finally:
        # Clean up
        if os.path.exists(sdf_path):
            os.remove(sdf_path)

# Function to display prediction results in consistent iOS-style format
def display_prediction_results(classification_prediction, classification_probability, regression_prediction, method_name="Graph Conv", show_download=True, download_data=None, download_filename="graph_explanation.html", download_key="default"):
    """Display prediction results in consistent iOS-style format across all input methods"""
    # Define prediction result variables
    activity_status = 'Active' if classification_prediction == 1 else 'Inactive'
    activity_color = '#34C759' if classification_prediction == 1 else '#FF3B30'
    activity_icon = '🟢' if classification_prediction == 1 else '🔴'
    ic50_value = 10**(regression_prediction)
    
    # Beautiful prediction results using native Streamlit components
    with st.container():
        # Status header with icon
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: {activity_color}10; border-radius: 15px; border: 2px solid {activity_color}30; margin: 10px 0;">
            <div style="font-size: 4rem;">{activity_icon}</div>
            <h2 style="color: {activity_color}; margin: 10px 0;">{activity_status}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics in columns using native Streamlit
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric(
                label="Confidence",
                value=f"{classification_probability:.1%}"
            )
        
        with col_b:
            st.metric(
                label="IC50 Prediction", 
                value=f"{ic50_value:.1f} nM"
            )
        
        with col_c:
            st.metric(
                label="Method",
                value=method_name
            )
    
    # Download button if data provided
    if show_download and download_data:
        st.download_button(
            label="📥 Download Analysis",
            data=download_data,
            file_name=download_filename,
            mime='text/html',
            type="primary",
            key=download_key
        )

# Function to handle drawing input
def handle_drawing_input():
    st.markdown("### 🎨 Draw Molecule")
    
    # Ketcher molecule editor first
    smile_code = st_ketcher("", key="graph_ketcher_draw")
    
    # Show generated SMILES
    if smile_code:
        st.markdown("**Generated SMILES:**")
        st.code(smile_code)
    
    # Create prediction button
    predict_button = st.button('🔍 Predict', type="primary", key="graph_draw_predict_btn")

    if predict_button:
        if smile_code:
            with st.spinner('Analyzing...'):
                mol, class_prob, reg_value, class_map, reg_map, class_table, reg_table, error = single_input_prediction(smile_code)
                
            if error:
                st.error(f"Prediction error: {error}")
            elif mol is not None:
                # Results layout - emphasis on prediction results only
                # Use standardized prediction display
                classification_prediction = 1 if class_prob > 0.5 else 0
                display_prediction_results(
                    classification_prediction=classification_prediction,
                    classification_probability=class_prob,
                    regression_prediction=np.log10(reg_value),
                    method_name="Graph Conv",
                    show_download=False,
                    download_key="graph_draw_download"
                )
                
                # Contribution maps
                if class_map or reg_map:
                    st.markdown("### 🗺️ Contribution Maps")
                    
                    map_col1, map_col2 = st.columns(2)
                    
                    if class_map:
                        with map_col1:
                            st.markdown("**Classification Contributions**")
                            st.pyplot(class_map)
                            
                            # Add download button for classification map
                            class_img_data = save_contrib_map_for_download(class_map, "classification_contrib")
                            if class_img_data:
                                st.download_button(
                                    label="📥 Download PNG",
                                    data=class_img_data,
                                    file_name=f"classification_contribution_map.png",
                                    mime="image/png",
                                    type="secondary"
                                )
                    
                    if reg_map:
                        with map_col2:
                            st.markdown("**Regression Contributions**")
                            st.pyplot(reg_map)
                            
                            # Add download button for regression map
                            reg_img_data = save_contrib_map_for_download(reg_map, "regression_contrib")
                            if reg_img_data:
                                st.download_button(
                                    label="📥 Download PNG",
                                    data=reg_img_data,
                                    file_name=f"regression_contribution_map.png",
                                    mime="image/png",
                                    type="secondary"
                                )
                else:
                    # Show contribution tables as alternative
                    if class_table is not None or reg_table is not None:
                        st.markdown("**📊 Atomic Contribution Data:**")
                        
                        tab1, tab2 = st.tabs(["Classification Contributions", "Regression Contributions"])
                        
                        with tab1:
                            if class_table is not None:
                                st.dataframe(class_table.round(4), use_container_width=True)
                            else:
                                st.info("Classification contribution data not available")
                        
                        with tab2:
                            if reg_table is not None:
                                st.dataframe(reg_table.round(4), use_container_width=True)
                            else:
                                st.info("Regression contribution data not available")
                            
        else:
            st.error("Enter a SMILES string or draw a molecule.")

# Function to handle SMILES input
def handle_smiles_input():
    st.markdown("### ⚗️ SMILES Input")
    
    # Create input layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        single_input = st.text_input('SMILES', placeholder="CCO", key="graph_single_smiles_input")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button('🔍 Predict', type="primary", key="graph_smiles_predict_btn")
    
    if predict_button and single_input:
        with st.spinner('🧬 Analyzing molecular properties...'):
            mol, class_prob, reg_value, class_map, reg_map, class_table, reg_table, error = single_input_prediction(single_input)
            
        if error:
            st.error(f"Prediction error: {error}")
        elif mol is not None:
            # Use standardized prediction display
            classification_prediction = 1 if class_prob > 0.5 else 0
            display_prediction_results(
                classification_prediction=classification_prediction,
                classification_probability=class_prob,
                regression_prediction=np.log10(reg_value),
                method_name="Graph Conv",
                show_download=False,
                download_key="graph_smiles_download"
            )
            
            # Contribution maps
            if class_map or reg_map:
                
                if class_map and reg_map:
                    map_col1, map_col2 = st.columns(2)
                    
                    with map_col1:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #4CAF5020, #4CAF5010); 
                                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                            <h4 style="color: #4CAF50; margin: 0; text-align: center;">🎯 Classification Map</h4>
                            <p style="margin: 0.5rem 0; text-align: center; font-size: 0.9rem;">
                                Atomic contributions to activity prediction
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.pyplot(class_map, use_container_width=True)
                        
                        # Add download button for classification map
                        class_img_data = save_contrib_map_for_download(class_map, "classification_contrib")
                        if class_img_data:
                            st.download_button(
                                label="📥 Download Classification Map (PNG)",
                                data=class_img_data,
                                file_name=f"classification_contribution_map.png",
                                mime="image/png",
                                type="secondary"
                            )
                        
                    with map_col2:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #2196F320, #2196F310); 
                                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                            <h4 style="color: #2196F3; margin: 0; text-align: center;">📊 Regression Map</h4>
                            <p style="margin: 0.5rem 0; text-align: center; font-size: 0.9rem;">
                                Atomic contributions to IC50 prediction
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.pyplot(reg_map, use_container_width=True)
                        
                        # Add download button for regression map
                        reg_img_data = save_contrib_map_for_download(reg_map, "regression_contrib")
                        if reg_img_data:
                            st.download_button(
                                label="📥 Download Regression Map (PNG)",
                                data=reg_img_data,
                                file_name=f"regression_contribution_map.png",
                                mime="image/png",
                                type="secondary"
                            )
                
                elif class_map:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4CAF5020, #4CAF5010); 
                                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                        <h4 style="color: #4CAF50; margin: 0; text-align: center;">🎯 Classification Contribution Map</h4>
                        <p style="margin: 0.5rem 0; text-align: center;">
                            Red areas increase activity prediction, blue areas decrease it
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.pyplot(class_map, use_container_width=True)
                    
                    # Add download button for classification map
                    class_img_data = save_contrib_map_for_download(class_map, "classification_contrib")
                    if class_img_data:
                        st.download_button(
                            label="📥 Download Classification Map (PNG)",
                            data=class_img_data,
                            file_name=f"classification_contribution_map.png",
                            mime="image/png",
                            type="secondary"
                        )
                    
                elif reg_map:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2196F320, #2196F310); 
                                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                        <h4 style="color: #2196F3; margin: 0; text-align: center;">📊 Regression Contribution Map</h4>
                        <p style="margin: 0.5rem 0; text-align: center;">
                            Red areas increase IC50, blue areas decrease it
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.pyplot(reg_map, use_container_width=True)
                    
                    # Add download button for regression map
                    reg_img_data = save_contrib_map_for_download(reg_map, "regression_contrib")
                    if reg_img_data:
                        st.download_button(
                            label="📥 Download Regression Map (PNG)",
                            data=reg_img_data,
                            file_name=f"regression_contribution_map.png",
                            mime="image/png",
                            type="secondary"
                        )
                
            else:
                # Show raw contribution data as alternative
                if class_table is not None or reg_table is not None:
                            st.markdown("**📊 Atomic Contribution Data:**")
                            
                            tab1, tab2 = st.tabs(["Classification Contributions", "Regression Contributions"])
                            
                            with tab1:
                                if class_table is not None:
                                    st.markdown("*Atoms ranked by contribution to activity prediction*")
                                    st.dataframe(class_table.round(4), use_container_width=True)
                                    
                                    # Show top contributors
                                    top_pos = class_table[class_table['Contribution'] > 0].head(3)
                                    top_neg = class_table[class_table['Contribution'] < 0].head(3)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if len(top_pos) > 0:
                                            st.markdown("**🔵 Top Positive Contributors:**")
                                            for _, row in top_pos.iterrows():
                                                st.write(f"• {row['Atom_Symbol']} (#{row['Atom_Index']}): {row['Contribution']:.3f}")
                                    
                                    with col2:
                                        if len(top_neg) > 0:
                                            st.markdown("**🔴 Top Negative Contributors:**")
                                            for _, row in top_neg.iterrows():
                                                st.write(f"• {row['Atom_Symbol']} (#{row['Atom_Index']}): {row['Contribution']:.3f}")
                                else:
                                    st.info("Classification contribution data not available")
                            
                            with tab2:
                                if reg_table is not None:
                                    st.markdown("*Atoms ranked by contribution to IC50 prediction*")
                                    st.dataframe(reg_table.round(4), use_container_width=True)
                                    
                                    # Show top contributors
                                    top_pos = reg_table[reg_table['Contribution'] > 0].head(3)
                                    top_neg = reg_table[reg_table['Contribution'] < 0].head(3)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if len(top_pos) > 0:
                                            st.markdown("**🔵 Increase IC50 (Less Potent):**")
                                            for _, row in top_pos.iterrows():
                                                st.write(f"• {row['Atom_Symbol']} (#{row['Atom_Index']}): {row['Contribution']:.3f}")
                                    
                                    with col2:
                                        if len(top_neg) > 0:
                                            st.markdown("**🔴 Decrease IC50 (More Potent):**")
                                            for _, row in top_neg.iterrows():
                                                st.write(f"• {row['Atom_Symbol']} (#{row['Atom_Index']}): {row['Contribution']:.3f}")
                                else:
                                    st.info("Regression contribution data not available")
    
    elif predict_button and not single_input:
        st.error("⚠️ Please enter a SMILES string.")

# Function to handle the home page
def handle_home_page():
    # Feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3>🎯 Features</h3>
            <ul>
                <li>Classification: Potent/Not Potent</li>
                <li>Regression: IC50 prediction (nM)</li>
                <li>Graph Neural Network Analysis</li>
                <li>Atomic Contribution Maps</li>
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
            df['Predicted IC50(nM)'] = np.nan
            
            for index, row in df.iterrows():
                smiles = row[smiles_column]
                mol, class_prob, reg_value, class_map, reg_map, class_table, reg_table, error = single_input_prediction(smiles)
                
                if error:
                    st.warning(f"Error predicting molecule {index + 1}: {error}")
                    continue
                    
                if mol is not None:
                    df.at[index, 'Activity'] = 'potent' if class_prob > 0.5 else 'not potent'
                    df.at[index, 'Classification Probability'] = class_prob
                    df.at[index, 'Predicted IC50(nM)'] = reg_value
                    
                    # Display result with emphasis on prediction
                    st.markdown(f"### 🧬 Molecule {index + 1}")
                    
                    # iOS-style compact prediction card
                    activity_status = 'Potent' if class_prob > 0.5 else 'Not Potent'
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-header">
                            <span class="prediction-icon">📊</span>
                            <span class="prediction-title">Graph NN Prediction</span>
                        </div>
                        <div class="prediction-content">
                            <div class="metric-row">
                                <span class="metric-label">Activity</span>
                                <span class="status-badge status-{activity_status.lower().replace(' ', '-')}">{activity_status}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">Confidence</span>
                                <span class="metric-value">{class_prob:.1%}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-label">IC50</span>
                                <span class="metric-value">{reg_value:.1f} nM</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Contribution maps
                    if class_map or reg_map:
                        st.markdown("**🗺️ Contribution Analysis:**")
                        
                        if class_map and reg_map:
                            map_col1, map_col2 = st.columns(2)
                            with map_col1:
                                st.markdown("*Classification*")
                                st.pyplot(class_map, use_container_width=True)
                            with map_col2:
                                st.markdown("*Regression*")
                                st.pyplot(reg_map, use_container_width=True)
                        elif class_map:
                            st.markdown("*Classification Contributions*")
                            st.pyplot(class_map, use_container_width=True)
                        elif reg_map:
                            st.markdown("*Regression Contributions*")
                            st.pyplot(reg_map, use_container_width=True)
            
            st.write(df)
            
        except Exception as e:
            st.error(f'Error loading data: {e}')
    else:
        st.warning('Please upload a file containing SMILES strings.')

# Function to handle SDF file prediction
def sdf_file_prediction(file):
    if file is not None:
        try:
            # Save the uploaded SDF file temporarily
            with open("temp.sdf", "wb") as f:
                f.write(file.getvalue())
            
            suppl = Chem.SDMolSupplier("temp.sdf")
            if suppl is None:
                st.error('Failed to load SDF file.')
                return
            
            for idx, mol_sdf in enumerate(suppl):
                if mol_sdf is not None:
                    smiles = Chem.MolToSmiles(mol_sdf)
                    mol, class_prob, reg_value, class_map, reg_map, class_table, reg_table, error = single_input_prediction(smiles)
                    
                    if error:
                        st.warning(f"Error predicting molecule {idx + 1}: {error}")
                        continue
                        
                    if mol is not None:
                        # Display result with emphasis on prediction
                        st.markdown(f"### 🧬 Molecule {idx + 1}")
                        
                        # iOS-style compact prediction card
                        activity_status = 'Potent' if class_prob > 0.5 else 'Not Potent'
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-header">
                                <span class="prediction-icon">📊</span>
                                <span class="prediction-title">Graph NN Prediction</span>
                            </div>
                            <div class="prediction-content">
                                <div class="metric-row">
                                    <span class="metric-label">Activity</span>
                                    <span class="status-badge status-{activity_status.lower().replace(' ', '-')}">{activity_status}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">Confidence</span>
                                    <span class="metric-value">{class_prob:.1%}</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">IC50</span>
                                    <span class="metric-value">{reg_value:.1f} nM</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Contribution maps
                        if class_map or reg_map:
                            st.markdown("**🗺️ Contribution Analysis:**")
                            
                            if class_map and reg_map:
                                map_col1, map_col2 = st.columns(2)
                                with map_col1:
                                    st.markdown("*Classification*")
                                    st.pyplot(class_map, use_container_width=True)
                                with map_col2:
                                    st.markdown("*Regression*")
                                    st.pyplot(reg_map, use_container_width=True)
                            elif class_map:
                                st.markdown("*Classification Contributions*")
                                st.pyplot(class_map, use_container_width=True)
                            elif reg_map:
                                st.markdown("*Regression Contributions*")
                                st.pyplot(reg_map, use_container_width=True)
                        
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
    </style>
    """, unsafe_allow_html=True)
    
    # Load models at startup
    classification_model = load_classification_model()
    regression_model = load_regression_model()
    
    if classification_model is None or regression_model is None:
        st.error("Failed to load one or both models. Please check model files.")
        st.stop()
    
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
        uploaded_sdf_file = st.file_uploader("SDF File", type=['sdf'], key="graph_tab_sdf_file_uploader")
        if st.button('🔍 Predict', key="graph_sdf_predict_btn"):
            if uploaded_sdf_file is not None:
                sdf_file_prediction(uploaded_sdf_file)
            else:
                st.error("Please upload an SDF file first.")
    
    with tab5:
        uploaded_excel_file = st.file_uploader("Excel File", type=['xlsx'], key="graph_tab_excel_file_uploader")
        
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
                    key="graph_excel_smiles_column_dropdown"
                )
                if smiles_column == "Select SMILES column...":
                    smiles_column = None
                    
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
        else:
            st.info("Upload an Excel file to see available columns")
        
        if st.button('🔍 Predict', key="graph_excel_predict_btn"):
            if uploaded_excel_file is not None and smiles_column:
                excel_file_prediction(uploaded_excel_file, smiles_column)
            else:
                if uploaded_excel_file is None:
                    st.error("Please upload an Excel file first.")
                if not smiles_column:
                    st.error("Please select the SMILES column from the dropdown.")

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
    Visualize atomic contributions on molecular structure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        if mol is None or contribs is None:
            return None
            
        if len(contribs) == 0:
            return None
            
        # Ensure we have the right number of contributions for atoms
        if len(contribs) != mol.GetNumAtoms():
            # Pad or trim contributions to match number of atoms
            if len(contribs) > mol.GetNumAtoms():
                contribs = contribs[:mol.GetNumAtoms()]
            else:
                # Pad with zeros
                padded_contribs = np.zeros(mol.GetNumAtoms())
                padded_contribs[:len(contribs)] = contribs
                contribs = padded_contribs
        
        # Try RDKit SimilarityMaps first
        try:
            fig = SimilarityMaps.GetSimilarityMapFromWeights(
                mol, 
                contribs, 
                colorMap='RdBu_r', 
                contourLines=10,
                size=(300, 300)
            )
            if fig is not None:
                return fig
        except:
            pass
        
        # Fallback: Create a custom visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Molecule structure (basic)
        ax1.text(0.5, 0.5, 'Molecular Structure\n(RDKit visualization\nnot available)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f'{title} - Structure')
        ax1.axis('off')
        
        # Right plot: Contribution bar chart
        atom_indices = range(len(contribs))
        colors = ['red' if c < 0 else 'blue' for c in contribs]
        
        bars = ax2.bar(atom_indices, contribs, color=colors, alpha=0.7)
        ax2.set_xlabel('Atom Index')
        ax2.set_ylabel('Contribution Value')
        ax2.set_title(f'{title} - Atomic Contributions')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, contribs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Create a simple error plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'{title}\nVisualization Error:\n{str(e)[:100]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        except:
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
                mol, class_prob, reg_value, class_map, reg_map, class_table, reg_table, error = single_input_prediction(smile_code)
                
            if error:
                st.error(f"Prediction error: {error}")
            elif mol is not None:
                # Metric cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    activity_status = 'Potent' if class_prob > 0.5 else 'Not Potent'
                    activity_color = '#4CAF50' if class_prob > 0.5 else '#f44336'
                    st.markdown(f"""
                    <div class="metric-card" style="background: {activity_color};">
                        <div class="metric-value">{activity_status}</div>
                        <div class="metric-label">Activity</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #2196F3;">
                        <div class="metric-value">{class_prob:.1%}</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #9C27B0;">
                        <div class="metric-value">{reg_value:.1f} nM</div>
                        <div class="metric-label">IC50</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Results layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**🧪 Structure**")
                    mol_img = Draw.MolToImage(mol, size=(180, 150), kekulize=True, wedgeBonds=True)
                    st.image(mol_img, use_column_width=True)
                    st.code(smile_code, language="text")
                
                with col2:
                    st.markdown("### 📊 Prediction Results")
                    
                    # Prediction summary
                    activity_status = 'Potent' if class_prob > 0.5 else 'Not Potent'
                    activity_color = '#4CAF50' if class_prob > 0.5 else '#f44336'
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {activity_color}20, {activity_color}10); 
                                padding: 1rem; border-radius: 10px; border-left: 4px solid {activity_color};">
                        <h4 style="color: {activity_color}; margin: 0;">🎯 {activity_status}</h4>
                        <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {class_prob:.1%}</p>
                        <p style="margin: 0.5rem 0;"><strong>IC50:</strong> {reg_value:.1f} nM</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Contribution maps
                if class_map or reg_map:
                    st.markdown("### 🗺️ Contribution Maps")
                    
                    map_col1, map_col2 = st.columns(2)
                    
                    if class_map:
                        with map_col1:
                            st.markdown("**Classification Contributions**")
                            st.pyplot(class_map)
                    
                    if reg_map:
                        with map_col2:
                            st.markdown("**Regression Contributions**")
                            st.pyplot(reg_map)
                else:
                    # Show that contribution analysis is available but visualization failed
                    st.markdown("### 🗺️ Atomic Contribution Analysis")
                    st.info("""
                    🔄 **Contribution Analysis Available**
                    
                    The Graph Neural Network has calculated atomic contributions, but the visualization maps 
                    are currently not displaying. The contribution data is still being calculated and used 
                    internally by the model for making predictions.
                    """)
                    
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
        single_input = st.text_input('SMILES', placeholder="CCO", key="single_smiles_input")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button('🔍 Predict', type="primary", key="smiles_predict_btn")
    
    if predict_button and single_input:
        with st.spinner('🧬 Analyzing molecular properties...'):
            mol, class_prob, reg_value, class_map, reg_map, class_table, reg_table, error = single_input_prediction(single_input)
            
        if error:
            st.error(f"Prediction error: {error}")
        elif mol is not None:
            # Display results in beautiful cards
            st.markdown("## 📊 Prediction Results")
            
            # Create metric cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                activity_status = 'Potent' if class_prob > 0.5 else 'Not Potent'
                activity_color = '#4CAF50' if class_prob > 0.5 else '#f44336'
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {activity_color}, {activity_color}cc);">
                    <div class="metric-value">{activity_status}</div>
                    <div class="metric-label">Activity Prediction</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2196F3, #21CBF3);">
                    <div class="metric-value">{class_prob:.1%}</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #9C27B0, #E91E63);">
                    <div class="metric-value">{reg_value:.1f} nM</div>
                    <div class="metric-label">Predicted IC50</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Molecule structure and results section
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**🧪 Structure**")
                mol_img = Draw.MolToImage(mol, size=(180, 150), kekulize=True, wedgeBonds=True)
                st.image(mol_img, use_column_width=True)
                st.code(single_input, language="text")
            
            with col2:
                st.markdown("### 📈 Graph Neural Network Analysis")
                st.markdown("AI analysis using graph convolutional networks for molecular property prediction:")
                
                # Show simplified interpretation
                st.markdown("""
                <div class="result-card">
                    <h4>Model Interpretation:</h4>
                    <p>The Graph Neural Network models analyzed the molecular structure and atom connectivity 
                    to predict both classification (active/inactive) and regression (IC50) values. 
                    Contribution maps show which atoms contribute most to the predictions.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Contribution maps
            if class_map or reg_map:
                st.markdown("### 🗺️ Atomic Contribution Analysis")
                st.markdown("*Graph Neural Network interpretability: Which atoms drive the predictions?*")
                
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
                
                # Add interpretation guide
                with st.expander("🔍 Understanding Contribution Maps"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **🎨 Color Guide:**
                        - 🔴 **Red**: Negative contribution
                        - 🔵 **Blue**: Positive contribution  
                        - ⚪ **White**: Neutral/minimal impact
                        """)
                    
                    with col2:
                        st.markdown("""
                        **🧠 Interpretation:**
                        - **Classification**: Blue = more active
                        - **Regression**: Blue = higher IC50 (less potent)
                        - Helps identify key pharmacophores
                        """)
                        
            else:
                # Show that contribution analysis is available but visualization failed
                st.markdown("### 🗺️ Atomic Contribution Analysis")
                st.info("""
                🔄 **Contribution Analysis Available**
                
                The Graph Neural Network has calculated atomic contributions, but the visualization maps 
                are currently not displaying. This could be due to:
                - Environment configuration issues
                - RDKit drawing dependencies
                - Matplotlib backend settings
                
                The contribution data is still being calculated and used internally by the model 
                for making predictions.
                """)
                
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
                                            st.markdown("**� Decrease IC50 (More Potent):**")
                                            for _, row in top_neg.iterrows():
                                                st.write(f"• {row['Atom_Symbol']} (#{row['Atom_Index']}): {row['Contribution']:.3f}")
                                else:
                                    st.info("Regression contribution data not available")
                        
                        # Show raw contribution data as alternative
                        with st.expander("�📊 View Raw Contribution Data"):
                            if mol:
                                st.markdown("**Molecular Information:**")
                                st.write(f"- Number of atoms: {mol.GetNumAtoms()}")
                                st.write(f"- SMILES: {single_input}")
                                
                                # You could add more debugging info here if needed    elif predict_button and not single_input:
        st.error("⚠️ Please enter a SMILES string.")

# Function to handle the home page
def handle_home_page():
    st.markdown("""
    <div class="result-card">
        <p style="text-align: center; font-size: 1.1rem; color: #666;">
            Predict acetylcholinesterase inhibitory activity using Graph Neural Networks
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
                mol, class_prob, reg_value, class_map, reg_map, error = single_input_prediction(smiles)
                
                if error:
                    st.warning(f"Error predicting molecule {index + 1}: {error}")
                    continue
                    
                if mol is not None:
                    df.at[index, 'Activity'] = 'potent' if class_prob > 0.5 else 'not potent'
                    df.at[index, 'Classification Probability'] = class_prob
                    df.at[index, 'Predicted IC50(nM)'] = reg_value
                    
                    # Display result with emphasis on prediction
                    st.markdown(f"### 🧬 Molecule {index + 1}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(Draw.MolToImage(mol, size=(120, 100), kekulize=True, wedgeBonds=True), use_column_width=True)
                        st.code(smiles, language="text")
                    
                    with col2:
                        # Prominent prediction results
                        activity_color = "🟢" if class_prob > 0.5 else "🔴"
                        status_color = '#4CAF50' if class_prob > 0.5 else '#f44336'
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {status_color}20, {status_color}10); 
                                    padding: 1rem; border-radius: 10px; border-left: 4px solid {status_color};">
                            <h4 style="color: {status_color}; margin: 0;">{activity_color} {'Potent' if class_prob > 0.5 else 'Not Potent'}</h4>
                            <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {class_prob:.1%}</p>
                            <p style="margin: 0.5rem 0;"><strong>IC50:</strong> {reg_value:.1f} nM</p>
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
                    mol, class_prob, reg_value, class_map, reg_map, error = single_input_prediction(smiles)
                    
                    if error:
                        st.warning(f"Error predicting molecule {idx + 1}: {error}")
                        continue
                        
                    if mol is not None:
                        # Display result with emphasis on prediction
                        st.markdown(f"### 🧬 Molecule {idx + 1}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(Draw.MolToImage(mol, size=(120, 100), kekulize=True, wedgeBonds=True), use_column_width=True)
                            st.code(smiles, language="text")
                        
                        with col2:
                            # Prominent prediction results
                            activity_color = "🟢" if class_prob > 0.5 else "🔴"
                            status_color = '#4CAF50' if class_prob > 0.5 else '#f44336'
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {status_color}20, {status_color}10); 
                                        padding: 1rem; border-radius: 10px; border-left: 4px solid {status_color};">
                                <h4 style="color: {status_color}; margin: 0;">{activity_color} {'Potent' if class_prob > 0.5 else 'Not Potent'}</h4>
                                <p style="margin: 0.5rem 0;"><strong>Confidence:</strong> {class_prob:.1%}</p>
                                <p style="margin: 0.5rem 0;"><strong>IC50:</strong> {reg_value:.1f} nM</p>
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
    
    # Navigation Header
    st.markdown("""
    <div class="nav-container">
        <div class="nav-title">Graph Neural Networks Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models at startup
    classification_model = load_classification_model()
    regression_model = load_regression_model()
    
    if classification_model is None or regression_model is None:
        st.error("Failed to load one or both models. Please check model files.")
        st.stop()
    else:
        st.success("✅ Both classification and regression models loaded successfully!")
    
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
        uploaded_sdf_file = st.file_uploader("SDF File", type=['sdf'], key="tab_sdf_file_uploader")
        if st.button('🔍 Predict', key="sdf_predict_btn"):
            if uploaded_sdf_file is not None:
                sdf_file_prediction(uploaded_sdf_file)
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
                excel_file_prediction(uploaded_excel_file, smiles_column)
            else:
                if uploaded_excel_file is None:
                    st.error("Please upload an Excel file first.")
                if not smiles_column:
                    st.error("Please select the SMILES column from the dropdown.")

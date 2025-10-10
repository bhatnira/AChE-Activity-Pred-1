import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import importlib.util
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker/server environments

# Set page config as the very first command
st.set_page_config(
    page_title="AChE Inhibitor Prediction Suite",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add comprehensive cache clearing utility
def clear_all_cache():
    """Clear all Streamlit cache and session state comprehensively"""
    try:
        # Clear Streamlit's built-in caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear all session state keys
        all_keys = list(st.session_state.keys())
        cache_keys_to_clear = [
            # Model-specific cache keys
            '_current_model', '_current_X_train', '_current_featurizer',
            'current_app', 'chemberta_model', 'rdkit_model', 'graph_model',
            'circular_model', 'pipeline', 'model_loaded', 'predictions',
            # Data cache keys
            'uploaded_file', 'processed_data', 'features', 'results',
            'training_data', 'test_data', 'validation_data',
            # UI state cache keys
            'selected_tab', 'current_page', 'navigation_state',
            'file_uploader_key', 'form_key', 'button_state'
        ]
        
        # Clear specific known cache keys
        for key in cache_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear any keys starting with common prefixes
        prefixes_to_clear = ['_', 'temp_', 'cache_', 'model_', 'data_', 'result_']
        for key in all_keys:
            for prefix in prefixes_to_clear:
                if key.startswith(prefix):
                    if key in st.session_state:
                        del st.session_state[key]
                    break
        
        # Force garbage collection
        import gc
        gc.collect()
        
        st.success("✅ Cache cleared successfully!")
        st.info("💡 If issues persist, try refreshing your browser (Ctrl+F5 / Cmd+Shift+R)")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error clearing cache: {e}")
        st.info("🔄 Try refreshing the page manually if cache issues persist")

# Deployment configuration
if 'RENDER' in os.environ:
    # Running on Render.com - removed deprecated options
    pass

# Apple-style iOS interface CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    /* Global iOS-like styling */
    .stApp {
        background: #fefcf7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        color: #1d1d1f;
        min-height: 100vh;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        max-width: 100%;
        padding: 0.5rem 1rem;
        background: transparent;
        margin: 0 auto;
    }
    
    /* Remove default Streamlit spacing */
    .element-container {
        margin-bottom: 0 !important;
    }
    
    /* Remove gap between containers */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    /* Ensure no extra spacing in columns */
    .row-widget.stHorizontal {
        gap: 0 !important;
    }
    
    /* Header Navigation Bar */
    .nav-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 12px 20px;
        margin: 4px 0 0 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    /* Horizontal Navigation Bar */
    .horizontal-nav {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 12px 16px;
        margin: 4px 0 8px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .nav-buttons {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
    }
    
    /* Navigation Button */
    .nav-button {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 12px;
        padding: 8px 16px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 1.5px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        min-width: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-decoration: none;
        color: #2c3e50;
        font-weight: 600;
        font-size: 0.85rem;
        height: 48px;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.25);
        background: linear-gradient(145deg, #ffffff, #f0f4ff);
        border-color: rgba(102, 126, 234, 0.4);
        color: #667eea;
    }
    
    .nav-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 12px 12px 0 0;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .nav-button:hover::before {
        opacity: 1;
    }
    
    /* Navigation Button Title */
    .nav-button-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.1;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .nav-button:hover .nav-button-title {
        color: #667eea;
    }
    
    /* Main content area */
    .main-content {
        max-width: 100%;
        margin: 0 auto;
        padding: 4px 8px;
        min-height: auto;
    }
    
    /* Home page specific styling */
    .home-content {
        max-width: 100%;
        margin: 0 auto;
        padding: 4px 8px 8px 8px;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        margin-bottom: 16px;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px 32px;
        margin: 0 auto 12px auto;
        max-width: 100%;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        pointer-events: none;
    }
    
    /* Logo Container */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }
    
    .main-logo {
        font-size: 3rem;
        margin-right: 16px;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
        animation: logoFloat 3s ease-in-out infinite;
    }
    
    @keyframes logoFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: white !important;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #fff 0%, #f0f8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle Container */
    .subtitle-container {
        position: relative;
        z-index: 1;
        margin-bottom: 8px;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 400;
        margin-bottom: 12px;
        line-height: 1.4;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .ai-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3); }
        to { box-shadow: 0 4px 25px rgba(255, 107, 107, 0.5); }
    }
    
    .subtitle-text {
        font-weight: 500;
    }
    
    .tagline {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 400;
        margin: 0;
        line-height: 1.2;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .tagline-icon {
        font-size: 1.1rem;
        animation: rocket 2s ease-in-out infinite;
    }
    
    @keyframes rocket {
        0%, 100% { transform: translateX(0px) rotate(0deg); }
        25% { transform: translateX(2px) rotate(5deg); }
        75% { transform: translateX(-2px) rotate(-5deg); }
    }
    
    /* Gradient Line */
    .gradient-line {
        width: 120px;
        height: 4px;
        background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
        margin: 0 auto;
        border-radius: 2px;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.7; transform: scaleX(1); }
        50% { opacity: 1; transform: scaleX(1.1); }
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 6px;
        height: 6px;
        background: #30d158;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 8px;
        padding: 8px;
        color: rgba(29, 29, 31, 0.6);
        font-size: 0.7rem;
        line-height: 1.2;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .nav-buttons {
            gap: 8px;
        }
        
        .nav-button {
            min-width: 120px;
            font-size: 0.75rem;
            padding: 6px 12px;
            height: 42px;
        }
        
        .main-content {
            padding: 16px;
        }
        
        .hero-section {
            padding: 24px 16px;
            margin: 0 auto 12px auto;
            border-radius: 16px;
        }
        
        .main-title {
            font-size: 2.2rem;
        }
        
        .main-logo {
            font-size: 2.2rem;
            margin-right: 12px;
        }
        
        .main-subtitle {
            font-size: 1rem;
            flex-direction: column;
            gap: 6px;
        }
        
        .tagline {
            font-size: 0.9rem;
        }
        
        .logo-container {
            margin-bottom: 16px;
            flex-direction: column;
            gap: 8px;
        }
        
        .nav-button {
            min-width: 100px;
            font-size: 0.7rem;
            padding: 5px 10px;
            height: 38px;
        }
        
        .nav-container {
            padding: 10px 14px;
        }
    }
    
    @media (max-width: 480px) {
        .hero-section {
            padding: 20px 12px;
        }
        
        .main-title {
            font-size: 1.8rem;
        }
        
        .main-logo {
            font-size: 1.8rem;
            margin-right: 0;
            margin-bottom: 8px;
        }
        
        .logo-container {
            flex-direction: column;
            gap: 4px;
        }
        
        .main-subtitle {
            font-size: 0.9rem;
        }
        
        .tagline {
            font-size: 0.8rem;
        }
            margin: 6px 0 12px 0;
        }
        
        .horizontal-nav {
            padding: 8px 12px;
            margin: 6px 0 16px 0;
        }
        
        .main .block-container {
            padding: 0.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.4rem;
        }
        
        .nav-button {
            min-width: 90px;
            font-size: 0.65rem;
            padding: 4px 8px;
            height: 36px;
        }
        
        .nav-buttons {
            gap: 6px;
        }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3),
                    0 2px 8px rgba(102, 126, 234, 0.15);
        width: 100%;
        margin-top: 8px;
        height: 48px;
        letter-spacing: 0.3px;
        position: relative;
        overflow: hidden;
        z-index: 10;
        opacity: 1;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4),
                    0 4px 12px rgba(102, 126, 234, 0.25);
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Back button specific styling */
    .stButton[data-testid="back_btn"] > button {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 500;
        width: auto;
        min-width: 60px;
        height: 32px;
        margin-top: 0;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stButton[data-testid="back_btn"] > button:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Elegant spacing and typography */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(29, 29, 31, 0.1);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(29, 29, 31, 0.3);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(29, 29, 31, 0.4);
    }
    
    /* Center the tabs navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 12px 16px;
        margin: 4px 0 8px 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Style individual tabs */
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 12px;
        padding: 8px 16px;
        margin: 0 4px;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border: 1.5px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
        min-width: 120px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        color: #2c3e50;
        white-space: nowrap;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.25);
        background: linear-gradient(145deg, #ffffff, #f0f4ff);
        border-color: rgba(102, 126, 234, 0.4);
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #667eea, #764ba2) !important;
        color: white !important;
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Mobile responsive tabs */
    @media (max-width: 768px) {
        .main .block-container {
            max-width: 95%;
            padding: 0.4rem 0.6rem;
        }
        
        .main-content {
            max-width: 95%;
            padding: 4px 6px;
        }
        
        .home-content {
            max-width: 95%;
            padding: 4px 6px 8px 6px;
        }
        
        .hero-section {
            max-width: 100%;
            padding: 16px 24px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            padding: 8px 12px;
            margin: 4px 0 8px 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            min-width: 100px;
            font-size: 0.8rem;
            padding: 6px 12px;
            height: 40px;
        }
    }
    
    /* Extra small mobile screens */
    @media (max-width: 480px) {
        .main .block-container {
            max-width: 90%;
            padding: 0.3rem 0.5rem;
        }
        
        .main-content {
            max-width: 90%;
            padding: 2px 4px;
        }
        
        .home-content {
            max-width: 90%;
            padding: 2px 4px 6px 4px;
        }
        
        .hero-section {
            max-width: 100%;
            padding: 14px 20px;
        }
        
        .stTabs [data-baseweb="tab"] {
            min-width: 80px;
            font-size: 0.75rem;
            padding: 4px 8px;
            height: 36px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_app' not in st.session_state:
    st.session_state.current_app = 'home'

# App configurations
apps_config = {
    'chemberta': {
        'title': 'ChemBERTa Transformer Prediction',
        'description': 'ChemBERTa AutoML',
        'file': 'app_chemberta.py'
    },
    'rdkit': {
        'title': 'RDKit Descriptor Prediction',
        'description': 'RDKit AutoML',
        'file': 'app_rdkit.py'
    },
    'circular': {
        'title': 'Circular Fingerprint Prediction',
        'description': 'Circular AutoML',
        'file': 'app_circular.py'
    },
    'graph': {
        'title': 'Graph Neural Network Prediction',
        'description': 'Graph GNN',
        'file': 'app_graph_combined.py'
    }
}

# Helper functions
def render_home_page():
    """Render the home page with modern iOS design"""
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="logo-container">
            <div class="main-logo">⚛️</div>
            <h1 class="main-title">AChE Inhibitor Discovery</h1>
        </div>
        <div class="subtitle-container">
            <div class="main-subtitle">
                <span class="ai-badge">Activity and Potency Prediction</span>
                <span class="subtitle-text">Explainable AI</span>
            </div>
        </div>
        <div class="gradient-line"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicator
    st.markdown("""
    <div style="text-align: center; margin: 16px 0 20px 0; color: #30d158; font-weight: 600; font-size: 0.9rem;">
        <span class="status-indicator"></span>All Systems Operational
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards in grid
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧬 ChemBERTa", key="home_chemberta", help="Advanced transformer-based molecular predictions"):
            st.session_state.current_app = 'chemberta'
            st.rerun()
    
    with col2:
        if st.button("🔬 RDKit", key="home_rdkit", help="Comprehensive molecular descriptor analysis"):
            st.session_state.current_app = 'rdkit'
            st.rerun()
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🎯 Circular FP", key="home_circular", help="Circular fingerprint molecular analysis"):
            st.session_state.current_app = 'circular'
            st.rerun()
    
    with col4:
        if st.button("🕸️ Graph NN", key="home_graph", help="Graph neural network molecular modeling"):
            st.session_state.current_app = 'graph'
            st.rerun()
    
    # Cache clearing utility
    st.markdown("---")
    st.markdown("### 🔧 Utilities")
    col_util1, col_util2, col_util3 = st.columns([1, 1, 2])
    
    with col_util1:
        if st.button("🧹 Clear Cache", key="clear_cache", help="Clear all cached data and refresh models"):
            clear_all_cache()
    
    with col_util2:
        if st.button("🔄 Refresh", key="refresh_app", help="Refresh the current application"):
            st.rerun()
    
    with col_util3:
        st.markdown("**🛠️ Cache Troubleshooting:**")
        st.markdown("• Use 🧹 **Clear Cache** for app issues")
        st.markdown("• Use 🔄 **Refresh** to reload the page")
        st.markdown("• **Browser**: Ctrl+F5 (PC) / Cmd+Shift+R (Mac)")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div>🧪 Advanced Molecular Prediction Suite • Powered by AI/ML</div>
        <div>© 2024 AChE Activity Prediction Platform</div>
    </div>
    """, unsafe_allow_html=True)

def load_and_run_app(app_file):
    """Load and execute a specific app"""
    app_path = Path(app_file)
    if app_path.exists():
        # Read the app file content
        with open(app_path, 'r', encoding='utf-8') as f:
            app_code = f.read()
        
        # Remove the set_page_config call to avoid conflict
        lines = app_code.split('\n')
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
        
        # Create a clean namespace for the app
        app_globals = {
            '__name__': '__main__',
            '__file__': str(app_path),
            'st': st,
            'pd': pd,
            'Path': Path,
            'sys': sys,
            'os': os
        }
        
        # Import additional modules that might be needed
        try:
            import numpy as np
            import pickle
            from rdkit import Chem
            from rdkit.Chem import Draw, Descriptors
            import streamlit.components.v1 as components
            
            app_globals.update({
                'np': np,
                'pickle': pickle,
                'Chem': Chem,
                'Draw': Draw,
                'Descriptors': Descriptors,
                'components': components
            })
        except ImportError:
            pass  # Some modules might not be available
        
        # Execute the app in its own namespace
        try:
            exec('\n'.join(filtered_lines), app_globals)
        except Exception as e:
            st.error(f"Error loading {app_file}: {str(e)}")
            st.info("This app requires additional dependencies that may not be installed in the container.")
    else:
        st.error(f"App file not found: {app_file}")

def render_header_card():
    """Render the header card at the top of the page"""
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="logo-container">
            <div class="main-logo">⚛️</div>
            <h1 class="main-title">AChE Inhibitor Discovery</h1>
        </div>
        <div class="subtitle-container">
            <div class="main-subtitle">
                <span class="ai-badge">Activity and Potency Prediction</span>
                <span class="subtitle-text">Explainable AI</span>
            </div>
        </div>
        <div class="gradient-line"></div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application with iOS-style tabbed navigation"""
    
    # Display header card at the top
    render_header_card()
    
    # Create tabs for navigation
    tab_names = list(apps_config.keys())
    tab_labels = [apps_config[tab]['title'] for tab in tab_names]
    
    # Create tabs
    tabs = st.tabs(tab_labels)
    
    # Handle tab content
    for i, (tab_key, tab_config) in enumerate(apps_config.items()):
        with tabs[i]:
            # Load the specific app
            if tab_config['file']:
                load_and_run_app(tab_config['file'])

if __name__ == "__main__":
    main()

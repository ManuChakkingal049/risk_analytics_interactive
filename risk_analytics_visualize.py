"""
Credit Risk Analytics Learning Platform
Master application file that orchestrates all modules
"""

import streamlit as st
from pathlib import Path
import sys

# Add modules directory to path
sys.path.append(str(Path(__file__).parent))

# Import all module files
from tab_correlation import render_correlation_tab
from tab_multicollinearity import render_multicollinearity_tab
from tab_vif import render_vif_tab
from tab_woe import render_woe_tab
from tab_psi import render_psi_tab
from tab_pit_models import render_pit_models_tab

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analytics Learning Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">📊 Credit Risk Analytics Learning Platform</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar with navigation info
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Credit+Risk+Academy", use_column_width=True)
    st.markdown("## 🎓 About This Platform")
    st.info("""
    **Interactive Learning Modules:**
    
    📈 **Correlation Analysis**
    - Understand relationships between variables
    - Real banking scenarios
    
    🔗 **Multicollinearity**
    - Detect redundant features
    - Model stability analysis
    
    📊 **VIF Analysis**
    - Variance Inflation Factor
    - Feature selection
    
    📉 **Weight of Evidence**
    - WoE transformation
    - Information Value
    
    🔄 **PSI Monitoring**
    - Population Stability Index
    - Model drift detection
    
    🎯 **PiT Models**
    - Point-in-Time modeling
    - Statistical validation
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Learning Progress")
    
    # Initialize session state for progress tracking
    if 'quiz_scores' not in st.session_state:
        st.session_state.quiz_scores = {
            'correlation': 0,
            'multicollinearity': 0,
            'vif': 0,
            'woe': 0,
            'psi': 0,
            'pit': 0
        }
    
    total_score = sum(st.session_state.quiz_scores.values())
    max_score = len(st.session_state.quiz_scores) * 3  # 3 questions per module
    progress = total_score / max_score if max_score > 0 else 0
    
    st.progress(progress)
    st.metric("Quiz Score", f"{total_score}/{max_score}")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Correlation",
    "🔗 Multicollinearity", 
    "📊 VIF Analysis",
    "📉 WoE & IV",
    "🔄 PSI",
    "🎯 PiT Models"
])

with tab1:
    render_correlation_tab()

with tab2:
    render_multicollinearity_tab()

with tab3:
    render_vif_tab()

with tab4:
    render_woe_tab()

with tab5:
    render_psi_tab()

with tab6:
    render_pit_models_tab()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <b>Tips:</b> Use the interactive sliders to explore different scenarios | Complete quizzes to test your knowledge</p>
    <p>Built with Streamlit | For Credit Risk Professionals</p>
</div>
""", unsafe_allow_html=True)

"""
Correlation Analysis Module for Credit Risk Analytics
Comprehensive learning module with theory, examples, and interactive analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from plotly.subplots import make_subplots

def render_correlation_tab():
    """Main function to render the correlation analysis tab"""
    
    st.header("📈 Correlation Analysis in Credit Risk")
    
    # Create sub-tabs for organized content
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "📚 Concepts",
        "🏢 Real-Life Scenarios", 
        "🔬 Interactive Analysis",
        "📊 Visualizations",
        "📝 Knowledge Check"
    ])
    
    with subtab1:
        render_concepts()
    
    with subtab2:
        render_scenarios()
    
    with subtab3:
        render_interactive_analysis()
    
    with subtab4:
        render_visualizations()
    
    with subtab5:
        render_quiz()


def render_concepts():
    """Render theoretical concepts and formulas"""
    
    st.subheader("📚 Understanding Correlation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Correlation?
        
        **Correlation** measures the strength and direction of the linear relationship between two variables.
        In credit risk modeling, understanding correlations helps us:
        
        - **Identify relationships** between borrower characteristics and default risk
        - **Avoid redundancy** by detecting highly correlated predictors
        - **Understand risk drivers** and their interactions
        - **Build better models** by selecting complementary features
        
        ### Types of Correlation:
        
        1. **Positive Correlation** (0 to +1): Variables move together
           - Example: Income ↑ → Credit Limit ↑
        
        2. **Negative Correlation** (-1 to 0): Variables move in opposite directions
           - Example: Credit Utilization ↑ → Credit Score ↓
        
        3. **No Correlation** (≈0): No linear relationship
           - Example: Hair color and default risk
        """)
        
    with col2:
        st.info("""
        **📊 Correlation Strength Guide**
        
        |r| = 0.00-0.19: Very weak  
        |r| = 0.20-0.39: Weak  
        |r| = 0.40-0.59: Moderate  
        |r| = 0.60-0.79: Strong  
        |r| = 0.80-1.00: Very strong
        
        **⚠️ Note:** 
        Correlation ≠ Causation
        """)
    
    st.markdown("---")
    
    # Mathematical formulas
    st.subheader("🔢 Mathematical Formulas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Pearson Correlation Coefficient")
        st.latex(r"""
        r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
        {\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
        """)
        
        st.markdown("""
        **Where:**
        - $r$ = Correlation coefficient
        - $x_i, y_i$ = Individual data points
        - $\\bar{x}, \\bar{y}$ = Mean values
        - $n$ = Number of observations
        
        **Range:** -1 ≤ r ≤ +1
        """)
    
    with col2:
        st.markdown("#### Test Statistic")
        st.latex(r"""
        t = r\sqrt{\frac{n-2}{1-r^2}}
        """)
        
        st.markdown("""
        **Where:**
        - $t$ = t-statistic (follows t-distribution)
        - $n$ = Sample size
        - Degrees of freedom: $df = n - 2$
        
        **Use:** Test if correlation is significantly different from zero
        """)
    
    st.markdown("---")
    
    # Step-by-step example
    st.subheader("📝 Step-by-Step Calculation Example")
    
    with st.expander("👉 Click to see detailed calculation"):
        st.markdown("""
        ### Example: Income vs Default Rate
        
        **Dataset:** 5 borrowers
        
        | Borrower | Income ($1000s) | Default (1=Yes) |
        |----------|-----------------|-----------------|
        | A        | 50              | 0               |
        | B        | 65              | 0               |
        | C        | 45              | 1               |
        | D        | 80              | 0               |
        | E        | 40              | 1               |
        
        **Step 1:** Calculate means
        - $\\bar{x}$ (Income) = (50+65+45+80+40)/5 = 56
        - $\\bar{y}$ (Default) = (0+0+1+0+1)/5 = 0.4
        
        **Step 2:** Calculate deviations and products
        
        | Borrower | $(x_i - \\bar{x})$ | $(y_i - \\bar{y})$ | Product |
        |----------|----------------|----------------|---------|
        | A        | -6             | -0.4           | 2.4     |
        | B        | 9              | -0.4           | -3.6    |
        | C        | -11            | 0.6            | -6.6    |
        | D        | 24             | -0.4           | -9.6    |
        | E        | -16            | 0.6            | -9.6    |
        
        **Step 3:** Calculate correlation
        - Numerator: Σ(products) = 2.4 - 3.6 - 6.6 - 9.6 - 9.6 = -27
        - Denominator: √[(-6)² + 9² + (-11)² + 24² + (-16)²] × √[(-0.4)² + (-0.4)² + 0.6² + (-0.4)² + 0.6²]
        - Denominator: √(854) × √(1.2) = 29.22 × 1.095 = 32
        
        **Result:** r = -27/32 = **-0.844**
        
        **Interpretation:** Strong negative correlation - higher income associated with lower default rate ✅
        """)
    
    st.markdown("---")
    
    # When to use
    st.subheader("🎯 When to Use Correlation in Credit Risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **✅ Feature Selection**
        
        - Remove redundant variables
        - Reduce multicollinearity
        - Improve model stability
        """)
    
    with col2:
        st.success("""
        **✅ Risk Factor Analysis**
        
        - Identify key drivers
        - Understand relationships
        - Validate business logic
        """)
    
    with col3:
        st.success("""
        **✅ Model Validation**
        
        - Check expected signs
        - Verify relationships
        - Detect data issues
        """)


def render_scenarios():
    """Render real-life banking scenarios"""
    
    st.subheader("🏢 Real-Life Credit Risk Scenarios")
    
    # Scenario 1: CECL Modeling
    st.markdown("### Scenario 1: CECL Lifetime Loss Modeling")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Context:** A regional bank building a CECL model for auto loans
        
        **Challenge:** Determine which macroeconomic variables to include:
        - Unemployment Rate
        - GDP Growth
        - Housing Price Index (HPI)
        - Consumer Confidence Index
        
        **Analysis:** Correlation matrix revealed:
        - Unemployment Rate vs Default Rate: **r = 0.72** (strong positive)
        - GDP Growth vs Default Rate: **r = -0.68** (strong negative)
        - Unemployment Rate vs GDP Growth: **r = -0.85** (multicollinearity issue!)
        - HPI vs Default Rate: **r = 0.12** (weak, not useful)
        
        **Decision:** Selected Unemployment Rate only (stronger correlation, easier to forecast)
        
        **Business Impact:**
        - Avoided multicollinearity in stress testing
        - Reduced model complexity
        - Improved forecast stability
        - Met regulatory SR 11-7 requirements ✅
        """)
    
    with col2:
        # Create correlation heatmap example
        scenarios_data = pd.DataFrame({
            'Unemp': [1.00, -0.85, 0.42, -0.65],
            'GDP': [-0.85, 1.00, -0.38, 0.70],
            'HPI': [0.42, -0.38, 1.00, -0.22],
            'Default': [-0.65, 0.70, -0.22, 1.00]
        }, index=['Unemp', 'GDP', 'HPI', 'Default'])
        
        fig = px.imshow(scenarios_data,
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       aspect='auto',
                       title='Correlation Matrix')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.warning("""
        **⚠️ Red Flag:**
        Unemp vs GDP: r=-0.85
        Too high! Keep only one.
        """)
    
    st.markdown("---")
    
    # Scenario 2: Credit Card Risk
    st.markdown("### Scenario 2: Credit Card Default Prediction")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.info("""
        **💳 Credit Card Portfolio**
        
        **Bank:** Top 10 US issuer  
        **Portfolio:** $50B outstanding  
        **Goal:** Reduce charge-offs by 15%
        
        **Key Finding:**
        Credit Utilization Ratio showed:
        - **r = 0.58** with default
        - **Threshold:** >80% utilization
        - **Default Rate:** 12.5% vs 2.3% baseline
        
        **Action Taken:**
        - Enhanced monitoring >70%
        - Proactive credit limit reviews
        - Early intervention program
        
        **Result:**
        - Charge-offs reduced by **18%**
        - Saved **$135M annually**
        """)
    
    with col2:
        # Simulate utilization vs default data
        np.random.seed(42)
        utilization = np.random.beta(2, 2, 1000) * 100
        default_prob = 0.02 + (utilization/100)**2 * 0.15 + np.random.normal(0, 0.02, 1000)
        default_prob = np.clip(default_prob, 0, 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=utilization,
            y=default_prob,
            mode='markers',
            marker=dict(size=4, color=default_prob, colorscale='Reds', showscale=True),
            name='Accounts'
        ))
        
        # Add trend line
        z = np.polyfit(utilization, default_prob, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 100, 100)
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            line=dict(color='blue', width=3),
            name='Trend'
        ))
        
        fig.update_layout(
            title='Credit Utilization vs Default Probability',
            xaxis_title='Credit Utilization (%)',
            yaxis_title='Default Probability',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Scenario 3: CCAR Stress Testing
    st.markdown("### Scenario 3: CCAR Stress Testing")
    
    st.markdown("""
    **Context:** Large bank preparing CCAR submission
    
    **Regulatory Requirement (SR 11-7):** Model inputs must be:
    - Economically meaningful
    - Statistically significant
    - Robust under stress scenarios
    
    **Portfolio:** $200B commercial real estate loans
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Variable Analysis**
        
        | Variable | Correlation | p-value |
        |----------|-------------|---------|
        | CRE Price Index | -0.76 | <0.001 |
        | Office Vacancy | +0.64 | <0.001 |
        | Interest Rate | +0.42 | 0.002 |
        | Stock Market | -0.18 | 0.234 |
        """)
    
    with col2:
        st.markdown("""
        **✅ Selected Variables**
        
        1. **CRE Price Index**
           - Strongest correlation
           - Direct relationship
           - Available in stress scenarios
        
        2. **Office Vacancy Rate**
           - Strong positive correlation
           - Lagged effect (6 months)
           - Complements price index
        """)
    
    with col3:
        st.markdown("""
        **❌ Rejected Variables**
        
        1. **Stock Market**
           - Weak correlation (r=0.18)
           - Not significant (p=0.234)
           - Noisy relationship
        
        2. **Interest Rate**
           - Already in CRE prices
           - Would create collinearity
        """)
    
    st.success("""
    **✅ Outcome:** Model approved by Federal Reserve. Passed CCAR with no qualitative objections.
    **Key Success Factor:** Strong correlation analysis supported by economic theory.
    """)


def render_interactive_analysis():
    """Render interactive correlation analysis"""
    
    st.subheader("🔬 Interactive Correlation Analysis")
    
    st.markdown("""
    Explore how different parameters affect correlation strength and significance.
    Adjust the sliders to see real-time changes in the relationship.
    """)
    
    # Scenario selection
    scenario = st.selectbox(
        "Select Scenario:",
        ["Income vs Default Rate", "Credit Utilization vs Default", "Custom Scenario"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎛️ Parameters")
        
        sample_size = st.slider("Sample Size", 50, 1000, 200, 50,
                               help="Larger samples give more reliable correlations")
        
        if scenario == "Custom Scenario":
            true_correlation = st.slider("True Correlation", -1.0, 1.0, -0.6, 0.1,
                                        help="The underlying population correlation")
        else:
            if scenario == "Income vs Default Rate":
                true_correlation = st.slider("Income Effect Strength", -1.0, 0.0, -0.7, 0.1,
                                            help="Negative: higher income → lower default")
            else:  # Credit Utilization
                true_correlation = st.slider("Utilization Effect Strength", 0.0, 1.0, 0.6, 0.1,
                                            help="Positive: higher utilization → higher default")
        
        noise_level = st.slider("Noise Level", 0.0, 2.0, 0.5, 0.1,
                               help="Higher noise makes relationship less clear")
        
        random_seed = st.number_input("Random Seed", 0, 9999, 42,
                                     help="Change for different random samples")
        
        st.markdown("---")
        
        # Display interpretation
        st.markdown("#### 📊 Interpretation Guide")
        
        corr_strength = abs(true_correlation)
        if corr_strength < 0.2:
            strength_label = "Very Weak"
            color = "🔵"
        elif corr_strength < 0.4:
            strength_label = "Weak"
            color = "🟢"
        elif corr_strength < 0.6:
            strength_label = "Moderate"
            color = "🟡"
        elif corr_strength < 0.8:
            strength_label = "Strong"
            color = "🟠"
        else:
            strength_label = "Very Strong"
            color = "🔴"
        
        st.info(f"""
        **Expected Strength:**  
        {color} {strength_label}
        
        **Direction:**  
        {'📉 Negative' if true_correlation < 0 else '📈 Positive' if true_correlation > 0 else '➖ None'}
        """)
    
    with col2:
        # Generate data based on scenario
        np.random.seed(random_seed)
        
        if scenario == "Income vs Default Rate":
            # Income in thousands
            income = np.random.gamma(4, 15, sample_size)
            # Default probability decreases with income
            default_prob = 1 / (1 + np.exp(0.05 * (income - 60)))
            default_prob += np.random.normal(0, noise_level * 0.1, sample_size)
            default = (default_prob > 0.5).astype(int)
            
            # Calculate actual correlation
            actual_corr, p_value = stats.pearsonr(income, default)
            
            x_label = "Income ($1000s)"
            y_label = "Default (1=Yes, 0=No)"
            x_data = income
            y_data = default
            
        elif scenario == "Credit Utilization vs Default":
            # Utilization percentage
            utilization = np.random.beta(2, 2, sample_size) * 100
            # Default probability increases with utilization
            default_prob = 0.02 + (utilization/100)**2 * 0.2
            default_prob += np.random.normal(0, noise_level * 0.05, sample_size)
            default_prob = np.clip(default_prob, 0, 1)
            default = (default_prob > np.random.random(sample_size)).astype(int)
            
            actual_corr, p_value = stats.pearsonr(utilization, default)
            
            x_label = "Credit Utilization (%)"
            y_label = "Default (1=Yes, 0=No)"
            x_data = utilization
            y_data = default
            
        else:  # Custom
            x_data = np.random.randn(sample_size)
            y_data = true_correlation * x_data + noise_level * np.random.randn(sample_size)
            
            actual_corr, p_value = stats.pearsonr(x_data, y_data)
            
            x_label = "Variable X"
            y_label = "Variable Y"
        
        # Calculate confidence interval
        n = sample_size
        stderr = 1.0 / np.sqrt(n - 3)
        z = 0.5 * np.log((1 + actual_corr) / (1 - actual_corr))
        z_crit = 1.96  # 95% confidence
        ci_lower = np.tanh(z - z_crit * stderr)
        ci_upper = np.tanh(z + z_crit * stderr)
        
        # Create scatter plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Scatter Plot with Trend', 'Distribution of X', 
                          'Distribution of Y', 'Residuals'),
            specs=[[{"rowspan": 2}, {}],
                   [None, {}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # Main scatter plot
        fig.add_trace(
            go.Scatter(x=x_data, y=y_data, mode='markers',
                      marker=dict(size=6, color=y_data, colorscale='Viridis', 
                                showscale=True, colorbar=dict(x=0.45)),
                      name='Data',
                      text=[f'X: {x:.2f}<br>Y: {y:.2f}' for x, y in zip(x_data, y_data)],
                      hovertemplate='%{text}<extra></extra>'),
            row=1, col=1
        )
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
        fig.add_trace(
            go.Scatter(x=x_trend, y=p(x_trend), mode='lines',
                      line=dict(color='red', width=3, dash='dash'),
                      name='Trend Line'),
            row=1, col=1
        )
        
        # Distribution of X
        fig.add_trace(
            go.Histogram(x=x_data, nbinsx=30, name='X Distribution',
                        marker=dict(color='lightblue')),
            row=1, col=2
        )
        
        # Distribution of Y
        fig.add_trace(
            go.Histogram(x=y_data, nbinsx=30, name='Y Distribution',
                        marker=dict(color='lightgreen')),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text=x_label, row=1, col=1)
        fig.update_yaxes(title_text=y_label, row=1, col=1)
        fig.update_xaxes(title_text=x_label, row=1, col=2)
        fig.update_xaxes(title_text=y_label, row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Correlation Analysis")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        st.markdown("#### 📈 Statistical Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Correlation (r)", f"{actual_corr:.4f}",
                     delta=f"{actual_corr - true_correlation:.4f}" if scenario != "Custom Scenario" else None)
        
        with col2:
            st.metric("p-value", f"{p_value:.4e}",
                     delta="Significant ✅" if p_value < 0.05 else "Not Sig. ❌")
        
        with col3:
            st.metric("Sample Size", f"{n}")
        
        with col4:
            # Calculate t-statistic
            t_stat = actual_corr * np.sqrt(n - 2) / np.sqrt(1 - actual_corr**2)
            st.metric("t-statistic", f"{t_stat:.2f}")
        
        # Detailed interpretation
        st.markdown("#### 💡 Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if p_value < 0.001:
                sig_level = "p < 0.001 (Highly Significant)"
                sig_color = "success"
            elif p_value < 0.01:
                sig_level = "p < 0.01 (Very Significant)"
                sig_color = "success"
            elif p_value < 0.05:
                sig_level = "p < 0.05 (Significant)"
                sig_color = "success"
            else:
                sig_level = f"p = {p_value:.3f} (Not Significant)"
                sig_color = "error"
            
            if sig_color == "success":
                st.success(f"""
                **✅ Statistical Significance**
                
                {sig_level}
                
                We can reject the null hypothesis that there is no correlation.
                The relationship is statistically reliable.
                """)
            else:
                st.error(f"""
                **❌ Not Significant**
                
                {sig_level}
                
                Cannot reject null hypothesis. The correlation may be due to chance.
                Increase sample size or reduce noise.
                """)
        
        with col2:
            st.info(f"""
            **📊 Confidence Interval (95%)**
            
            [{ci_lower:.4f}, {ci_upper:.4f}]
            
            We are 95% confident the true correlation falls within this range.
            
            **Coefficient of Determination:**
            R² = {actual_corr**2:.4f} ({actual_corr**2*100:.2f}%)
            
            This means {actual_corr**2*100:.1f}% of variance in Y is explained by X.
            """)
        
        # Business recommendations
        st.markdown("#### 💼 Business Recommendations")
        
        if abs(actual_corr) > 0.7 and p_value < 0.05:
            st.success(f"""
            **✅ Strong Relationship Detected**
            
            **Recommended Actions:**
            1. Include this variable in your credit risk model
            2. Monitor this relationship for stability over time
            3. Use for early warning indicators
            4. Consider for limit management decisions
            
            **Expected Impact:** High predictive power for risk assessment
            """)
        elif abs(actual_corr) > 0.4 and p_value < 0.05:
            st.warning(f"""
            **⚠️ Moderate Relationship**
            
            **Recommended Actions:**
            1. Consider including with other variables
            2. Combine with domain knowledge
            3. Test in multivariate models
            4. Monitor for changes in correlation strength
            
            **Expected Impact:** Moderate improvement in model performance
            """)
        else:
            st.error(f"""
            **❌ Weak or Insignificant Relationship**
            
            **Recommended Actions:**
            1. Do not use as standalone predictor
            2. Investigate if non-linear relationship exists
            3. Check for data quality issues
            4. Consider alternative variables
            
            **Risk:** May not add value to risk models
            """)


def render_visualizations():
    """Render advanced visualizations"""
    
    st.subheader("📊 Advanced Correlation Visualizations")
    
    st.markdown("""
    Explore different visualization techniques for understanding correlations in credit risk data.
    """)
    
    # Generate sample credit risk data
    np.random.seed(42)
    n_samples = 500
    
    # Create correlated features
    income = np.random.gamma(4, 15, n_samples)
    age = np.random.normal(45, 12, n_samples)
    credit_score = 300 + 400 / (1 + np.exp(-0.05 * (income - 50))) + np.random.normal(0, 30, n_samples)
    credit_score = np.clip(credit_score, 300, 850)
    dti_ratio = np.maximum(10, 60 - 0.3 * income + np.random.normal(0, 8, n_samples))
    utilization = np.random.beta(2, 3, n_samples) * 100
    default = ((credit_score < 600) | (dti_ratio > 45) | (utilization > 80)).astype(int)
    
    df = pd.DataFrame({
        'Income': income,
        'Age': age,
        'Credit_Score': credit_score,
        'DTI_Ratio': dti_ratio,
        'Utilization': utilization,
        'Default': default
    })
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type:",
        ["Correlation Matrix Heatmap", "Pair Plot", "3D Scatter Plot", "Correlation Network"]
    )
    
    if viz_type == "Correlation Matrix Heatmap":
        st.markdown("### Correlation Matrix Heatmap")
        st.markdown("Shows all pairwise correlations in a single view. Useful for identifying multicollinearity.")
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Correlation Matrix - Credit Risk Variables',
            xaxis_title='Variables',
            yaxis_title='Variables',
            height=600,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight key findings
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ Key Positive Correlations:**
            - Income ↔ Credit Score: Strong relationship
            - Expected and desirable for modeling
            """)
        
        with col2:
            st.warning("""
            **⚠️ Key Negative Correlations:**
            - Income ↔ DTI Ratio: Makes sense
            - Credit Score ↔ Default: Strong predictor
            """)
    
    elif viz_type == "Pair Plot":
        st.markdown("### Pair Plot - Scatter Matrix")
        st.markdown("Shows relationships between all variable pairs. Diagonal shows distributions.")
        
        # Select variables for pair plot
        selected_vars = st.multiselect(
            "Select variables to plot:",
            ['Income', 'Credit_Score', 'DTI_Ratio', 'Utilization'],
            default=['Income', 'Credit_Score', 'DTI_Ratio']
        )
        
        if len(selected_vars) >= 2:
            fig = px.scatter_matrix(
                df,
                dimensions=selected_vars,
                color='Default',
                color_continuous_scale='RdYlGn_r',
                title='Pair Plot - Credit Risk Variables',
                height=800
            )

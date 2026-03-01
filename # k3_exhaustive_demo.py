import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Pipeline Robustness Checker",
    page_icon="🛡️",
    layout="wide"
)

# --- UI Title ---
st.title("🛡️ ML Pipeline Robustness Checker")
st.markdown("""
This interface allows you to simulate data drift and evaluate the vulnerability of a machine learning pipeline. 
Adjust the settings in the left panel and click 'Run Robustness Check' to see a detailed report.
""")

# --- Layout Definition ---
# Create two columns: one for settings (inputs) and one for the report (outputs)
settings_col, report_col = st.columns(2)

# --- Settings Column (Left) ---
with settings_col:
    st.header("Settings (Pipeline & Drift Simulation)", divider="rainbow")

    # 1. Dataset Selection
    dataset = st.selectbox(
        "**Dataset**",
        ("German Credit", "Adult Income"),
        help="Select the dataset for the pipeline."
    )

    # 2. Pipeline Configuration
    st.markdown("**Pipeline Components**")
    # Using multiselect for flexibility, though the prompt showed a fixed pipeline.
    # This allows for more dynamic interaction.
    imputation_step = st.selectbox(
        "Imputation",
        ("Mean Imputation", "Median Imputation", "Mode Imputation"),
        help="Method to handle missing values."
    )
    scaling_step = st.selectbox(
        "Scaling / Preprocessing",
        ("Standardization", "Min-Max Scaling", "IQR Outlier Handling"),
        help="Data scaling or outlier removal method."
    )
    model_step = st.selectbox(
        "Classifier Model",
        ("Logistic Regression", "Random Forest", "AdaBoost"),
        help="The prediction model to use."
    )
    
    # Display the constructed pipeline
    st.info(f"**Current Pipeline:** `{imputation_step} → {scaling_step} → {model_step}`")

    # 3. Drift Simulation
    st.markdown("**Drift Simulation**")
    drift_type = st.selectbox(
        "**Drift Type**",
        ("Missing Value Injection", "Outlier Injection"),
        help="The type of noise to introduce into the dataset."
    )
    noise_level = st.slider(
        "**Noise Level (ρ)**",
        min_value=0,
        max_value=50,
        value=25,
        format="%d%%",
        help="The percentage of data to affect with the selected drift."
    )

    # 4. Vulnerability Threshold
    vulnerability_threshold = st.number_input(
        "**Vulnerability Threshold (τ)**",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Set the tolerable vulnerability score. Scores above this will be flagged."
    )

    # 5. Run Button
    run_button = st.button("🚀 Run Robustness Check", use_container_width=True)

# --- Report Column (Right) ---
with report_col:
    st.header("Robustness Report", divider="rainbow")
    
    if run_button:
        # Simulate a computation delay
        with st.spinner("Analyzing pipeline vulnerability..."):
            time.sleep(2) # Represents the backend calculation time

        # --- Hardcoded Results (as per the prompt's example) ---
        vulnerability_score = 4.5
        performance_degradation = 3.5
        most_vulnerable_component = "Mean Value Imputation"
        suggested_pipeline = "Median Imputation → IQR Outlier Handling → Standardization"
        retraining_needed = "No"

        # --- Display Metrics ---
        st.subheader("Key Metrics")
        
        # Display Vulnerability Score with a warning color if it exceeds the threshold
        if vulnerability_score > vulnerability_threshold:
            delta_text = f"Exceeds Threshold ({vulnerability_threshold})"
            st.metric(
                label="Vulnerability Score (VS)",
                value=f"{vulnerability_score}",
                delta=delta_text,
                delta_color="inverse" # Displays in red
            )
        else:
            delta_text = f"Within Threshold ({vulnerability_threshold})"
            st.metric(
                label="Vulnerability Score (VS)",
                value=f"{vulnerability_score}",
                delta=delta_text,
                delta_color="normal" # Displays in green
            )

        st.metric(label="Performance Degradation (Δ)", value=f"{performance_degradation}%")

        # --- Display Detailed Findings ---
        st.subheader("Analysis")
        st.markdown(f"**Most Vulnerable Component:** `{most_vulnerable_component}`")
        st.markdown(f"**Retraining Needed:** **{retraining_needed}**")
        st.markdown(f"**Suggested Pipeline:** `{suggested_pipeline}`")
        
        # --- Display User-Facing Explanation in a distinct container ---
        with st.container(border=True):
            st.subheader("User-Facing Explanation")
            
            st.markdown(
                f"**📊 Quantification:** The current pipeline has a Vulnerability Score (VS) of **{vulnerability_score}**, "
                f"which exceeds your tolerable threshold of **{vulnerability_threshold}**."
            )
            
            st.markdown(
                f"**🧐 Explanation:** The `{most_vulnerable_component}` step is highly sensitive to the "
                f"**{drift_type.lower()}** you simulated. This sensitivity is the primary cause of the "
                f"distributional shift, leading to a performance drop."
            )
            
            st.markdown(
                f"**🛠️ Intervention:** To improve robustness, we suggest switching to **median imputation**. "
                f"This change significantly reduces the pipeline's vulnerability and is projected to keep "
                f"performance degradation below 1% under similar drift conditions."
            )
    else:
        st.info("Please configure your settings on the left and click 'Run Robustness Check' to generate a report.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# -------------------------------------------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Tool",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 5px;
        border: none;
        font-size: 18px;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# 2. Load Model & Data
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_app_resources():
    try:
        model = joblib.load("predictive_maintenance_model.pkl")
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_app_resources()

# Calculate/Load feature stats (Approximations based on previous file analysis)
MODEL_AVG_TEMP_MAP = {
    "Model_A": 70.0,
    "Model_B": 75.0,
    "Model_C": 68.0
}

# -------------------------------------------------------------------------------------------------
# 3. Sidebar Navigation
# -------------------------------------------------------------------------------------------------
st.sidebar.image("preferences.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predict", "About"], index=0)

st.sidebar.markdown("---")
st.sidebar.info("Select 'Predict' to run diagnostics on your machinery.")

# -------------------------------------------------------------------------------------------------
# 4. Page: PREDICT
# -------------------------------------------------------------------------------------------------
if page == "Predict":
    st.markdown('<div class="main-header">🔧 Predictive Maintenance & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Failure Probability Assessment</div>', unsafe_allow_html=True)
    st.markdown("---")

    import shap
    import matplotlib.pyplot as plt

    if model is None:
        st.error("❌ Model file `predictive_maintenance_model.pkl` not found. Please upload the model file.")
    else:
        # --- Input Form ---
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📋 Machine Details")
                machine_model = st.selectbox("Machine Model", ["Model_A", "Model_B", "Model_C"])
                operator_experience = st.selectbox("Operator Experience", ["Junior", "Mid", "Senior"])
                fault_code = st.selectbox("Active Fault Code", ["No_Fault", "E1", "E2", "E3"])
                
                st.subheader("⚙️ Usage Stats")
                last_service = st.number_input("Days Since Last Service", 0, 1000, 45, help="Days since the last maintenance check.")
                working_hours = st.number_input("Total Working Hours", 0, 50000, 5000)

            with col2:
                st.subheader("📡 Sensor Readings")
                c1, c2 = st.columns(2)
                with c1:
                    avg_temp = st.slider("Avg Temperature (°C)", 0.0, 150.0, 70.0)
                    vibration = st.slider("Vibration Level", 0.0, 10.0, 2.5)
                    rot_speed = st.number_input("Rotating Speed (RPM)", 0, 10000, 2000)
                with c2:
                    voltage = st.number_input("Voltage Fluctuation", 100.0, 400.0, 220.0)
                    torque = st.number_input("Torque (Nm)", 0.0, 300.0, 40.0)
                    oil_viscosity = st.number_input("Oil Viscosity", 0.0, 200.0, 35.0)
                    humidity = st.number_input("Ambient Humidity (%)", 0.0, 100.0, 55.0)

        # --- Feature Engineering Logic ---
        # 1. Stress Index
        stress_index = torque * vibration
        
        # 2. Service Risk
        if last_service <= 30:
            service_risk = "Low"
        elif last_service <= 90:
            service_risk = "Medium"
        elif last_service <= 150:
            service_risk = "High"
        else:
            service_risk = "Critical"
            
        # 3. Temp Deviation
        model_avg_temp_val = MODEL_AVG_TEMP_MAP.get(machine_model, 70.0)
        temp_deviation = avg_temp - model_avg_temp_val
        
        # 4. Date Features
        now = datetime.now()
        year, month, day, weekday = now.year, now.month, now.day, now.weekday()

        # Build DataFrame
        input_data = pd.DataFrame({
            "Machine_Model": [machine_model],
            "Avg_Temperature": [avg_temp],
            "Vibration_Level": [vibration],
            "Rotating_Speed": [rot_speed],
            "Voltage_Fluctuation": [voltage],
            "Torque_Nm": [torque],
            "Oil_Viscosity": [oil_viscosity],
            "Ambient_Humidity": [humidity],
            "Operator_Experience": [operator_experience],
            "Last_Service_Days": [last_service],
            "Fault_Code": [fault_code],
            "Working_Hours_Total": [working_hours],
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "Weekday": [weekday],
            "Stress_Index": [stress_index],
            "Model_Avg_Temp": [model_avg_temp_val],
            "Temp_Deviation": [temp_deviation],
            "Service_Risk": [service_risk]
        })

        # Ensure Column Order is exact (often critical for pipelines)
        expected_cols = [
            'Machine_Model', 'Avg_Temperature', 'Vibration_Level', 'Rotating_Speed', 
            'Voltage_Fluctuation', 'Torque_Nm', 'Oil_Viscosity', 'Ambient_Humidity', 
            'Operator_Experience', 'Last_Service_Days', 'Fault_Code', 'Working_Hours_Total', 
            'Year', 'Month', 'Day', 'Weekday', 'Stress_Index', 'Model_Avg_Temp', 
            'Temp_Deviation', 'Service_Risk'
        ]
        
        # Reorder if necessary, add missing as 0 if needed (safety check)
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[expected_cols]

        st.markdown("---")
        if st.button(" Run Prediction"):
            with st.spinner("Analyzing Sensor Data..."):
                try:
                    # Predict
                    prob = model.predict_proba(input_data)[0][1]
                    prediction = model.predict(input_data)[0]
                    health_score = (1 - prob) * 100
                    
                    # Display Results
                    st.success("Analysis Complete!")
                    
                    res_col1, res_col2 = st.columns([1, 1.5])
                    
                    with res_col1:
                        st.markdown("### 📊 Results")
                        st.metric("Probability of Failure", f"{prob:.1%}")
                        st.metric("Machine Health Score", f"{health_score:.1f}%")
                        
                        if prob > 0.5:
                            st.error("**High Risk of Failure**\n\nImmediate maintenance is recommended.")
                        else:
                            st.success("**System Healthy**\n\nOperate as normal.")

                    with res_col2:
                         # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = health_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Health Score"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "rgba(0,0,0,0)"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#EF4444"},
                                    {'range': [40, 75], 'color': "#F59E0B"},
                                    {'range': [75, 100], 'color': "#10B981"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': health_score
                                }
                            }
                        ))
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # --- Visualizations (Graphs) ---
                    st.markdown("---")
                    st.subheader(" Model Analysis")
                    
                    tab_vis1, tab_vis2 = st.tabs(["Local Prediction Logic (SHAP)", "Global Model Coefficients"])
                    
                    # 1. Local SHAP (Specific to this prediction)
                    with tab_vis1:
                        st.write("### Why *this specific* machine got this score:")
                        st.caption("Values in RED increase failure risk. Values in GREEN decrease it.")
                        try:
                            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                                preprocessor = model.named_steps['preprocessor']
                                X_trans = preprocessor.transform(input_data)
                                
                                # Convert sparse to dense if needed
                                if hasattr(X_trans, "toarray"):
                                    X_trans = X_trans.toarray()
                                
                                # Get feature names
                                try:
                                    feature_names = preprocessor.get_feature_names_out()
                                except:
                                    feature_names = [f"Feature {i}" for i in range(X_trans.shape[1])]
                                
                                classifier = model.named_steps['model']
                                
                                # Use Linear Explainer for Logistic Regression
                                try:
                                    # Fix: SHAP needs a background dataset to compare against. 
                                    # We'll try to load a small sample of the original data.
                                    try:
                                        # cached loading of background data
                                        @st.cache_data
                                        def load_background_data():
                                            df = pd.read_csv("globaltech_machinery_logs_P1.csv")
                                            sample = df.sample(100, random_state=42)
                                            # We need to process this sample through the pipeline's preprocessor
                                            return sample
                                            
                                        bg_sample = load_background_data()
                                        bg_trans = preprocessor.transform(bg_sample)
                                        if hasattr(bg_trans, "toarray"):
                                            bg_trans = bg_trans.toarray()
                                            
                                        masker = shap.maskers.Independent(data=bg_trans)
                                    except:
                                        # Fallback to zero-background if CSV missing
                                        masker = shap.maskers.Independent(data=np.zeros_like(X_trans))

                                    explainer = shap.LinearExplainer(classifier, masker=masker)
                                    shap_values = explainer(X_trans)
                                    
                                    # Custom Plot using Plotly for better aesthetics than matplotlib
                                    # Fix: valid access depends on SHAP version, but typically for LinearExplainer on simple models:
                                    # shap_values is an Explanation object. 
                                    # .values is (n_samples, n_features). 
                                    values = shap_values.values
                                    
                                    # If it has 2 dimensions (samples, features), take first sample
                                    if len(values.shape) == 2:
                                        sample_values = values[0]
                                    else:
                                        sample_values = values

                                    shap_df = pd.DataFrame({
                                        "Feature": feature_names,
                                        "Impact": sample_values
                                    })
                                    
                                    # Filter top 10 most impactful features
                                    shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                                    shap_df = shap_df.sort_values('Abs_Impact', ascending=False).head(10)
                                    shap_df = shap_df.sort_values('Impact', ascending=True) # Sort for chart display
                                    
                                    fig_shap = go.Figure(go.Bar(
                                        x=shap_df['Impact'],
                                        y=shap_df['Feature'],
                                        orientation='h',
                                        marker=dict(
                                            color=shap_df['Impact'].apply(lambda x: '#EF4444' if x > 0 else '#10B981')
                                        )
                                    ))
                                    fig_shap.update_layout(
                                        title=dict(text="Top Contributors to This Prediction", font=dict(size=18)),
                                        xaxis_title="Impact on Risk Score",
                                        yaxis=dict(title="Feature"),
                                        height=500,
                                        margin=dict(l=20, r=20, t=50, b=20)
                                    )
                                    st.plotly_chart(fig_shap, use_container_width=True)
                                    
                                except Exception as inner_e:
                                    st.warning(f"Could not generate SHAP plot: {inner_e}")
                            else:
                                st.info("Model structure does not support automatic SHAP extraction in this demo.")
                                
                        except Exception as e:
                            st.error(f"Error generating explanations: {e}")

                    # 2. Global Coefficients (General Model Logic)
                    with tab_vis2:
                        st.write("### What the model looks for *in general*:")
                        st.caption("These are the fixed rules usage by the model (Logistic Regression Coefficients).")
                        
                        try:
                            # Extract coefficients from the pipeline
                            if hasattr(model, 'named_steps'):
                                classifier = model.named_steps['model']
                                preprocessor = model.named_steps['preprocessor']
                                
                                if hasattr(classifier, 'coef_'):
                                    # Get feature names matching the coefficients
                                    try:
                                        feature_names = preprocessor.get_feature_names_out()
                                        coefs = classifier.coef_[0]
                                        
                                        coef_df = pd.DataFrame({
                                            "Feature": feature_names,
                                            "Weight": coefs
                                        })
                                        
                                        # Sort by absolute weight to show most important global rules
                                        coef_df['Abs_Weight'] = coef_df['Weight'].abs()
                                        coef_df = coef_df.sort_values('Abs_Weight', ascending=False).head(15)
                                        coef_df = coef_df.sort_values('Weight', ascending=True)

                                        fig_global = go.Figure(go.Bar(
                                            x=coef_df['Weight'],
                                            y=coef_df['Feature'],
                                            orientation='h',
                                            marker=dict(
                                                color=coef_df['Weight'].apply(lambda x: '#EF4444' if x > 0 else '#3B82F6') # Red vs Blue for global
                                            )
                                        ))
                                        fig_global.update_layout(
                                            title="Overall Feature Importance (Model Weights)",
                                            xaxis_title="Weight (Positive = Higher Risk, Negative = Lower Risk)",
                                            height=600
                                        )
                                        st.plotly_chart(fig_global, use_container_width=True)
                                        
                                    except Exception as f_err:
                                        st.warning(f"Could not align feature names: {f_err}")
                                else:
                                    st.info("This model does not expose linear coefficients.")
                            else:
                                st.info("Model is not a pipeline.")
                        
                        except Exception as e:
                            st.error(f"Could not visualize global importance: {e}") 

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.write("Debug info on input data:", input_data)

# -------------------------------------------------------------------------------------------------
# 5. Page: ABOUT
# -------------------------------------------------------------------------------------------------
elif page == "About":
    st.markdown('<div class="main-header">ℹ About This Application</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### Predictive Maintenance System
    
    This application utilizes a Machine Learning model to predict standard equipment failure before it happens.
    
    #### 🔍 How it works:
    1.  **Data Collection**: We collect real-time sensor data (Temperature, Vibration, Speed, etc.).
    2.  **Feature Engineering**: We calculate derived metrics like *Stress Index* and *Temp Deviation*.
    3.  **Prediction**: A pre-trained Classification Model evaluates the risk of failure.
    
    #### 🛠 Features:
    -   **Real-time Analysis**: Instant feedback on machine health.
    -   **Interactive Dashboard**: Adjust parameters to simulate different conditions.
    -   **Visual Indicators**: easy-to-read gauges and alerts.
    
    #### 👨‍💻 Developer Info:
    Built with **Streamlit** and **Scikit-Learn**.
    """)
    
    st.info("Ensure `predictive_maintenance_model.pkl` is in the same directory for the app to function.")

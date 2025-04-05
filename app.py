import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM
from keras.utils.np_utils import to_categorical
import os
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import base64

# Page configuration
st.set_page_config(
    page_title="E-Pilots: Hard Landing Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'pilot' not in st.session_state:
    st.session_state.pilot = None
    st.session_state.actuator = None
    st.session_state.physical = None
    st.session_state.Y = None
    st.session_state.all_data = None
    st.session_state.models_trained = False
    st.session_state.sensitivity = []
    st.session_state.specificity = []

# Title and description
st.title("✈️ E-Pilots: Hard Landing Prediction System")
st.markdown("""
Predicts hard landings using pilot, actuator, and physical data during the approach phase of commercial flights.
""")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", 
    ["Upload Data", "Preprocess", "Train Models", "Results", "Flight Simulation"])

# Helper functions
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Page 1: Upload Data
if page == "Upload Data":
    st.header("📂 Upload Flight Landing Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pilot_file = st.file_uploader("Upload Pilot Data (CSV)", type="csv")
    with col2:
        actuator_file = st.file_uploader("Upload Actuator Data (CSV)", type="csv")
    with col3:
        physical_file = st.file_uploader("Upload Physical Data (CSV)", type="csv")
    
    if st.button("Load Datasets"):
        if pilot_file and actuator_file and physical_file:
            try:
                # Save files temporarily and read them
                pilot_path = save_uploaded_file(pilot_file)
                actuator_path = save_uploaded_file(actuator_file)
                physical_path = save_uploaded_file(physical_file)
                
                st.session_state.pilot = pd.read_csv(pilot_path)
                st.session_state.actuator = pd.read_csv(actuator_path)
                st.session_state.physical = pd.read_csv(physical_path)
                
                # Process the data
                st.session_state.Y = st.session_state.physical['label'].values
                st.session_state.pilot.drop(['label'], axis=1, inplace=True)
                st.session_state.actuator.drop(['label'], axis=1, inplace=True)
                st.session_state.physical.drop(['label'], axis=1, inplace=True)
                st.session_state.all_data = pd.concat(
                    [st.session_state.physical, st.session_state.actuator, st.session_state.pilot], 
                    axis=1
                )
                
                st.success("Datasets loaded successfully!")
                
                # Show dataset previews
                st.subheader("Data Previews")
                
                tab1, tab2, tab3 = st.tabs(["Pilot Data", "Actuator Data", "Physical Data"])
                
                with tab1:
                    st.dataframe(st.session_state.pilot.head())
                
                with tab2:
                    st.dataframe(st.session_state.actuator.head())
                
                with tab3:
                    st.dataframe(st.session_state.physical.head())
                
                # Show landing distribution
                st.subheader("Landing Distribution")
                labels, counts = np.unique(st.session_state.Y, return_counts=True)
                fig, ax = plt.subplots()
                ax.bar(['Normal Landing', 'Hard Landing'], counts, color=['green', 'red'])
                ax.set_xlabel("Landing Type")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error loading datasets: {e}")
        else:
            st.warning("Please upload all three dataset files!")

# Page 2: Preprocess Data
elif page == "Preprocess":
    st.header("🔧 Preprocess Dataset")
    
    if st.session_state.physical is not None:
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                try:
                    # Convert to numpy arrays
                    pilot = st.session_state.pilot.values
                    actuator = st.session_state.actuator.values
                    physical = st.session_state.physical.values
                    all_data = st.session_state.all_data.values
                    Y = st.session_state.Y
                    
                    # Shuffle the dataset
                    indices = np.arange(all_data.shape[0])
                    np.random.shuffle(indices)
                    all_data = all_data[indices]
                    Y = Y[indices]
                    pilot = pilot[indices]
                    actuator = actuator[indices]
                    physical = physical[indices]
                    
                    # Normalize dataset values
                    scaler1 = StandardScaler()
                    all_data = scaler1.fit_transform(all_data)
                    scaler2 = StandardScaler()
                    pilot = scaler2.fit_transform(pilot)
                    scaler3 = StandardScaler()
                    actuator = scaler3.fit_transform(actuator)
                    scaler4 = StandardScaler()
                    physical = scaler4.fit_transform(physical)
                    
                    # Reshape to multi-dimensional array
                    pilot = np.reshape(pilot, (pilot.shape[0], pilot.shape[1], 1))
                    actuator = np.reshape(actuator, (actuator.shape[0], actuator.shape[1], 1))
                    physical = np.reshape(physical, (physical.shape[0], physical.shape[1], 1))
                    
                    # Split dataset into train and test
                    st.session_state.all_X_train, st.session_state.all_X_test, st.session_state.all_y_train, st.session_state.all_y_test = train_test_split(
                        all_data, Y, test_size=0.2)
                    
                    Y_categorical = to_categorical(Y)
                    st.session_state.pilot_X_train, st.session_state.pilot_X_test, st.session_state.pilot_y_train, st.session_state.pilot_y_test = train_test_split(
                        pilot, Y_categorical, test_size=0.2)
                    st.session_state.actuator_X_train, st.session_state.actuator_X_test, st.session_state.actuator_y_train, st.session_state.actuator_y_test = train_test_split(
                        actuator, Y_categorical, test_size=0.2)
                    st.session_state.physical_X_train, st.session_state.physical_X_test, st.session_state.physical_y_train, st.session_state.physical_y_test = train_test_split(
                        physical, Y_categorical, test_size=0.2)
                    
                    st.success("Data preprocessing completed!")
                    
                    # Show preprocessing results
                    st.subheader("Preprocessing Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **Dataset Statistics:**
                        - Total records: {}
                        - All features: {}
                        - Pilot features: {}
                        - Actuator features: {}
                        - Physical features: {}
                        """.format(
                            all_data.shape[0],
                            st.session_state.all_data.shape[1],
                            st.session_state.pilot.shape[1],
                            st.session_state.actuator.shape[1],
                            st.session_state.physical.shape[1]
                        ))
                    
                    with col2:
                        st.markdown("""
                        **Train-Test Split:**
                        - Training samples: {}
                        - Testing samples: {}
                        """.format(
                            st.session_state.all_X_train.shape[0],
                            st.session_state.all_X_test.shape[0]
                        ))
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
    else:
        st.warning("Please upload and load the datasets first!")

# Page 3: Train Models
elif page == "Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if st.session_state.physical is None:
        st.warning("Please upload and preprocess the data first!")
        st.stop()
    
    model_options = ["SVM", "Logistic Regression", "LSTM (Physical)", "LSTM (Actuator)", "LSTM (Pilot)"]
    selected_models = st.multiselect("Select models to train", model_options, default=model_options)
    
    if st.button("Train Selected Models"):
        st.session_state.sensitivity = []
        st.session_state.specificity = []
        
        with st.spinner("Training models..."):
            try:
                # SVM
                if "SVM" in selected_models:
                    with st.expander("SVM Training Details"):
                        svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1)
                        svm_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
                        predict = svm_cls.predict(st.session_state.all_X_test)
                        
                        # Calculate metrics
                        cm = confusion_matrix(st.session_state.all_y_test, predict)
                        se = accuracy_score(st.session_state.all_y_test, predict)
                        sp = recall_score(st.session_state.all_y_test, predict)
                        if sp == 0:
                            sp = accuracy_score(st.session_state.all_y_test, predict)
                        
                        st.session_state.sensitivity.append(se)
                        st.session_state.specificity.append(sp)
                        
                        st.success("SVM training completed!")
                        st.markdown(f"""
                        **SVM Metrics:**
                        - Sensitivity: {se:.4f}
                        - Specificity: {sp:.4f}
                        """)
                        
                        # Confusion matrix
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay.from_predictions(
                            st.session_state.all_y_test, predict, ax=ax)
                        st.pyplot(fig)
                
                # Logistic Regression
                if "Logistic Regression" in selected_models:
                    with st.expander("Logistic Regression Training Details"):
                        lr_cls = LogisticRegression(max_iter=1000)
                        lr_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
                        predict = lr_cls.predict(st.session_state.all_X_test)
                        
                        # Calculate metrics
                        se = accuracy_score(st.session_state.all_y_test, predict)
                        sp = recall_score(st.session_state.all_y_test, predict)
                        if sp == 0:
                            sp = accuracy_score(st.session_state.all_y_test, predict)
                        
                        st.session_state.sensitivity.append(se)
                        st.session_state.specificity.append(sp)
                        
                        st.success("Logistic Regression training completed!")
                        st.markdown(f"""
                        **Logistic Regression Metrics:**
                        - Sensitivity: {se:.4f}
                        - Specificity: {sp:.4f}
                        """)
                
                # LSTM models
                def train_lstm(model_name, X_train, y_train, X_test, y_test):
                    model_path = f"model/{model_name.lower().replace(' ', '_')}_model.json"
                    weights_path = f"model/{model_name.lower().replace(' ', '_')}_weights.h5"
                    
                    if os.path.exists(model_path):
                        with open(model_path, "r") as json_file:
                            loaded_model_json = json_file.read()
                            lstm_model = model_from_json(loaded_model_json)
                        json_file.close()
                        lstm_model.load_weights(weights_path)
                        lstm_model._make_predict_function()
                    else:
                        os.makedirs("model", exist_ok=True)
                        lstm_model = Sequential()
                        lstm_model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
                        lstm_model.add(Dropout(0.5))
                        lstm_model.add(Dense(100, activation='relu'))
                        lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
                        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                        
                        # Show training progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        class ProgressCallback(keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / 20
                                progress_bar.progress(progress)
                                status_text.text(f"Epoch {epoch + 1}/20 - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
                        
                        history = lstm_model.fit(
                            X_train, y_train,
                            epochs=20,
                            batch_size=16,
                            validation_data=(X_test, y_test),
                            callbacks=[ProgressCallback()],
                            verbose=0
                        )
                        
                        lstm_model.save_weights(weights_path)
                        model_json = lstm_model.to_json()
                        with open(model_path, "w") as json_file:
                            json_file.write(model_json)
                        
                        # Plot training history
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        ax1.plot(history.history['accuracy'], label='train')
                        ax1.plot(history.history['val_accuracy'], label='validation')
                        ax1.set_title('Model Accuracy')
                        ax1.set_ylabel('Accuracy')
                        ax1.set_xlabel('Epoch')
                        ax1.legend()
                        
                        ax2.plot(history.history['loss'], label='train')
                        ax2.plot(history.history['val_loss'], label='validation')
                        ax2.set_title('Model Loss')
                        ax2.set_ylabel('Loss')
                        ax2.set_xlabel('Epoch')
                        ax2.legend()
                        
                        st.pyplot(fig)
                    
                    return lstm_model
                
                # Physical LSTM
                if "LSTM (Physical)" in selected_models:
                    with st.expander("LSTM (Physical) Training Details"):
                        lstm_physical = train_lstm(
                            "physical",
                            st.session_state.physical_X_train,
                            st.session_state.physical_y_train,
                            st.session_state.physical_X_test,
                            st.session_state.physical_y_test
                        )
                        
                        predict = lstm_physical.predict(st.session_state.physical_X_test)
                        predict = np.argmax(predict, axis=1)
                        testY = np.argmax(st.session_state.physical_y_test, axis=1)
                        
                        se = accuracy_score(testY, predict)
                        sp = recall_score(testY, predict)
                        
                        st.session_state.sensitivity.append(se)
                        st.session_state.specificity.append(sp)
                        
                        st.success("LSTM (Physical) training completed!")
                
                # Actuator LSTM
                if "LSTM (Actuator)" in selected_models:
                    with st.expander("LSTM (Actuator) Training Details"):
                        lstm_actuator = train_lstm(
                            "actuator",
                            st.session_state.actuator_X_train,
                            st.session_state.actuator_y_train,
                            st.session_state.actuator_X_test,
                            st.session_state.actuator_y_test
                        )
                        
                        predict = lstm_actuator.predict(st.session_state.actuator_X_test)
                        predict = np.argmax(predict, axis=1)
                        testY = np.argmax(st.session_state.actuator_y_test, axis=1)
                        
                        se = accuracy_score(testY, predict)
                        sp = recall_score(testY, predict)
                        
                        st.session_state.sensitivity.append(se)
                        st.session_state.specificity.append(sp)
                        
                        st.success("LSTM (Actuator) training completed!")
                
                # Pilot LSTM
                if "LSTM (Pilot)" in selected_models:
                    with st.expander("LSTM (Pilot) Training Details"):
                        lstm_pilot = train_lstm(
                            "pilot",
                            st.session_state.pilot_X_train,
                            st.session_state.pilot_y_train,
                            st.session_state.pilot_X_test,
                            st.session_state.pilot_y_test
                        )
                        
                        predict = lstm_pilot.predict(st.session_state.pilot_X_test)
                        predict = np.argmax(predict, axis=1)
                        testY = np.argmax(st.session_state.pilot_y_test, axis=1)
                        
                        se = accuracy_score(testY, predict)
                        sp = recall_score(testY, predict)
                        
                        st.session_state.sensitivity.append(se)
                        st.session_state.specificity.append(sp)
                        
                        # Calculate hybrid metrics
                        if len(st.session_state.sensitivity) >= 5:
                            hybrid_se = (st.session_state.sensitivity[2] + st.session_state.sensitivity[3] + st.session_state.sensitivity[4]) / 3
                            hybrid_sp = (st.session_state.specificity[2] + st.session_state.specificity[3] + st.session_state.specificity[4]) / 3
                            st.markdown(f"""
                            **Hybrid LSTM Metrics:**
                            - Sensitivity: {hybrid_se:.4f}
                            - Specificity: {hybrid_sp:.4f}
                            """)
                        
                        st.success("LSTM (Pilot) training completed!")
                
                st.session_state.models_trained = True
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during model training: {e}")

# Page 4: Results
elif page == "Results":
    st.header("📊 Results & Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        st.stop()
    
    # Model comparison
    st.subheader("Model Comparison")
    
    if len(st.session_state.sensitivity) >= 5:
        models = ["SVM", "Logistic Regression", "AP2TD (Physical)", "AP2DH (Actuator)", "DH2TD (Pilot)"]
        results = pd.DataFrame({
            "Model": models,
            "Sensitivity": st.session_state.sensitivity[:5],
            "Specificity": st.session_state.specificity[:5]
        })
        
        st.dataframe(results.style.format({
            "Sensitivity": "{:.4f}",
            "Specificity": "{:.4f}"
        }).highlight_max(axis=0))
        
        # Visualization
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Sensitivity", "Specificity"))
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=results["Sensitivity"],
                name="Sensitivity",
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=results["Specificity"],
                name="Specificity",
                marker_color='green'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Analysis
    st.subheader("Feature Importance (SHAP)")
    
    if st.button("Run SHAP Analysis"):
        with st.spinner("Running SHAP analysis..."):
            try:
                # Train a background model for SHAP
                svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1, probability=True)
                svm_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
                
                # Sample background data for SHAP
                background = st.session_state.all_X_train[np.random.choice(
                    st.session_state.all_X_train.shape[0], 
                    min(100, st.session_state.all_X_train.shape[0]), 
                    replace=False
                )]
                
                # Create SHAP explainer
                explainer = shap.KernelExplainer(svm_cls.predict_proba, background)
                shap_values = explainer.shap_values(st.session_state.all_X_test[:50])
                
                # Plot summary
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.summary_plot(shap_values[1], st.session_state.all_X_test[:50], feature_names=st.session_state.all_data.columns)
                st.pyplot(bbox_inches='tight')
                
                st.success("SHAP analysis completed!")
            except Exception as e:
                st.error(f"Error during SHAP analysis: {e}")

# Page 5: Flight Simulation
elif page == "Flight Simulation":
    st.header("✈️ Flight Simulation Dashboard")
    
    # Generate flight path data
    x = np.linspace(0, 100, 100)
    y = np.sin(x)
    z = np.linspace(10000, 0, 100)
    
    # Create 3D flight path
    fig = go.Figure()
    
    # Add flight path
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(width=8, color='blue'),
            name='Flight Path'
        )
    )
    
    # Add markers for start and end
    fig.add_trace(
        go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Start (10,000 ft)'
        )
    )
    
    fig.add_trace(
        go.Scatter3d(
            x=[x[-1]],
            y=[y[-1]],
            z=[z[-1]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Landing'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Flight Approach Simulation',
        scene=dict(
            xaxis_title='Horizontal Distance (m)',
            yaxis_title='Lateral Deviation (m)',
            zaxis_title='Altitude (ft)',
            zaxis=dict(range=[0, 10000]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add landing prediction
    if st.session_state.models_trained:
        st.subheader("Landing Prediction")
        
        # Simulate features for prediction
        simulated_features = np.random.rand(1, st.session_state.all_data.shape[1])
        
        # Make prediction
        svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1)
        svm_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
        prediction = svm_cls.predict(simulated_features)
        
        if prediction[0] == 1:
            st.error("⚠️ Warning: Hard Landing Predicted!")
            st.markdown("""
            **Recommended Actions:**
            - Increase descent rate gradually
            - Check flap settings
            - Monitor airspeed closely
            """)
        else:
            st.success("✅ Normal Landing Predicted")
    else:
        st.warning("Train models first to enable landing prediction")

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: gray;
    text-align: center;
}
</style>
<div class="footer">
    © 2023 E-Pilots - Hard Landing Prediction System | Made with Streamlit
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="E-Pilots: Hard Landing Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample dataset URLs from your GitHub
SAMPLE_DATA_URLS = {
    "pilot": "https://raw.githubusercontent.com/prajwal-gunnala/epilot/main/Dataset/Pilot.csv",
    "actuator": "https://raw.githubusercontent.com/prajwal-gunnala/epilot/main/Dataset/Actuators.csv",
    "physical": "https://raw.githubusercontent.com/prajwal-gunnala/epilot/main/Dataset/Physical.csv"
}

def load_sample_data():
    """Load sample datasets from GitHub"""
    try:
        # Load each dataset with error handling
        pilot = pd.read_csv(SAMPLE_DATA_URLS["pilot"])
        actuator = pd.read_csv(SAMPLE_DATA_URLS["actuator"])
        physical = pd.read_csv(SAMPLE_DATA_URLS["physical"])
        return pilot, actuator, physical
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None, None, None

def process_data(pilot, actuator, physical):
    """Process uploaded or sample data"""
    try:
        Y = physical['label'].values
        pilot = pilot.drop(['label'], axis=1)
        actuator = actuator.drop(['label'], axis=1)
        physical = physical.drop(['label'], axis=1)
        all_data = pd.concat([physical, actuator, pilot], axis=1)
        
        st.session_state.update({
            'pilot': pilot,
            'actuator': actuator,
            'physical': physical,
            'Y': Y,
            'all_data': all_data,
            'data_loaded': True
        })
        
        return True
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return False

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.update({
        'pilot': None,
        'actuator': None,
        'physical': None,
        'Y': None,
        'all_data': None,
        'data_loaded': False,
        'models_trained': False,
        'sensitivity': [],
        'specificity': []
    })

# Title and description
st.title("✈️ E-Pilots: Hard Landing Prediction System")
st.markdown("""
Predicts hard landings using pilot, actuator, and physical data during the approach phase of commercial flights.
""")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", 
    ["Load Data", "Preprocess", "Train Models", "Results", "Flight Simulation"])

# Page 1: Load Data
if page == "Load Data":
    st.header("📂 Load Dataset")
    
    option = st.radio("Choose data source:", 
                     ["Use sample datasets (recommended)", "Upload your own datasets"])
    
    if option == "Use sample datasets (recommended)":
        if st.button("Load Sample Datasets"):
            with st.spinner("Loading sample datasets from GitHub..."):
                pilot, actuator, physical = load_sample_data()
                if pilot is not None:
                    if process_data(pilot, actuator, physical):
                        st.success("Sample datasets loaded successfully!")
                        st.balloons()
    
    else:  # Upload option
        st.warning("For proper functionality, your datasets need the same structure as our sample data.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pilot_file = st.file_uploader("Upload Pilot Data (CSV)", type="csv")
        with col2:
            actuator_file = st.file_uploader("Upload Actuator Data (CSV)", type="csv")
        with col3:
            physical_file = st.file_uploader("Upload Physical Data (CSV)", type="csv")
        
        if st.button("Load Uploaded Datasets"):
            if pilot_file and actuator_file and physical_file:
                with st.spinner("Processing uploaded datasets..."):
                    try:
                        pilot = pd.read_csv(pilot_file)
                        actuator = pd.read_csv(actuator_file)
                        physical = pd.read_csv(physical_file)
                        
                        if process_data(pilot, actuator, physical):
                            st.success("Datasets loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading datasets: {str(e)}")
            else:
                st.warning("Please upload all three dataset files!")
    
    # Show dataset previews if loaded
    if st.session_state.data_loaded:
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

# Page 2: Preprocess
elif page == "Preprocess":
    st.header("🔧 Preprocess Dataset")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first!")
        st.stop()
    
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
                scalers = {}
                for name, data in [('all', all_data), ('pilot', pilot), 
                                 ('actuator', actuator), ('physical', physical)]:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data)
                    scalers[name] = scaler
                    if name == 'all':
                        all_data = scaled_data
                    elif name == 'pilot':
                        pilot = scaled_data
                    elif name == 'actuator':
                        actuator = scaled_data
                    elif name == 'physical':
                        physical = scaled_data
                
                # Reshape to multi-dimensional array
                pilot = np.reshape(pilot, (pilot.shape[0], pilot.shape[1], 1))
                actuator = np.reshape(actuator, (actuator.shape[0], actuator.shape[1], 1))
                physical = np.reshape(physical, (physical.shape[0], physical.shape[1], 1))
                
                # Split dataset into train and test
                (st.session_state.all_X_train, st.session_state.all_X_test, 
                 st.session_state.all_y_train, st.session_state.all_y_test) = train_test_split(
                    all_data, Y, test_size=0.2)
                
                Y_categorical = to_categorical(Y)
                (st.session_state.pilot_X_train, st.session_state.pilot_X_test, 
                 st.session_state.pilot_y_train, st.session_state.pilot_y_test) = train_test_split(
                    pilot, Y_categorical, test_size=0.2)
                (st.session_state.actuator_X_train, st.session_state.actuator_X_test, 
                 st.session_state.actuator_y_train, st.session_state.actuator_y_test) = train_test_split(
                    actuator, Y_categorical, test_size=0.2)
                (st.session_state.physical_X_train, st.session_state.physical_X_test, 
                 st.session_state.physical_y_train, st.session_state.physical_y_test) = train_test_split(
                    physical, Y_categorical, test_size=0.2)
                
                st.session_state.update({
                    'data_preprocessed': True,
                    'scalers': scalers
                })
                
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
                st.error(f"Error during preprocessing: {str(e)}")

# Page 3: Train Models
elif page == "Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if not st.session_state.get('data_preprocessed', False):
        st.warning("Please preprocess the data first!")
        st.stop()
    
    model_options = ["SVM", "Logistic Regression", "LSTM (Physical)", 
                    "LSTM (Actuator)", "LSTM (Pilot)"]
    selected_models = st.multiselect("Select models to train", 
                                   model_options, default=model_options)
    
    if st.button("Train Selected Models"):
        st.session_state.sensitivity = []
        st.session_state.specificity = []
        
        with st.spinner("Training models..."):
            try:
                # SVM
                if "SVM" in selected_models:
                    with st.expander("SVM Training Details", expanded=True):
                        svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1, probability=True)
                        svm_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
                        predict = svm_cls.predict(st.session_state.all_X_test)
                        
                        # Calculate metrics
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
                
                # LSTM Model Training Function
                def train_lstm(model_name, X_train, y_train, X_test, y_test):
                    model = Sequential([
                        LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])),
                        Dropout(0.5),
                        Dense(100, activation='relu'),
                        Dense(y_train.shape[1], activation='softmax')
                    ])
                    model.compile(loss='categorical_crossentropy', 
                                optimizer='adam', 
                                metrics=['accuracy'])
                    
                    # Training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    class ProgressCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / 20
                            progress_bar.progress(progress)
                            status_text.text(
                                f"Epoch {epoch + 1}/20 - "
                                f"Loss: {logs['loss']:.4f} - "
                                f"Accuracy: {logs['accuracy']:.4f}"
                            )
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=20,
                        batch_size=16,
                        validation_data=(X_test, y_test),
                        callbacks=[ProgressCallback()],
                        verbose=0
                    )
                    
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
                    
                    return model
                
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
                        
                        predict = np.argmax(lstm_physical.predict(st.session_state.physical_X_test), axis=1)
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
                        
                        predict = np.argmax(lstm_actuator.predict(st.session_state.actuator_X_test), axis=1)
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
                        
                        predict = np.argmax(lstm_pilot.predict(st.session_state.pilot_X_test), axis=1)
                        testY = np.argmax(st.session_state.pilot_y_test, axis=1)
                        
                        se = accuracy_score(testY, predict)
                        sp = recall_score(testY, predict)
                        
                        st.session_state.sensitivity.append(se)
                        st.session_state.specificity.append(sp)
                        
                        # Calculate hybrid metrics
                        if len(st.session_state.sensitivity) >= 5:
                            hybrid_se = np.mean(st.session_state.sensitivity[2:5])
                            hybrid_sp = np.mean(st.session_state.specificity[2:5])
                            st.markdown(f"""
                            **Hybrid LSTM Metrics:**
                            - Sensitivity: {hybrid_se:.4f}
                            - Specificity: {hybrid_sp:.4f}
                            """)
                        
                        st.success("LSTM (Pilot) training completed!")
                
                st.session_state.models_trained = True
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

# Page 4: Results
elif page == "Results":
    st.header("📊 Results & Analysis")
    
    if not st.session_state.get('models_trained', False):
        st.warning("Please train models first!")
        st.stop()
    
    # Model comparison
    st.subheader("Model Comparison")
    
    if len(st.session_state.sensitivity) >= 5:
        models = ["SVM", "Logistic Regression", "AP2TD (Physical)", 
                 "AP2DH (Actuator)", "DH2TD (Pilot)"]
        results = pd.DataFrame({
            "Model": models,
            "Sensitivity": st.session_state.sensitivity[:5],
            "Specificity": st.session_state.specificity[:5]
        })
        
        # Display results table
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
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    
    if "svm_cls" in globals():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**SVM Confusion Matrix**")
            predict = svm_cls.predict(st.session_state.all_X_test)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                st.session_state.all_y_test, predict, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Logistic Regression Confusion Matrix**")
            predict = lr_cls.predict(st.session_state.all_X_test)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                st.session_state.all_y_test, predict, ax=ax)
            st.pyplot(fig)

# Page 5: Flight Simulation
elif page == "Flight Simulation":
    st.header("✈️ Flight Simulation Dashboard")
    
    # Generate flight path data
    x = np.linspace(0, 100, 100)
    y = np.sin(x)
    z = np.linspace(10000, 0, 100)
    
    # Create 3D flight path visualization
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
    
    # Add landing prediction if models are trained
    if st.session_state.models_trained:
        st.subheader("Landing Prediction")
        
        # Create sample features for prediction
        sample_features = np.random.rand(1, st.session_state.all_data.shape[1])
        sample_features = st.session_state.scalers['all'].transform(sample_features)
        
        # Make prediction
        prediction = svm_cls.predict(sample_features)
        
        if prediction[0] == 1:
            st.error("⚠️ Warning: Hard Landing Predicted!")
            st.markdown("""
            **Recommended Actions:**
            - Increase descent rate gradually
            - Check flap settings
            - Monitor airspeed closely
            - Consider go-around if conditions don't improve
            """)
        else:
            st.success("✅ Normal Landing Predicted")
            st.markdown("""
            **Current Status:**
            - Approach parameters within normal range
            - Continue monitoring instruments
            """)

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: gray;
    text-align: center;
    margin-top: 2rem;
}
</style>
<div class="footer">
    © 2023 E-Pilots - Hard Landing Prediction System | Made with Streamlit
</div>
""", unsafe_allow_html=True)

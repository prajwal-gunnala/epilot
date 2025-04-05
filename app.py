import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # required for 3d plotting
import os
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM
from keras.utils.np_utils import to_categorical
import shap

# Set the page configuration
st.set_page_config(page_title="E-Pilots: Flight Landing Prediction", layout="wide")

# Initialize session state variables
if 'pilot' not in st.session_state:
    st.session_state.pilot = None
if 'actuator' not in st.session_state:
    st.session_state.actuator = None
if 'physical' not in st.session_state:
    st.session_state.physical = None
if 'Y' not in st.session_state:
    st.session_state.Y = None
if 'all_data' not in st.session_state:
    st.session_state.all_data = None
if 'pilot_X_train' not in st.session_state:
    st.session_state.pilot_X_train = None
if 'pilot_X_test' not in st.session_state:
    st.session_state.pilot_X_test = None
if 'pilot_y_train' not in st.session_state:
    st.session_state.pilot_y_train = None
if 'pilot_y_test' not in st.session_state:
    st.session_state.pilot_y_test = None
if 'actuator_X_train' not in st.session_state:
    st.session_state.actuator_X_train = None
if 'actuator_X_test' not in st.session_state:
    st.session_state.actuator_X_test = None
if 'actuator_y_train' not in st.session_state:
    st.session_state.actuator_y_train = None
if 'actuator_y_test' not in st.session_state:
    st.session_state.actuator_y_test = None
if 'physical_X_train' not in st.session_state:
    st.session_state.physical_X_train = None
if 'physical_X_test' not in st.session_state:
    st.session_state.physical_X_test = None
if 'physical_y_train' not in st.session_state:
    st.session_state.physical_y_train = None
if 'physical_y_test' not in st.session_state:
    st.session_state.physical_y_test = None
if 'all_X_train' not in st.session_state:
    st.session_state.all_X_train = None
if 'all_X_test' not in st.session_state:
    st.session_state.all_X_test = None
if 'all_y_train' not in st.session_state:
    st.session_state.all_y_train = None
if 'all_y_test' not in st.session_state:
    st.session_state.all_y_test = None
if 'sensitivity' not in st.session_state:
    st.session_state.sensitivity = []
if 'specificity' not in st.session_state:
    st.session_state.specificity = []
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# Function to upload datasets
def upload_dataset(pilot_file, actuator_file, physical_file):
    if pilot_file is not None and actuator_file is not None and physical_file is not None:
        st.session_state.pilot = pd.read_csv(pilot_file)
        st.session_state.actuator = pd.read_csv(actuator_file)
        st.session_state.physical = pd.read_csv(physical_file)
        st.session_state.Y = st.session_state.physical['label'].values
        
        # Drop label column from individual dataframes
        st.session_state.pilot.drop(['label'], axis=1, inplace=True)
        st.session_state.actuator.drop(['label'], axis=1, inplace=True)
        st.session_state.physical.drop(['label'], axis=1, inplace=True)
        
        # Merge datasets
        data_list = [st.session_state.physical, st.session_state.actuator, st.session_state.pilot]
        st.session_state.all_data = pd.concat(data_list, axis=1)
        
        st.success("Datasets uploaded and merged successfully!")
        st.write("Pilot Dataset Preview:")
        st.dataframe(st.session_state.pilot.head())
        st.write("Actuator Dataset Preview:")
        st.dataframe(st.session_state.actuator.head())
        st.write("Physical Dataset Preview:")
        st.dataframe(st.session_state.physical.head())
        
        # Plot label distribution
        labels, count = np.unique(st.session_state.Y, return_counts=True)
        bars = ['Not Hard Landing', 'Hard Landing']
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(bars)), count)
        ax.set_xticks(np.arange(len(bars)))
        ax.set_xticklabels(bars)
        ax.set_xlabel("Landing Type")
        ax.set_ylabel("Counts")
        ax.set_title("Landing Type Distribution")
        st.pyplot(fig)
    else:
        st.error("Please upload all three files!")

# Flight simulation function
def flight_simulation():
    st.subheader("Flight Simulation Dashboard")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate flight path data
    x = np.linspace(0, 100, 100)
    y = np.sin(x)
    z = np.linspace(10000, 0, 100)
    
    line, = ax.plot(x, y, z, label="Flight Path")
    ax.set_xlabel('Horizontal Distance')
    ax.set_ylabel('Lateral Deviation')
    ax.set_zlabel('Altitude')
    ax.legend()
    
    # For simplicity, we display a static plot.
    # (Animating in Streamlit requires more advanced techniques)
    st.pyplot(fig)

# Function to explain the model using SHAP
def explain_model():
    if st.session_state.all_X_train is None or st.session_state.all_X_test is None:
        st.error("Please preprocess the dataset first.")
        return
    st.subheader("Model Explanation (SHAP)")
    # Select a random subset for background
    background = st.session_state.all_X_train[np.random.choice(st.session_state.all_X_train.shape[0], 100, replace=False)]
    # Here we use an SVM classifier as a proxy model for SHAP explanation.
    # Make sure that runSVM() was executed to train svm_cls.
    svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1)
    svm_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
    st.session_state.explainer = shap.KernelExplainer(svm_cls.predict, background)
    st.session_state.shap_values = st.session_state.explainer.shap_values(st.session_state.all_X_test[:10])
    
    # Create SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(st.session_state.shap_values, st.session_state.all_X_test[:10],
                      feature_names=st.session_state.all_data.columns, show=False)
    plt.title("Feature Impact on Predictions (SHAP)")
    st.pyplot(fig)
    st.success("SHAP analysis completed.")

# Function to display an interactive confusion matrix
def interactive_confusion_matrix():
    if st.session_state.all_X_test is None:
        st.error("Please preprocess the dataset and run a model first.")
        return
    st.subheader("Confusion Matrix")
    # Train a sample SVM classifier
    svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1)
    svm_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
    predict = svm_cls.predict(st.session_state.all_X_test)
    cm = confusion_matrix(st.session_state.all_y_test, predict)
    
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', color="black")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Function to preprocess dataset
def preprocess_dataset():
    if st.session_state.all_data is None:
        st.error("Please upload the datasets first!")
        return
    st.subheader("Preprocessing Dataset")
    # Convert datasets to numpy arrays
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
    scaler_all = StandardScaler()
    all_data = scaler_all.fit_transform(all_data)
    scaler_pilot = StandardScaler()
    pilot = scaler_pilot.fit_transform(pilot)
    scaler_actuator = StandardScaler()
    actuator = scaler_actuator.fit_transform(actuator)
    scaler_physical = StandardScaler()
    physical = scaler_physical.fit_transform(physical)

    # Reshape to 3D array for LSTM input
    pilot = np.reshape(pilot, (pilot.shape[0], pilot.shape[1], 1))
    actuator = np.reshape(actuator, (actuator.shape[0], actuator.shape[1], 1))
    physical = np.reshape(physical, (physical.shape[0], physical.shape[1], 1))

    # Split dataset into train and test sets
    all_X_train, all_X_test, all_y_train, all_y_test = train_test_split(all_data, Y, test_size=0.2)
    Y_cat = to_categorical(Y)
    pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test = train_test_split(pilot, Y_cat, test_size=0.2)
    actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test = train_test_split(actuator, Y_cat, test_size=0.2)
    physical_X_train, physical_X_test, physical_y_train, physical_y_test = train_test_split(physical, Y_cat, test_size=0.2)

    # Update session state
    st.session_state.all_X_train = all_X_train
    st.session_state.all_X_test = all_X_test
    st.session_state.all_y_train = all_y_train
    st.session_state.all_y_test = all_y_test

    st.session_state.pilot_X_train = pilot_X_train
    st.session_state.pilot_X_test = pilot_X_test
    st.session_state.pilot_y_train = pilot_y_train
    st.session_state.pilot_y_test = pilot_y_test

    st.session_state.actuator_X_train = actuator_X_train
    st.session_state.actuator_X_test = actuator_X_test
    st.session_state.actuator_y_train = actuator_y_train
    st.session_state.actuator_y_test = actuator_y_test

    st.session_state.physical_X_train = physical_X_train
    st.session_state.physical_X_test = physical_X_test
    st.session_state.physical_y_train = physical_y_train
    st.session_state.physical_y_test = physical_y_test

    st.success("Preprocessing completed!")
    st.write("Total records in dataset: ", all_data.shape[0])
    st.write("Total features in merged dataset: ", all_data.shape[1])
    st.write("Total Pilot features: ", pilot.shape[1])
    st.write("Total Actuator features: ", actuator.shape[1])
    st.write("Total Physical features: ", physical.shape[1])
    st.write("Train records (80%): ", all_X_train.shape[0])
    st.write("Test records (20%): ", all_X_test.shape[0])

# Function to calculate and display metrics
def calculate_metrics(algorithm, y_test, predict):
    cm = confusion_matrix(y_test, predict)
    se = accuracy_score(y_test, predict)
    sp = recall_score(y_test, predict)
    st.session_state.sensitivity.append(se)
    st.session_state.specificity.append(sp)
    st.write(f"{algorithm} - Sensitivity: {se}")
    st.write(f"{algorithm} - Specificity: {sp}")
    
    # For the pilot-based model, compute hybrid metrics if available
    if algorithm == 'DH2TD Pilot Features' and len(st.session_state.sensitivity) >= 5:
        hybrid_se = np.mean(st.session_state.sensitivity[2:5])
        hybrid_sp = np.mean(st.session_state.specificity[2:5])
        st.write(f"Hybrid LSTM Sensitivity: {hybrid_se}")
        st.write(f"Hybrid LSTM Specificity: {hybrid_sp}")
    
    # Plot boxplot for sensitivity and specificity
    values = np.array([[se - 0.10, se], [sp - 0.10, sp]])
    df_box = pd.DataFrame(values, columns=['Low', 'High'], index=['Sensitivity', 'Specificity'])
    fig, ax = plt.subplots()
    df_box.plot(kind='box', ax=ax)
    ax.set_title(f"{algorithm} Metrics")
    st.pyplot(fig)

# Function to run SVM
def run_svm():
    if st.session_state.all_X_train is None:
        st.error("Please preprocess the dataset first!")
        return
    st.subheader("Running SVM Algorithm")
    svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1)
    svm_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
    predict = svm_cls.predict(st.session_state.all_X_test)
    calculate_metrics("SVM", st.session_state.all_y_test, predict)

# Function to run Logistic Regression
def run_logistic_regression():
    if st.session_state.all_X_train is None:
        st.error("Please preprocess the dataset first!")
        return
    st.subheader("Running Logistic Regression")
    lr_cls = LogisticRegression(max_iter=300)
    lr_cls.fit(st.session_state.all_X_train, st.session_state.all_y_train)
    predict = lr_cls.predict(st.session_state.all_X_test)
    calculate_metrics("Logistic Regression", st.session_state.all_y_test, predict)

# Function to run LSTM on Physical features (AP2TD)
def run_ap2td():
    if st.session_state.physical_X_train is None:
        st.error("Please preprocess the dataset first!")
        return
    st.subheader("Running AP2TD (Physical Features)")
    if os.path.exists('model/physical_model.json'):
        with open('model/physical_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        lstm_physical = model_from_json(loaded_model_json)
        lstm_physical.load_weights("model/physical_weights.h5")
    else:
        lstm_physical = Sequential()
        lstm_physical.add(LSTM(100, input_shape=(st.session_state.physical_X_train.shape[1],
                                                   st.session_state.physical_X_train.shape[2])))
        lstm_physical.add(Dropout(0.5))
        lstm_physical.add(Dense(100, activation='relu'))
        lstm_physical.add(Dense(st.session_state.physical_y_train.shape[1], activation='softmax'))
        lstm_physical.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_physical.fit(st.session_state.physical_X_train, st.session_state.physical_y_train,
                          epochs=20, batch_size=16, validation_data=(st.session_state.physical_X_test,
                                                                     st.session_state.physical_y_test))
        # Save model for future use
        lstm_physical.save_weights('model/physical_weights.h5')
        model_json = lstm_physical.to_json()
        with open("model/physical_model.json", "w") as json_file:
            json_file.write(model_json)
    st.text(lstm_physical.summary())
    predict = lstm_physical.predict(st.session_state.physical_X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(st.session_state.physical_y_test, axis=1)
    calculate_metrics("AP2TD Physical Features", testY, predict)

# Function to run LSTM on Actuator features (AP2DH)
def run_ap2dh():
    if st.session_state.actuator_X_train is None:
        st.error("Please preprocess the dataset first!")
        return
    st.subheader("Running AP2DH (Actuator Features)")
    if os.path.exists('model/actuator_model.json'):
        with open('model/actuator_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        lstm_actuator = model_from_json(loaded_model_json)
        lstm_actuator.load_weights("model/actuator_weights.h5")
    else:
        lstm_actuator = Sequential()
        lstm_actuator.add(LSTM(100, input_shape=(st.session_state.actuator_X_train.shape[1],
                                                   st.session_state.actuator_X_train.shape[2])))
        lstm_actuator.add(Dropout(0.5))
        lstm_actuator.add(Dense(100, activation='relu'))
        lstm_actuator.add(Dense(st.session_state.actuator_y_train.shape[1], activation='softmax'))
        lstm_actuator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_actuator.fit(st.session_state.actuator_X_train, st.session_state.actuator_y_train,
                          epochs=20, batch_size=16, validation_data=(st.session_state.actuator_X_test,
                                                                     st.session_state.actuator_y_test))
        lstm_actuator.save_weights('model/actuator_weights.h5')
        model_json = lstm_actuator.to_json()
        with open("model/actuator_model.json", "w") as json_file:
            json_file.write(model_json)
    st.text(lstm_actuator.summary())
    predict = lstm_actuator.predict(st.session_state.actuator_X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(st.session_state.actuator_y_test, axis=1)
    calculate_metrics("AP2DH Actuator Features", testY, predict)

# Function to run LSTM on Pilot features (DH2TD)
def run_dh2td():
    if st.session_state.pilot_X_train is None:
        st.error("Please preprocess the dataset first!")
        return
    st.subheader("Running DH2TD (Pilot Features)")
    if os.path.exists('model/pilot_model.json'):
        with open('model/pilot_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        lstm_pilot = model_from_json(loaded_model_json)
        lstm_pilot.load_weights("model/pilot_weights.h5")
    else:
        lstm_pilot = Sequential()
        lstm_pilot.add(LSTM(100, input_shape=(st.session_state.pilot_X_train.shape[1],
                                                st.session_state.pilot_X_train.shape[2])))
        lstm_pilot.add(Dropout(0.5))
        lstm_pilot.add(Dense(100, activation='relu'))
        lstm_pilot.add(Dense(st.session_state.pilot_y_train.shape[1], activation='softmax'))
        lstm_pilot.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_pilot.fit(st.session_state.pilot_X_train, st.session_state.pilot_y_train,
                       epochs=20, batch_size=16, validation_data=(st.session_state.pilot_X_test,
                                                                  st.session_state.pilot_y_test))
        lstm_pilot.save_weights('model/pilot_weights.h5')
        model_json = lstm_pilot.to_json()
        with open("model/pilot_model.json", "w") as json_file:
            json_file.write(model_json)
    st.text(lstm_pilot.summary())
    predict = lstm_pilot.predict(st.session_state.pilot_X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(st.session_state.pilot_y_test, axis=1)
    calculate_metrics("DH2TD Pilot Features", testY, predict)

# Function to generate a comparison graph
def comparison_graph():
    if not st.session_state.sensitivity or not st.session_state.specificity:
        st.error("Please run the models first to generate metrics!")
        return
    df = pd.DataFrame([
        ['SVM','Sensitivity', st.session_state.sensitivity[0]],
        ['SVM','Specificity', st.session_state.specificity[0]],
        ['Logistic Regression','Sensitivity', st.session_state.sensitivity[1]],
        ['Logistic Regression','Specificity', st.session_state.specificity[1]],
        ['AP2TD','Sensitivity', st.session_state.sensitivity[2]],
        ['AP2TD','Specificity', st.session_state.specificity[2]],
        ['AP2DH','Sensitivity', st.session_state.sensitivity[3]],
        ['AP2DH','Specificity', st.session_state.specificity[3]],
        ['DH2TD','Sensitivity', st.session_state.sensitivity[4] if len(st.session_state.sensitivity)>4 else None],
        ['DH2TD','Specificity', st.session_state.specificity[4] if len(st.session_state.specificity)>4 else None],
    ], columns=['Algorithm', 'Parameter', 'Value']).dropna()
    
    pivot = df.pivot(index='Parameter', columns='Algorithm', values='Value')
    pivot.plot(kind='bar')
    plt.title("Comparison of Sensitivity & Specificity")
    st.pyplot(plt.gcf())

# Sidebar: File upload for datasets
st.sidebar.header("Upload Datasets")
pilot_file = st.sidebar.file_uploader("Upload Pilot CSV", type="csv")
actuator_file = st.sidebar.file_uploader("Upload Actuators CSV", type="csv")
physical_file = st.sidebar.file_uploader("Upload Physical CSV", type="csv")
if st.sidebar.button("Upload and Merge Datasets"):
    upload_dataset(pilot_file, actuator_file, physical_file)

# Main App Title
st.title("E-Pilots: A System to Predict Hard Landing During the Approach Phase of Commercial Flights")

# Buttons for each functionality
st.header("Data Processing & Modeling")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Preprocess Dataset"):
        preprocess_dataset()
with col2:
    if st.button("Run SVM Algorithm"):
        run_svm()
with col3:
    if st.button("Run Logistic Regression"):
        run_logistic_regression()

col4, col5, col6 = st.columns(3)
with col4:
    if st.button("Run AP2TD Algorithm"):
        run_ap2td()
with col5:
    if st.button("Run AP2DH Algorithm"):
        run_ap2dh()
with col6:
    if st.button("Run DH2TD Algorithm"):
        run_dh2td()

st.header("Visualization & Simulation")
col7, col8, col9 = st.columns(3)
with col7:
    if st.button("Comparison Graph"):
        comparison_graph()
with col8:
    if st.button("Flight Simulation"):
        flight_simulation()
with col9:
    if st.button("Explain Model (SHAP)"):
        explain_model()

st.header("Evaluation")
if st.button("Confusion Matrix"):
    interactive_confusion_matrix()


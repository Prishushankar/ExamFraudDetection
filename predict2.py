import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load and preprocess the data
def preprocess_data(data):
    data.fillna(method='ffill', inplace=True)

    categorical_features = ['Nationality', 'Domicile State', 'Reservation_Category', 'Gender', 'Marital_Status', 'Home_Guard_Status', 'Are you a Ward of a Freedom Fighter?']
    numerical_features = ['10th_Marks', '12th_Marks']

    present_categorical_features = [feature for feature in categorical_features if feature in data.columns]
    if present_categorical_features:
        encoder = OneHotEncoder()
        encoded_features = encoder.fit_transform(data[present_categorical_features]).toarray()
    else:
        encoded_features = np.array([]).reshape(len(data), 0)

    present_numerical_features = [feature for feature in numerical_features if feature in data.columns]
    if present_numerical_features:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[present_numerical_features])
    else:
        scaled_features = np.array([]).reshape(len(data), 0)

    X = np.hstack((encoded_features, scaled_features))
    y = data['Exam_Marks']

    return X, y

def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def process_file(filepath):
    def run_processing():
        start_loading_animation("Processing Data...")

        # Load data
        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {e}")
            stop_loading_animation()
            return

        if 'Exam_Marks' not in data.columns or 'Name' not in data.columns:
            messagebox.showerror("Error", "Dataset must contain 'Exam_Marks' and 'Name' columns.")
            stop_loading_animation()
            return

        try:
            X, y = preprocess_data(data)
        except Exception as e:
            messagebox.showerror("Error", f"Error preprocessing data: {e}")
            stop_loading_animation()
            return

        # Load the pre-trained model
        model_path = r"C:\Users\Priyanshu Shankar\Desktop\csbc_fraud_detection\csbc_fraud\fraud_detection_model.h5"
        model = load_trained_model(model_path)
        if model is None:
            messagebox.showerror("Error", "Failed to load the model.")
            stop_loading_animation()
            return

        # Predict using the model
        try:
            y_pred = model.predict(X)
        except Exception as e:
            messagebox.showerror("Error", f"Error making predictions: {e}")
            stop_loading_animation()
            return

        # Calculate errors
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        threshold = 6
        fraudulent_applications = np.abs(y - y_pred.flatten()) > threshold
        fraud_indices = np.where(fraudulent_applications)[0]
        fraud_names = data.iloc[fraud_indices]['Name'].tolist()

        # Save fraudulent students to a CSV file
        if fraud_names:
            fraud_df = pd.DataFrame({'Name': fraud_names})
            fraud_df.to_csv("fraudulent_students.csv", index=False)
            messagebox.showinfo("Success", "Fraudulent students saved to 'fraudulent_students.csv'.")

        stop_loading_animation()
        show_fraudulent_students(fraud_names)
        messagebox.showinfo("Model Evaluation Complete", f"Mean Absolute Error: {mae:.2f}\nMean Squared Error: {mse:.2f}")

    threading.Thread(target=run_processing).start()

def upload_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        process_file(filepath)

# Loading animation
def start_loading_animation(text):
    loading_label.config(text=text)
    loading_label.pack(pady=20)
    animate_loading()

def stop_loading_animation():
    loading_label.pack_forget()
    root.after_cancel(animation_id)

def animate_loading():
    global animation_index, animation_id
    dots = ['.', '..', '...', '....']
    loading_label.config(text=f"{loading_label['text'][:-len(dots[animation_index])]}{dots[animation_index]}")
    animation_index = (animation_index + 1) % len(dots)
    animation_id = root.after(500, animate_loading)

# Create the window for fraudulent students
def show_fraudulent_students(fraud_names):
    fraud_window = tk.Toplevel(root)
    fraud_window.title("Fraudulent Students")
    fraud_window.geometry("500x300")
    
    # Keep the window open until it's manually closed
    fraud_window.grab_set()

    # Add a scrollable treeview to display fraudulent students in a table format
    tree = ttk.Treeview(fraud_window, columns=("Name"), show="headings", height=10)
    tree.heading("Name", text="Fraudulent Students")

    for name in fraud_names:
        tree.insert("", tk.END, values=(name,))

    tree.pack(fill="both", expand=True, padx=20, pady=20)

# Create the Tkinter window
root = tk.Tk()
root.title("Fraud Detection")
root.geometry("600x400")
root.configure(bg="#f4f4f4")

# Title
title_label = tk.Label(root, text="Fraud Detection System", font=("Arial", 22, "bold"), bg="#f4f4f4", fg="#333333")
title_label.pack(pady=20)

# Input box design
input_frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
input_frame.pack(pady=20, padx=20, fill="both", expand=True)

upload_button = tk.Button(input_frame, text="Upload Dataset", command=upload_file, bg="#4CAF50", fg="#ffffff", font=("Arial", 16), padx=20, pady=10)
upload_button.pack(pady=30)

# Loading label for animation
loading_label = tk.Label(root, text="", font=("Arial", 14), bg="#f4f4f4", fg="#555555")
animation_index = 0
animation_id = None

# Run the Tkinter event loop
root.mainloop()

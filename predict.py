import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def process_file(filepath):
    def run_processing():
        start_loading_animation("Training Model...")
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

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            messagebox.showerror("Error", f"Error splitting data: {e}")
            stop_loading_animation()
            return

        try:
            model = build_model(X_train.shape[1])
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {e}")
            stop_loading_animation()
            return

        try:
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
        except Exception as e:
            messagebox.showerror("Error", f"Error evaluating model: {e}")
            stop_loading_animation()
            return

        threshold = 5
        fraudulent_applications = np.abs(y_test - y_pred.flatten()) > threshold
        fraud_indices = np.where(fraudulent_applications)[0]
        fraud_names = data.iloc[fraud_indices]['Name'].tolist()

        # Save fraudulent students to a CSV file
        if fraud_names:
            fraud_df = pd.DataFrame({'Name': fraud_names})
            fraud_df.to_csv("fraudulent_students.csv", index=False)
            messagebox.showinfo("Success", "Fraudulent students saved to 'fraudulent_students.csv'.")

        stop_loading_animation()
        show_fraudulent_students(fraud_names)
        messagebox.showinfo("Model Training Complete", f"Mean Absolute Error: {mae:.2f}\nMean Squared Error: {mse:.2f}")

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
    fraud_window.title("Suspected  Students")
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

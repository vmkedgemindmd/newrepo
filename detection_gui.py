import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load your trained Xception model
model = tf.keras.models.load_model("xception_model.h5")

# Define your class labels (must match your training)
class_labels = [
    'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
    'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Yellow Leaf Curl Virus',
    'Mosaic Virus', 'Healthy'
]

# Preprocess the image to match model input
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence, predictions[0]

# Function to choose file and predict
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        # Display image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk)
        panel.image = img_tk

        # Get prediction
        result, confidence, probs = predict_image(file_path)

        # Display result
        result_label.config(
            text=f"Prediction: {result}\nConfidence: {confidence:.2f}%",
            fg="white", bg="#2E8B57", font=("Arial", 14, "bold")
        )

        # Show graph
        show_graph(probs)

# Show probability graph
def show_graph(probabilities):
    plt.figure(figsize=(8, 5))
    plt.barh(class_labels, probabilities, color="tomato")
    plt.xlabel("Probability")
    plt.ylabel("Disease Class")
    plt.title("Prediction Probability Distribution")
    plt.tight_layout()
    plt.show()

# ---------------- GUI -----------------
root = tk.Tk()
root.title("Tomato Leaf Disease Detection - Xception AI")
root.geometry("700x600")
root.configure(bg="#f4f4f4")

# Title label
title_label = tk.Label(root, text="🍅 Tomato Leaf Disease Detector", font=("Arial", 20, "bold"), bg="#f4f4f4", fg="#D35400")
title_label.pack(pady=10)

# Image panel
panel = tk.Label(root, bg="#f4f4f4")
panel.pack(pady=10)

# Result label
result_label = tk.Label(root, text="Please upload an image", font=("Arial", 14), bg="#f4f4f4", fg="black")
result_label.pack(pady=10)

# Upload button
style = ttk.Style()
style.configure("TButton", font=("Arial", 12, "bold"), padding=10)

upload_btn = ttk.Button(root, text="📁 Select Leaf Image", command=open_file)
upload_btn.pack(pady=20)

root.mainloop()

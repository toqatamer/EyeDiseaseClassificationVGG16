import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry, Button, Text, Scrollbar
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import pydicom
from skimage.feature import hog

class EyeDiseaseDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def dataPaths(self):
        filepaths = []
        labels = []
        folds = os.listdir(self.data_dir)
        for fold in folds:
            foldPath = os.path.join(self.data_dir, fold)
            filelist = os.listdir(foldPath)
            for file in filelist:
                fpath = os.path.join(foldPath, file)
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')):
                    filepaths.append(fpath)
                    labels.append(fold)
        return filepaths, labels

def read_image(filepath):
    if filepath.lower().endswith('.dcm'):
        dicom_image = pydicom.dcmread(filepath)
        image = dicom_image.pixel_array
        if len(image.shape) == 2:  # grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.imread(filepath)
    image = cv2.resize(image, (64, 64))
    return image

def load_images_and_labels(data_dir, batch_size):
    filepaths, labels = EyeDiseaseDataset(data_dir).dataPaths()
    images = []
    for i in range(0, len(filepaths), batch_size):
        batch_filepaths = filepaths[i:i+batch_size]
        batch_images = [read_image(filepath) for filepath in batch_filepaths]
        images.extend(batch_images)
    return np.array(images), np.array(labels)

def train_model(data_dir, batch_size, epochs):
    # Load images and labels
    images, labels = load_images_and_labels(data_dir, batch_size)
    labels = pd.get_dummies(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Define data augmentation parameters
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Augment training data
    train_datagen.fit(X_train)

    # Convert numpy arrays to TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.repeat()  # Repeat the dataset indefinitely
    train_dataset = train_dataset.batch(batch_size)

    # Load VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    
    # Freeze convolutional base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Remove the MaxPooling2D layer
    model = Sequential([
        base_model,
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
 
    # Compile the model with a lower learning rate
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with augmented data
    history = model.fit(train_dataset,
                        steps_per_epoch=math.ceil(len(X_train) / batch_size),
                        epochs=epochs,
                        validation_data=(X_test, y_test))

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Save the trained model in native Keras format
    model.save("trained_model_vgg16.h5")

    # Print the final accuracy
    final_accuracy = history.history['accuracy'][-1]
    print(f'Final training accuracy: {final_accuracy:.4f}')
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f'Final validation accuracy: {final_val_accuracy:.4f}')

def classify_image(image_path):
    img = read_image(image_path)
    img_statistical_features = extract_statistical_features(img_preprocessed)
    img_statistical_features = np.expand_dims(img_statistical_features, axis=0)

    # Load the trained model
    model = load_model("trained_model_vgg16.h5")

    # Predict class
    prediction = model.predict((np.expand_dims(img_preprocessed, axis=0), img_statistical_features))
    class_index = np.argmax(prediction)
    classes = ['Cataracts', 'Diabetic retinopathy', 'Glaucoma', 'Normal']
    predicted_class = classes[class_index]

    messagebox.showinfo("Image Classification", f"The image belongs to the class: {predicted_class}")

def convert_to_dcm(image_path):
    # Function to create a DICOM dataset from user input

    def save_dicom(attributes, output_dir):
        # Convert the image to DICOM format
        img = read_image(image_path)
        img = np.uint16(img)  # Convert to 16-bit unsigned integer (DICOM format)

        # Create a new DICOM dataset
        ds = pydicom.dataset.FileDataset(image_path, {}, file_meta=None, preamble=b"\0" * 128)

        # Set DICOM attributes
        for tag, value in attributes.items():
            setattr(ds, tag, value)

        # Set pixel data
        ds.PixelData = img.tobytes()

        # Save the DICOM dataset
        dicom_filename = os.path.splitext(os.path.basename(image_path))[0] + ".dcm"
        dicom_path = os.path.join(output_dir, dicom_filename)
        ds.save_as(dicom_path)

        # Inform the user
        messagebox.showinfo("Conversion Successful", "Image converted to DICOM format successfully.")

    # Create a dialog to enter DICOM attributes and select output directory
    attributes_dialog = tk.Toplevel()
    attributes_dialog.title("Enter DIC OM Attributes and Select Output Directory")

    # Define DICOM attributes
    dicom_attributes = {
        "PatientName": "",
        "PatientID": "",
        "StudyDescription": "",
        "Modality": "",
        "ImageType": "",
        "PhotometricInterpretation": ""
    }

    # Input fields for DICOM attributes
    for tag, default_value in dicom_attributes.items():
        label = tk.Label(attributes_dialog, text=tag)
        label.pack()

        if isinstance(default_value, list):
            entry = tk.Entry(attributes_dialog, width=30)
            entry.insert(tk.END, ",".join(map(str, default_value)))
        else:
            entry = tk.Entry(attributes_dialog, width=30)
            entry.insert(tk.END, default_value)

        entry.pack()
        dicom_attributes[tag] = entry

    # Button to choose output directory and save DICOM attributes
    def choose_output_directory():
        output_dir = filedialog.askdirectory()
        if output_dir:
            save_dicom({tag: entry.get() for tag, entry in dicom_attributes.items()}, output_dir)

    choose_dir_button = tk.Button(attributes_dialog, text="Choose Output Directory", command=choose_output_directory)
    choose_dir_button.pack()

def view_dicom_info(image_path):
    dicom_image = pydicom.dcmread(image_path)
    info_window = tk.Toplevel()
    info_window.title("DICOM Image Info")
    
    scrollbar = Scrollbar(info_window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    text = Text(info_window, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    info = dicom_image.dir()
    for attribute in info:
        text.insert(tk.END, f"{attribute}: {getattr(dicom_image, attribute, 'N/A')}\n")
    
    scrollbar.config(command=text.yview)

def select_image_and_action(action):
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.dcm"), ("All files", "*.*")])
    if image_path:
        if action == "classify":
            classify_image(image_path)
        elif action == "convert":
            convert_to_dcm(image_path)
        elif action == "view_info":
            view_dicom_info(image_path)
    else:
        messagebox.showwarning("No Image Selected", "Please select an image.")

def select_folder_and_train_model():
    global dataset_dir
    dataset_dir = filedialog.askdirectory()
    if dataset_dir:
        train_model(dataset_dir, batch_size=32, epochs=20)  # Increased epochs for better training
    else:
        messagebox.showwarning("No Dataset Folder", "Please select a folder containing the dataset.")

def extract_hog_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    # Return the HOG features and HOG image
    return features, hog_image

def select_image_and_extract_features():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        hog_features, hog_image = extract_hog_features(image_path)
        
        # Display the original image and the HOG image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')

        ax2.axis('off')
        ax2.imshow(hog_image, cmap='gray')
        ax2.set_title('HOG Image')

        plt.show()
    else:
        messagebox.showwarning("No Image Selected", "Please select an image.")

def extract_statistical_features(image):
    # Compute statistical features from the image
    # Example: Mean, Variance, Skewness, Kurtosis
    mean_intensity = np.mean(image)
    variance_intensity = np.var(image)
    skewness_intensity = np.mean((image - np.mean(image))**3) / np.mean((image - np.mean(image))**2)**(3/2)
    kurtosis_intensity = np.mean((image - np.mean(image))**4) / np.mean((image - np.mean(image))**2)**2
    
    return mean_intensity, variance_intensity, skewness_intensity, kurtosis_intensity

def classify_image(image_path):
    img = read_image(image_path)
    img_statistical_features = extract_statistical_features(img)
    img_statistical_features = np.expand_dims(img_statistical_features, axis=0)

    # Load the trained model
    model = load_model("trained_model_vgg16.h5")

    # Predict class
    prediction = model.predict((np.expand_dims(img, axis=0), img_statistical_features))
    class_index = np.argmax(prediction)
    classes = ['Cataracts', 'Diabetic retinopathy', 'Glaucoma', 'Normal']
    predicted_class = classes[class_index]

    messagebox.showinfo("Image Classification", f"The image belongs to the class: {predicted_class}")

def extract_and_display_features(image_path):
    img = read_image(image_path)
    img_statistical_features = extract_statistical_features(img)

    # Create a new window to display the statistical features
    features_window = tk.Toplevel()
    features_window.title("Statistical Features")
    
    # Display the statistical features
    features_label = tk.Label(features_window, text=f"Mean Intensity: {img_statistical_features[0]}\n"
                                                     f"Variance Intensity: {img_statistical_features[1]}\n"
                                                     f"Skewness Intensity: {img_statistical_features[2]}\n"
                                                     f"Kurtosis Intensity: {img_statistical_features[3]}\n",
                              font=("Calibri", 12), padx=20, pady=20)
    features_label.pack()

def select_image_and_extract_features():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        extract_and_display_features(image_path)
    else:
        messagebox.showwarning("No Image Selected", "Please select an image.")

import tkinter as tk
# Create the main window
root = tk.Tk()
root.title("Eye Disease Classification")

# Set background color
root.configure(bg="#f0f0f0")

# Header frame
header_frame = tk.Frame(root, bg="#34495e", padx=20, pady=10)
header_frame.pack(fill="x")

title_label = tk.Label(header_frame, text="Eye Disease Classification", font=("Calibri", 24, "bold"), fg="#ffffff", bg="#34495e")
title_label.grid(row=0, column=0, padx=10)

# Overview label
overview_label = tk.Label(root, text="Our application is here to assist you.",
                          font=("Calibri", 10), bg="#ffffff", fg="#34495e", padx=20)
overview_label.pack(side="bottom", fill="x")

# Instructions label
instructions_label = tk.Label(root, text="Please select an option below:", font=("Calibri", 14), bg="#ffffff")
instructions_label.pack()

# Button frame
button_frame = tk.Frame(root, bg="#ffffff")
button_frame.pack(pady=10)

# Buttons with shades of blue
train_button = tk.Button(button_frame, text="Select Dataset Folder and Train Model", command=select_folder_and_train_model, bg="#1b4f72", fg="white", relief="raised", width=40, font=("Calibri", 12, "bold"))
train_button.grid(row=0, column=0, padx=10, pady=5)

classify_button = tk.Button(button_frame, text="Select Image and Classify", command=lambda: select_image_and_action("classify"), bg="#21618c", fg="white", relief="raised", width=40, font=("Calibri", 12, "bold"))
classify_button.grid(row=0, column=1, padx=10, pady=5)

convert_button = tk.Button(button_frame, text="Select Image and Convert to DICOM", command=lambda: select_image_and_action("convert"), bg="#2874a6", fg="white", relief="raised", width=40, font=("Calibri", 12, "bold"))
convert_button.grid(row=0, column=2, padx=10, pady=5)

view_info_button = tk.Button(button_frame, text="Select DICOM Image and View Info", command=lambda: select_image_and_action("view_info"), bg="#2e86c1", fg="white", relief="raised", width=40, font=("Calibri", 12, "bold"))
view_info_button.grid(row=1, column=0, padx=10, pady=5)

extract_features_button = tk.Button(button_frame, text="Select Image and Extract Features", command=select_image_and_extract_features, bg="#3498db", fg="white", relief="raised", width=40, font=("Calibri", 12, "bold"))
extract_features_button.grid(row=1, column=1, padx=10, pady=5)

statistics_button = tk.Button(button_frame, text="Select image to view it's Statistics", command=select_image_and_extract_features, bg="#5dade2", fg="white", relief="raised", width=40, font=("Calibri", 12, "bold"))
statistics_button.grid(row=1, column=2, padx=10, pady=5)

root.configure(bg="#ffffff")


# Start the main loop
root.mainloop()
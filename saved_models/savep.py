import tensorflow as tf

# Specify the absolute path to the SavedModel directory
saved_model_path = r"C:\AIML\potato-disease-classification\saved_models\1"

# Load the SavedModel
try:
    model = tf.saved_model.load(saved_model_path)
    print("Model loaded successfully.")
except OSError as e:
    print(f"Error loading the SavedModel: {e}")

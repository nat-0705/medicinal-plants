from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained CNN model
model = tf.keras.models.load_model(r"C:\Users\marko\Documents\School\flask-api\model\medicinal_plant_cnn.h5")

# Define allowed image size (must match model input shape)
IMG_SIZE = (224, 224)

@app.route("/predict", methods=["POST"])  # ✅ Make sure this allows POST
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize(IMG_SIZE)  # Resize to model input size
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Map index to class name (adjust this based on your dataset)
        classes = ['Lemongrass', 'Basil', 'Mint', 'Acapulco', 'Pandan', 'Turmeric', 'Goathe', 'Aloe Vera', 'Oregano', 'Gensing']
        plant_name = classes[class_index]

        return jsonify({"class": plant_name, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# ✅ Load the model only once (avoid reloading)
MODEL_PATH = "model/medicinal_plant_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize(IMG_SIZE)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        classes = ['Lemongrass', 'Basil', 'Mint', 'Acapulco', 'Pandan', 'Turmeric', 'Goathe', 'Aloe Vera', 'Oregano', 'Ginseng']
        plant_name = classes[class_index]

        # ✅ Free up TensorFlow memory after prediction
        tf.keras.backend.clear_session()

        return jsonify({"class": plant_name, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=PORT, debug=False)

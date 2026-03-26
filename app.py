from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import requests
from geopy.distance import geodesic
import uuid
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# -------------------- APP SETUP --------------------
app = Flask(__name__)
CORS(app)

# -------------------- OPENAI SETUP --------------------
import os
api_key = os.getenv("OPENAI_API_KEY")

# -------------------- LOAD MODEL --------------------
model = load_model("waste_model.h5")
model.trainable = False

classes = ["glass", "organic", "paper", "plastic"]

CONFIDENCE_THRESHOLD = 40.0

# -------------------- DAILY DASHBOARD --------------------
daily_stats = {
    "total_items": 0,
    "total_co2": 0.0,
    "total_points": 0
}

# -------------------- ECO DATA --------------------
ECO_INFO = {
    "plastic": {"harm": "High", "recyclable": "Yes", "co2": 2.5, "points": 10},
    "paper": {"harm": "Low", "recyclable": "Yes", "co2": 1.2, "points": 5},
    "glass": {"harm": "Medium", "recyclable": "Yes", "co2": 1.8, "points": 8},
    "organic": {"harm": "Low", "recyclable": "Compostable", "co2": 0.8, "points": 4}
}

# -------------------- RECYCLING CENTER DATABASE --------------------
RECYCLING_CENTERS = {
    "Chennai": [
        {"name": "Perungudi Waste Processing Facility", "lat": 12.9121, "lon": 80.2295},
        {"name": "Kodungaiyur Dump Yard", "lat": 13.1387, "lon": 80.2485}
    ],
    "Bangalore": [
        {"name": "Mavallipura Landfill", "lat": 13.1823, "lon": 77.4964}
    ]
}

# -------------------- AUTO LOCATION --------------------
def get_user_location():
    try:
        res = requests.get("http://ip-api.com/json/", timeout=3).json()
        return {
            "city": res.get("city", "Unknown"),
            "lat": res.get("lat"),
            "lon": res.get("lon")
        }
    except:
        return {"city": "Unknown", "lat": None, "lon": None}

# -------------------- REAL NEAREST CENTER --------------------
def get_nearest_recycling_center(lat, lon, city):
    if lat is None or lon is None:
        return None

    if city not in RECYCLING_CENTERS:
        return None

    user_coords = (lat, lon)
    nearest = None
    min_distance = float("inf")

    for center in RECYCLING_CENTERS[city]:
        center_coords = (center["lat"], center["lon"])
        distance = geodesic(user_coords, center_coords).km

        if distance < min_distance:
            min_distance = distance
            nearest = center

    if nearest:
        return {
            "name": nearest["name"],
            "city": city,
            "distance_km": round(min_distance, 2),
            "lat": nearest["lat"],
            "lon": nearest["lon"]
        }

    return None

# -------------------- LLM EXPLANATION --------------------
def generate_llm_explanation(waste, confidence, harm, co2):
    try:
        prompt = f"""
You are an environmental sustainability expert.

Provide a short eco explanation for:

Waste type: {waste}
Model confidence: {confidence}%
Environmental harm level: {harm}
CO2 savings potential: {co2} kg

Requirements:
- Professional but simple
- Mention sustainability impact
- Include one educational environmental fact
- Maximum 4 sentences
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sustainability expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("LLM ERROR:", e)
        return (
            f"The waste is identified as {waste}. "
            f"It has a {harm} environmental harm level. "
            f"Proper disposal can save approximately {co2} kg of CO₂ emissions."
        )

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return "Eco Waste Classifier Backend is LIVE ♻️"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        img_file = request.files["image"]

        temp_filename = f"{uuid.uuid4().hex}.jpg"
        img_file.save(temp_filename)

        # Preprocess
        img = image.load_img(temp_filename, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        img_arr = preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)

        preds = model.predict(img_arr, verbose=0)
        preds = preds[0]

        os.remove(temp_filename)

        predicted_index = int(np.argmax(preds))
        confidence = round(float(preds[predicted_index]) * 100, 2)

        location = get_user_location()
        center = get_nearest_recycling_center(
            location["lat"], location["lon"], location["city"]
        )

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "waste_type": "Unknown",
                "confidence": confidence,
                "harm_level": "Unknown",
                "recyclable": "Unknown",
                "eco_explanation": "Image unclear. Please upload a clearer waste image.",
                "co2_saved": 0,
                "reward_points": 0,
                "nearest_center": center,
                "daily_dashboard": daily_stats
            })

        predicted = classes[predicted_index]
        info = ECO_INFO[predicted]

        daily_stats["total_items"] += 1
        daily_stats["total_co2"] += info["co2"]
        daily_stats["total_points"] += info["points"]

        explanation = generate_llm_explanation(
            predicted.capitalize(),
            confidence,
            info["harm"],
            info["co2"]
        )

        return jsonify({
            "waste_type": predicted.capitalize(),
            "confidence": confidence,
            "harm_level": info["harm"],
            "recyclable": info["recyclable"],
            "eco_explanation": explanation,
            "co2_saved": info["co2"],
            "reward_points": info["points"],
            "nearest_center": center,
            "daily_dashboard": daily_stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
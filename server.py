import os
from dotenv import load_dotenv
load_dotenv()
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
THINGSPEAK_CH_ID    = os.environ.get("THINGSPEAK_CH_ID", "3274778")
THINGSPEAK_READ_KEY = os.environ.get("THINGSPEAK_READ_KEY", "")

FIELD_MAP = {
    "field1": ("DS18B20 Temperature", "C"),
    "field2": ("Watermark CB Value",  "cb"),
    "field3": ("NPK Moisture",        "%"),
    "field4": ("NPK pH",              ""),
    "field5": ("Nitrogen (N)",        "mg/kg"),
    "field6": ("Phosphorus (P)",      "mg/kg"),
    "field7": ("Potassium (K)",       "mg/kg"),
    "field8": ("Watermark Moisture",  "%"),
}

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"

def fetch_sensor_data():
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CH_ID}/feeds.json?days=7"
    if THINGSPEAK_READ_KEY:
        url += f"&api_key={THINGSPEAK_READ_KEY}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        feeds = data.get("feeds", [])
        if not feeds:
            return {"error": "No data found"}
            
        latest = feeds[-1]
        oldest = feeds[0]
        result = {}
        trend_text = []

        for field, (name, unit) in FIELD_MAP.items():
            val = latest.get(field)
            result[name] = f"{val} {unit}".strip() if val not in (None, "") else "N/A"
            
            # Trend calculation
            old_val = oldest.get(field)
            if val and old_val and val != "N/A" and old_val != "N/A":
                try:
                    diff = float(val) - float(old_val)
                    if abs(diff) > 0.1:
                        direction = "increased" if diff > 0 else "decreased"
                        trend_text.append(f"{name} {direction} by {abs(diff):.1f} {unit}.")
                except ValueError:
                    pass

        result["_timestamp"] = latest.get("created_at", "unknown")
        result["_trend"] = " ".join(trend_text) if trend_text else "Conditions are completely stable."
        
        # --- ML Prediction (1, 7, 10 day future projection) ---
        if len(feeds) > 5:
            try:
                times = []
                for f in feeds:
                    dt = datetime.strptime(f["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                    times.append(dt.timestamp())
                
                X = np.array(times).reshape(-1, 1)
                X_hours = (X - X[0][0]) / 3600.0
                
                last_hour = X_hours[-1][0]
                target_X = np.array([
                    [last_hour + 24.0],   # 1 Day (24 hours)
                    [last_hour + 168.0],  # 7 Days (168 hours)
                    [last_hour + 240.0]   # 10 Days (240 hours)
                ])
                
                ml_preds = []
                for field, (name, unit) in FIELD_MAP.items():
                    y = []
                    valid_x = []
                    for i, f in enumerate(feeds):
                        val = f.get(field)
                        if val not in (None, "", "N/A"):
                            try:
                                y.append(float(val))
                                valid_x.append(X_hours[i][0])
                            except ValueError:
                                pass
                    
                    if len(y) > 10:
                        model_lr = LinearRegression()
                        model_lr.fit(np.array(valid_x).reshape(-1, 1), np.array(y))
                        preds = model_lr.predict(target_X)
                        
                        def bound_val(p):
                            if "%" in unit: return max(0.0, min(100.0, p))
                            elif "cb" in unit: return max(0.0, min(240.0, p))
                            elif name == "NPK pH": return max(0.0, min(14.0, p))
                            elif "Temp" in name: return max(-10.0, min(60.0, p))
                            return max(0.0, p)
                            
                        pred_1d = bound_val(preds[0])
                        pred_7d = bound_val(preds[1])
                        pred_10d = bound_val(preds[2])
                        
                        ml_preds.append(f"{name} (Day 1: {pred_1d:.1f}, Day 7: {pred_7d:.1f}, Day 10: {pred_10d:.1f}{unit})")

                result["_ml_prediction"] = " | ".join(ml_preds) if ml_preds else "Not enough numerical data for ML."
            except Exception as e:
                result["_ml_prediction"] = f"[ML Error: {e}]"

        return result
    except Exception as e:
        return {"error": str(e)}

def fetch_weather_data():
    # Coordinates for C-DAC Tukkuguda, Hyderabad
    url = "https://api.open-meteo.com/v1/forecast?latitude=17.188&longitude=78.468&current=temperature_2m,precipitation&daily=precipitation_probability_max&timezone=Asia%2FKolkata&forecast_days=2"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return "Weather data unavailable."
        d = r.json()
        c_temp = d["current"]["temperature_2m"]
        c_rain = d["current"]["precipitation"]
        p_today = d["daily"]["precipitation_probability_max"][0]
        p_tmrw = d["daily"]["precipitation_probability_max"][1]
        return f"Current Temp: {c_temp}°C, Rain right now: {c_rain}mm. Rain chance: {p_today}% today, {p_tmrw}% tomorrow."
    except Exception as e:
        return f"[Weather error: {e}]"

def build_prompt(sensor_data, weather_text="", crop_info="Generic Crop Profile", language="English"):
    if "error" in sensor_data:
        sensor_text = f"[ThingSpeak error: {sensor_data['error']}]"
    else:
        lines = [f"  - {k}: {v}" for k, v in sensor_data.items() if not k.startswith("_")]
        sensor_text = f"Last updated: {sensor_data.get('_timestamp','')}\n" + "\n".join(lines)
        if "_trend" in sensor_data:
            sensor_text += f"\n\nRECENT TRENDS (last 7 days):\n{sensor_data['_trend']}"
        if "_history" in sensor_data and sensor_data["_history"]:
            sensor_text += f"\n\nHISTORICAL DATA POINTS:\n{sensor_data['_history']}"

    # Inject ML predictions into the prompt
    prediction_text = ""
    if "_ml_prediction" in sensor_data and sensor_data["_ml_prediction"]:
        prediction_text = f"""
ML-COMPUTED PREDICTIONS (Linear regression over last 7 days of real sensor data):
{sensor_data['_ml_prediction']}

These are real calculated predictions. When user asks about future values or 7-day prediction,
use EXACTLY these numbers. Do not make up different values.
"""

    return f"""You are SoilBot, an expert AI assistant for soil health and precision agriculture.
You have access to LIVE sensor readings from an IoT soil monitoring system (ESP32 + LoRaWAN).

CURRENT LIVE SENSOR DATA:
{sensor_text}

WEATHER FORECAST (Local Area):
{weather_text}

Crop Config and Container Size:
{crop_info}
{prediction_text}
SENSOR GUIDE:
- DS18B20 Temperature: Soil temperature probe
- Watermark CB: Water tension. 0-10=saturated, 10-30=optimal, 30-60=drying, above 60=dry stress
- NPK pH: Ideal 6.0-7.0 for most crops
- Nitrogen: Ideal 140-200 mg/kg for spinach. Phosphorus: 30-60. Potassium: 150-250.

INSTRUCTIONS:
- Answer questions concisely and directly. Do not add unnecessary preamble.
- Give practical farming advice based on live data and the current crop.
- If the user has a specific crop configuration (like Spinach in a 5x3x1 bed), tailor watering volume and fertilizer advice precisely matching that geometry.
- If sensor shows N/A, say it is temporarily unavailable.
- When predicting future values, use the ML-COMPUTED PREDICTIONS section above. Quote those exact numbers.

CRITICAL FORMATTING RULES:
1. NEVER use the asterisk or star symbol anywhere in your response. Not for bold, not for bullets.
2. NEVER use markdown formatting of any kind (no **, no ##, no __, no backticks).
3. Structure your answers using plain paragraphs with blank lines between them.
4. For lists, use hyphens (-) or numbers (1., 2., 3.).
5. Keep responses short and to the point. Maximum 6-8 lines unless a full summary is requested.
6. Respond completely in: {language}.
"""

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/logo.png")
def serve_logo():
    return send_from_directory(".", "soilbot_logo.png")

@app.route("/api/sensor-data")
def get_sensor_data():
    return jsonify(fetch_sensor_data())

@app.route("/api/history")
def get_history():
    """Returns raw 7-day feed data for the frontend chart."""
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CH_ID}/feeds.json?days=7&results=500"
    if THINGSPEAK_READ_KEY:
        url += f"&api_key={THINGSPEAK_READ_KEY}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    crop_info = data.get("crop_info", "Generic Crop Profile")
    language = data.get("language", "English")
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
        
    sensor_data = fetch_sensor_data()
    weather_data = fetch_weather_data()
    system_prompt = build_prompt(sensor_data, weather_data, crop_info, language)
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply, "sensor_data": sensor_data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\nSoilBot running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)

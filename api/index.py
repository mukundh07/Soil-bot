import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from datetime import datetime

app = Flask(__name__)
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

client = None
MODEL = "gpt-4o-mini"

def get_client():
    global client
    if client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=key)
    return client

def fetch_sensor_data():
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CH_ID}/feeds.json?results=50"
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
        
        # Build a simple history summary for ChatGPT to analyze
        history_points = []
        step = max(1, len(feeds) // 5)
        for i in range(0, len(feeds), step):
            f = feeds[i]
            ts = f.get("created_at", "")
            vals = []
            for field, (name, unit) in FIELD_MAP.items():
                v = f.get(field)
                if v not in (None, ""):
                    vals.append(f"{name}={v}{unit}")
            if vals:
                history_points.append(f"[{ts}] {', '.join(vals)}")
        
        result["_history"] = " | ".join(history_points) if history_points else ""

        return result
    except Exception as e:
        return {"error": str(e)}

def fetch_weather_data():
    url = "https://api.open-meteo.com/v1/forecast?latitude=17.188&longitude=78.468&current=temperature_2m,precipitation&daily=precipitation_probability_max&timezone=Asia%2FKolkata&forecast_days=2"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return "Weather data unavailable."
        d = r.json()
        c_temp = d["current"]["temperature_2m"]
        c_rain = d["current"]["precipitation"]
        p_today = d["daily"]["precipitation_probability_max"][0]
        p_tmrw = d["daily"]["precipitation_probability_max"][1]
        return f"Current Temp: {c_temp} C, Rain right now: {c_rain}mm. Rain chance: {p_today}% today, {p_tmrw}% tomorrow."
    except Exception as e:
        return f"[Weather error: {e}]"

def build_prompt(sensor_data, weather_text=""):
    if "error" in sensor_data:
        sensor_text = f"[ThingSpeak error: {sensor_data['error']}]"
    else:
        lines = [f"  * {k}: {v}" for k, v in sensor_data.items() if not k.startswith("_")]
        sensor_text = f"Last updated: {sensor_data.get('_timestamp','')}\n" + "\n".join(lines)
        if "_trend" in sensor_data:
            sensor_text += f"\n  * RECENT TRENDS: {sensor_data['_trend']}"
        if "_history" in sensor_data and sensor_data["_history"]:
            sensor_text += f"\n  * HISTORICAL DATA POINTS: {sensor_data['_history']}"
            
    return f"""You are SoilBot, an expert AI assistant for soil health and precision agriculture.
You have access to LIVE sensor readings from an IoT soil monitoring system (ESP32 + LoRaWAN).

CURRENT LIVE SENSOR DATA:
{sensor_text}

WEATHER FORECAST (Local Area):
{weather_text}

SENSOR GUIDE:
- DS18B20 Temperature: Soil temperature probe
- Watermark CB: Water tension. 0-10=saturated, 10-30=optimal, 30-60=drying, >60=dry stress
- NPK pH: Ideal 6.0-7.0 for most crops
- Nitrogen: Ideal 140-200 mg/kg for spinach. Phosphorus: 30-60. Potassium: 150-250.

INSTRUCTIONS:
- Give practical farming advice based on live data. Be friendly and concise.
- If sensor shows N/A, say it is temporarily unavailable.
- When asked to predict future values, analyze the historical data points and trends provided above, then calculate and provide specific predicted numbers for Day 1, Day 7, and Day 10.

CRITICAL FORMATTING RULES:
1. NEVER use the asterisk/star symbol (*) anywhere in your response. Do not use ** for bolding. Do not use * for bullet points.
2. Structure your answers clearly using paragraphs with empty line breaks.
3. If making a list, use standard numbers (1., 2., 3.) or simple hyphens (-). 
4. The output must be perfectly clean plain text that is easy to read.
"""

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SoilBot - AI Soil Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Inter", sans-serif;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }
  .header {
    text-align: center;
    margin-bottom: 20px;
    color: white;
  }
  .header .logo { font-size: 42px; margin-bottom: 8px; }
  .header h1 { font-size: 28px; font-weight: 700; }
  .header p { font-size: 14px; opacity: 0.75; margin-top: 4px; }

  .sensor-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    max-width: 900px;
    width: 100%;
    margin-bottom: 20px;
  }
  .sensor-card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 10px 16px;
    color: white;
    text-align: center;
    min-width: 110px;
    transition: transform 0.2s;
  }
  .sensor-card:hover { transform: translateY(-2px); }
  .sensor-card .label { font-size: 11px; opacity: 0.7; margin-bottom: 4px; }
  .sensor-card .value { font-size: 18px; font-weight: 700; color: #4ade80; }
  .sensor-card .value.na { color: #f87171; font-size: 14px; }
  .sensor-ts { color: rgba(255,255,255,0.5); font-size: 11px; text-align: center; margin-bottom: 16px; }

  .chat-container {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    width: 100%;
    max-width: 760px;
    display: flex;
    flex-direction: column;
    height: 520px;
    overflow: hidden;
  }
  .chat-header {
    padding: 16px 20px;
    background: rgba(74,222,128,0.15);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    gap: 10px;
    color: white;
  }
  .chat-header .dot {
    width: 10px; height: 10px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .chat-header span { font-weight: 600; font-size: 15px; }
  .chat-header small { opacity: 0.6; font-size: 12px; margin-left: auto; }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.2) transparent;
  }
  .message { display: flex; gap: 10px; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  .message.user { flex-direction: row-reverse; }
  .avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
  }
  .message.bot .avatar { background: rgba(74,222,128,0.2); }
  .message.user .avatar { background: rgba(99,102,241,0.3); }
  .bubble {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.6;
    color: white;
  }
  .message.bot .bubble {
    background: rgba(255,255,255,0.1);
    border-bottom-left-radius: 4px;
  }
  .message.user .bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-bottom-right-radius: 4px;
  }
  .typing-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #4ade80; border-radius: 50%; margin: 0 2px;
    animation: bounce 1.2s infinite;
  }
  .typing-dot:nth-child(2){animation-delay:.2s}
  .typing-dot:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-8px)}}

  .input-area {
    padding: 16px 20px;
    border-top: 1px solid rgba(255,255,255,0.1);
    display: flex;
    gap: 10px;
  }
  .input-area input {
    flex: 1;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 50px;
    padding: 12px 20px;
    color: white;
    font-size: 14px;
    font-family: "Inter", sans-serif;
    outline: none;
    transition: border-color 0.2s;
  }
  .input-area input::placeholder { color: rgba(255,255,255,0.4); }
  .input-area input:focus { border-color: #4ade80; }
  .send-btn {
    background: linear-gradient(135deg, #4ade80, #22c55e);
    border: none;
    border-radius: 50%;
    width: 46px; height: 46px;
    cursor: pointer;
    font-size: 18px;
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.2s, opacity 0.2s;
    flex-shrink: 0;
  }
  .send-btn:hover { transform: scale(1.08); }
  .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  .suggestions {
    display: flex; flex-wrap: wrap; gap: 8px;
    padding: 0 20px 14px;
  }
  .suggestion {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50px;
    padding: 6px 14px;
    color: rgba(255,255,255,0.8);
    font-size: 12px;
    cursor: pointer;
    transition: background 0.2s;
  }
  .suggestion:hover { background: rgba(74,222,128,0.2); color: white; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">🌱</div>
  <h1>SoilBot</h1>
  <p>AI-Powered Soil Health Assistant · Live IoT Data</p>
</div>

<div class="sensor-bar" id="sensorBar">
  <div class="sensor-card"><div class="label">Loading...</div><div class="value">--</div></div>
</div>
<div class="sensor-ts" id="sensorTs">Fetching live data...</div>

<div class="chat-container">
  <div class="chat-header">
    <div class="dot"></div>
    <span>SoilBot AI</span>
    <small>Powered by ChatGPT + ThingSpeak</small>
  </div>

  <div class="messages" id="messages">
    <div class="message bot">
      <div class="avatar">🤖</div>
      <div class="bubble">Hello! I am SoilBot 🌱 I can see your live soil sensor data from ThingSpeak. Ask me anything about your soil health, irrigation, or fertilizer needs!</div>
    </div>
  </div>

  <div class="suggestions" id="suggestions">
    <span class="suggestion" onclick="sendSuggestion(this)">Is my soil moist enough?</span>
    <span class="suggestion" onclick="sendSuggestion(this)">What is the pH level?</span>
    <span class="suggestion" onclick="sendSuggestion(this)">Do I need to fertilize?</span>
    <span class="suggestion" onclick="sendSuggestion(this)">Should I water now?</span>
    <span class="suggestion" onclick="sendSuggestion(this)">Summarize all readings</span>
  </div>

  <div class="input-area">
    <input type="text" id="userInput" placeholder="Ask about your soil..." onkeydown="if(event.key==='Enter')sendMessage()">
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
  </div>
</div>

<script>
  const BASE = "";
  let history = [];

  async function loadSensors() {
    try {
      const r = await fetch(BASE + "/api/sensor-data");
      const d = await r.json();
      const bar = document.getElementById("sensorBar");
      const ts  = document.getElementById("sensorTs");
      if (d.error) { ts.textContent = "Sensor fetch error: " + d.error; return; }
      ts.textContent = "Last updated: " + (d._timestamp || "unknown");
      const icons = {"DS18B20 Temperature":"🌡️","Watermark CB Value":"💧","NPK Moisture":"💦","NPK pH":"⚗️","Nitrogen (N)":"🌿","Phosphorus (P)":"🌾","Potassium (K)":"🍃","Watermark Moisture":"💦"};
      bar.innerHTML = "";
      for (const [k, v] of Object.entries(d)) {
        if (k.startsWith("_")) continue;
        const isNA = v === "N/A";
        bar.innerHTML += `<div class="sensor-card"><div class="label">${icons[k]||""} ${k}</div><div class="value ${isNA?"na":""}">${v}</div></div>`;
      }
    } catch(e) {
      document.getElementById("sensorTs").textContent = "Could not fetch sensors.";
    }
  }

  function addMessage(role, text) {
    const msgs = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message " + role;
    div.innerHTML = `<div class="avatar">${role==="user"?"👤":"🤖"}</div><div class="bubble">${text.replace(/\\n/g,"<br>")}</div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function showTyping() {
    const msgs = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message bot"; div.id = "typing";
    div.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function removeTyping() { const t = document.getElementById("typing"); if(t) t.remove(); }

  async function sendMessage() {
    const input = document.getElementById("userInput");
    const btn   = document.getElementById("sendBtn");
    const msg   = input.value.trim();
    if (!msg) return;
    input.value = ""; btn.disabled = true;
    document.getElementById("suggestions").style.display = "none";
    addMessage("user", msg);
    history.push({role:"user", text:msg});
    showTyping();
    try {
      const r = await fetch(BASE + "/api/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message: msg, history: history})
      });
      const d = await r.json();
      removeTyping();
      if (d.error) { addMessage("bot", "Error: " + d.error); }
      else {
        addMessage("bot", d.reply);
        history.push({role:"bot", text: d.reply});
      }
    } catch(e) {
      removeTyping();
      addMessage("bot", "Connection error — is the server running?");
    }
    btn.disabled = false;
    input.focus();
  }

  function sendSuggestion(el) {
    document.getElementById("userInput").value = el.textContent;
    sendMessage();
  }

  loadSensors();
  setInterval(loadSensors, 30000);
</script>
</body>
</html>"""

@app.route("/")
def serve_index():
    return HTML_CONTENT

@app.route("/api/sensor-data")

def get_sensor_data():
    return jsonify(fetch_sensor_data())

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
        
    sensor_data = fetch_sensor_data()
    weather_data = fetch_weather_data()
    system_prompt = build_prompt(sensor_data, weather_data)
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply, "sensor_data": sensor_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

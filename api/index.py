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
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CH_ID}/feeds.json?results=150"
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

def build_prompt(sensor_data, weather_text="", crop_info="Generic Crop Profile", language="English"):
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

Crop Config & Container Size:
{crop_info}

SENSOR GUIDE:
- DS18B20 Temperature: Soil temperature probe
- Watermark CB: Water tension. 0-10=saturated, 10-30=optimal, 30-60=drying, >60=dry stress
- NPK pH: Ideal 6.0-7.0 for most crops
- Nitrogen: Ideal 140-200 mg/kg for spinach. Phosphorus: 30-60. Potassium: 150-250.

INSTRUCTIONS:
- Give practical farming advice based on live data and the current crop. Be friendly and concise.
- If the user has a specific crop configuration provided (like Spinach in a 5x3x1 bed), tailor watering volume and fertilizer advice precisely matching that geometry.
- If sensor shows N/A, say it is temporarily unavailable.
- When asked to predict future values, analyze the historical data points and trends provided above, then calculate and provide specific predicted numbers for Day 1, Day 7, and Day 10.

CRITICAL FORMATTING RULES:
1. NEVER use the asterisk/star symbol (*) anywhere in your response. Do not use ** for bolding. Do not use * for bullet points.
2. Structure your answers clearly using paragraphs with empty line breaks.
3. If making a list, use standard numbers (1., 2., 3.) or simple hyphens (-). 
4. The output must be perfectly clean plain text that is easy to read.
5. You MUST respond completely in the requested language: {language}. Translate your expert agricultural advice fluently into {language}.
"""

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SoilBot — Smart Soil Dashboard</title>
<meta name="description" content="AI-powered precision agriculture dashboard with live IoT sensor monitoring">
<link rel="icon" type="image/png" href="/logo.png">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  /* ═══════ RESET & BASE ═══════ */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Inter", -apple-system, sans-serif;
    background: #080b14;
    color: #e2e8f0;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ═══════ TOP NAV BAR ═══════ */
  .navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: rgba(8,11,20,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .navbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .navbar-brand img {
    width: 38px; height: 38px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(74,222,128,0.3);
  }
  .navbar-brand h1 {
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
  }
  .navbar-controls {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .nav-select {
    padding: 6px 12px;
    border-radius: 8px;
    background: rgba(255,255,255,0.06);
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.1);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    outline: none;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .nav-select:hover { border-color: rgba(74,222,128,0.4); }
  .nav-select option { color: #111; background: #1e293b; }
  .live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .live-dot {
    width: 8px; height: 8px;
    background: #4ade80;
    border-radius: 50%;
    animation: livePulse 2s infinite;
    box-shadow: 0 0 8px rgba(74,222,128,0.6);
  }
  @keyframes livePulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }

  /* ═══════ MAIN LAYOUT ═══════ */
  .dashboard {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* ═══════ STATUS BAR ═══════ */
  .status-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
  }
  .status-bar .ts {
    font-size: 12px;
    color: rgba(255,255,255,0.4);
  }
  .status-bar .system-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.3);
    font-weight: 600;
  }

  /* ═══════ SENSOR GRID ═══════ */
  .sensor-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 14px;
  }
  .sensor-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 18px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, border-color 0.3s, box-shadow 0.3s;
  }
  .sensor-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    border-radius: 16px 16px 0 0;
    opacity: 0;
    transition: opacity 0.3s;
  }
  .sensor-card:hover {
    transform: translateY(-3px);
    border-color: rgba(74,222,128,0.25);
    box-shadow: 0 8px 30px rgba(74,222,128,0.08);
  }
  .sensor-card:hover::before { opacity: 1; }
  .sensor-card .card-icon { font-size: 22px; margin-bottom: 8px; }
  .sensor-card .card-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(255,255,255,0.4);
    font-weight: 600;
    margin-bottom: 6px;
  }
  .sensor-card .card-value {
    font-size: 26px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
    line-height: 1;
  }
  .sensor-card .card-unit {
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    margin-top: 4px;
    font-weight: 500;
  }
  /* Status glow variants */
  .sensor-card.status-good { border-color: rgba(74,222,128,0.2); }
  .sensor-card.status-good .card-value { color: #4ade80; }
  .sensor-card.status-warn { border-color: rgba(250,204,21,0.25); }
  .sensor-card.status-warn .card-value { color: #facc15; }
  .sensor-card.status-bad { border-color: rgba(239,68,68,0.25); }
  .sensor-card.status-bad .card-value { color: #ef4444; }
  .sensor-card.status-na { border-color: rgba(255,255,255,0.06); }
  .sensor-card.status-na .card-value { color: rgba(255,255,255,0.2); font-size: 16px; }

  /* ═══════ CHAT PANEL ═══════ */
  .chat-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    display: flex;
    flex-direction: column;
    height: 480px;
    overflow: hidden;
  }
  .chat-topbar {
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .chat-topbar .ai-dot {
    width: 9px; height: 9px;
    background: #4ade80;
    border-radius: 50%;
    animation: livePulse 2s infinite;
    box-shadow: 0 0 6px rgba(74,222,128,0.5);
  }
  .chat-topbar .ai-label {
    font-size: 14px;
    font-weight: 600;
    color: #e2e8f0;
  }
  .chat-topbar .ai-sub {
    font-size: 11px;
    color: rgba(255,255,255,0.3);
    margin-left: auto;
    margin-right: 10px;
  }
  .voice-toggle {
    background: none;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    color: white;
    cursor: pointer;
    font-size: 16px;
    padding: 5px 8px;
    transition: background 0.2s, border-color 0.2s;
  }
  .voice-toggle:hover { background: rgba(255,255,255,0.06); border-color: rgba(74,222,128,0.3); }

  /* Messages */
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.1) transparent;
  }
  .message { display: flex; gap: 10px; animation: msgIn 0.35s ease; }
  @keyframes msgIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  .message.user { flex-direction: row-reverse; }
  .avatar {
    width: 32px; height: 32px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    overflow: hidden;
  }
  .message.bot .avatar { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.15); }
  .message.user .avatar { background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.2); }
  .bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 13.5px;
    line-height: 1.65;
    color: #e2e8f0;
  }
  .message.bot .bubble {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.06);
    border-top-left-radius: 4px;
  }
  .message.user .bubble {
    background: linear-gradient(135deg, rgba(79,70,229,0.4), rgba(124,58,237,0.4));
    border: 1px solid rgba(124,58,237,0.3);
    border-top-right-radius: 4px;
  }
  .typing-dot {
    display: inline-block; width: 7px; height: 7px;
    background: #4ade80; border-radius: 50%; margin: 0 2px;
    animation: bounce 1.2s infinite;
  }
  .typing-dot:nth-child(2){animation-delay:.2s}
  .typing-dot:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}

  /* Quick Actions */
  .quick-actions {
    display: flex; flex-wrap: wrap; gap: 8px;
    padding: 0 20px 14px;
  }
  .quick-btn {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 50px;
    padding: 7px 16px;
    color: rgba(255,255,255,0.6);
    font-size: 12px;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: all 0.2s;
  }
  .quick-btn:hover {
    background: rgba(74,222,128,0.1);
    border-color: rgba(74,222,128,0.25);
    color: #4ade80;
  }

  /* Input Area */
  .input-area {
    padding: 14px 18px;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .input-area input {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 12px 18px;
    color: #e2e8f0;
    font-size: 13.5px;
    font-family: "Inter", sans-serif;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .input-area input::placeholder { color: rgba(255,255,255,0.25); }
  .input-area input:focus {
    border-color: rgba(74,222,128,0.4);
    box-shadow: 0 0 0 3px rgba(74,222,128,0.08);
  }
  .mic-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 8px;
    font-size: 20px;
    cursor: pointer;
    transition: all 0.2s;
    outline: none;
  }
  .mic-btn:hover { background: rgba(255,255,255,0.08); border-color: rgba(74,222,128,0.3); }
  .mic-btn.recording {
    background: rgba(239,68,68,0.15);
    border-color: rgba(239,68,68,0.4);
    animation: pulseMic 1s infinite;
    box-shadow: 0 0 12px rgba(239,68,68,0.3);
  }
  @keyframes pulseMic { 0%{transform:scale(1)} 50%{transform:scale(1.08)} 100%{transform:scale(1)} }
  .send-btn {
    background: linear-gradient(135deg, #4ade80, #22c55e);
    border: none;
    border-radius: 12px;
    width: 44px; height: 44px;
    cursor: pointer;
    font-size: 17px;
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.2s, opacity 0.2s;
    flex-shrink: 0;
    box-shadow: 0 4px 16px rgba(74,222,128,0.25);
  }
  .send-btn:hover { transform: scale(1.06); }
  .send-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; box-shadow: none; }

  /* ═══════ RESPONSIVE ═══════ */
  @media (max-width: 768px) {
    .navbar { padding: 12px 16px; }
    .navbar-brand h1 { font-size: 17px; }
    .dashboard { padding: 16px 12px; }
    .sensor-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .sensor-card .card-value { font-size: 22px; }
    .chat-panel { height: 440px; }
    .nav-select { font-size: 12px; padding: 5px 8px; }
  }
  @media (max-width: 480px) {
    .sensor-grid { grid-template-columns: repeat(2, 1fr); }
    .navbar-controls { gap: 8px; }
    .live-badge span { display: none; }
  }
</style>
</head>
<body>

<!-- ═══════ NAVIGATION ═══════ -->
<nav class="navbar">
  <div class="navbar-brand">
    <img src="/logo.png" alt="SoilBot">
    <h1>SoilBot</h1>
  </div>
  <div class="navbar-controls">
    <select id="cropSelect" class="nav-select">
      <option value="Spinach (Area: 5ft x 3ft, Depth: 1ft)" selected>🥬 Spinach</option>
      <option value="Tomatoes (Standard spacing)">🍅 Tomatoes</option>
      <option value="Generic Crop Profile">🌱 Generic</option>
    </select>
    <select id="langSelect" class="nav-select" onchange="updateVoiceLang()">
      <option value="English" selected>EN</option>
      <option value="Hindi">हि</option>
      <option value="Telugu">తె</option>
    </select>
    <div class="live-badge">
      <div class="live-dot"></div>
      <span>LIVE</span>
    </div>
  </div>
</nav>

<!-- ═══════ DASHBOARD ═══════ -->
<main class="dashboard">

  <!-- Status Bar -->
  <div class="status-bar">
    <span class="system-label">Soil Monitoring System</span>
    <span class="ts" id="sensorTs">Connecting to sensors...</span>
  </div>

  <!-- Sensor Grid -->
  <div class="sensor-grid" id="sensorGrid">
    <div class="sensor-card status-na">
      <div class="card-icon">📡</div>
      <div class="card-label">Awaiting Data</div>
      <div class="card-value">--</div>
      <div class="card-unit">loading</div>
    </div>
  </div>

  <!-- Chat Panel -->
  <div class="chat-panel">
    <div class="chat-topbar">
      <div class="ai-dot"></div>
      <span class="ai-label">SoilBot AI</span>
      <span class="ai-sub">GPT-4o · Real-time Analysis</span>
      <button id="voiceToggleBtn" class="voice-toggle" onclick="toggleVoiceOutput()" title="Toggle AI Voice">🔊</button>
    </div>

    <div class="messages" id="messages">
      <div class="message bot">
        <div class="avatar"><img src="/logo.png" style="width:100%; height:100%; border-radius:10px; object-fit:cover;"></div>
        <div class="bubble">Welcome to SoilBot 🌱 I have access to your live sensor data from the field. Ask me anything about soil health, irrigation timing, or fertilizer recommendations.</div>
      </div>
    </div>

    <div class="quick-actions" id="suggestions">
      <button class="quick-btn" onclick="sendSuggestion(this)">💧 Should I water?</button>
      <button class="quick-btn" onclick="sendSuggestion(this)">⚗️ pH Analysis</button>
      <button class="quick-btn" onclick="sendSuggestion(this)">🌿 Fertilizer Plan</button>
      <button class="quick-btn" onclick="sendSuggestion(this)">📊 Full Summary</button>
      <button class="quick-btn" onclick="sendSuggestion(this)">🔮 7-Day Prediction</button>
    </div>

    <div class="input-area">
      <button class="mic-btn" id="micBtn" onclick="toggleMic()" title="Voice Input">🎤</button>
      <input type="text" id="userInput" placeholder="Ask about your soil..." onkeydown="if(event.key==='Enter')sendMessage()">
      <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
    </div>
  </div>

</main>

<script>
  const BASE = "";
  let history = [];
  const langCodes = { "English": "en-US", "Hindi": "hi-IN", "Telugu": "te-IN" };

  function updateVoiceLang() {
    const selectedLang = document.getElementById("langSelect").value;
    if (recognition) {
       recognition.lang = langCodes[selectedLang] || "en-US";
    }
  }

  // ═══════ Sensor Status Logic ═══════
  function getSensorStatus(name, rawVal) {
    const v = parseFloat(rawVal);
    if (isNaN(v)) return "na";
    if (name.includes("Temperature")) return (v >= 18 && v <= 35) ? "good" : (v >= 10 && v <= 40) ? "warn" : "bad";
    if (name.includes("pH")) return (v >= 6.0 && v <= 7.0) ? "good" : (v >= 5.0 && v <= 8.0) ? "warn" : "bad";
    if (name.includes("Moisture") && !name.includes("CB")) return (v >= 40) ? "good" : (v >= 20) ? "warn" : "bad";
    if (name.includes("CB")) return (v <= 30) ? "good" : (v <= 60) ? "warn" : "bad";
    if (name.includes("Nitrogen") || name.includes("Phosphorus") || name.includes("Potassium")) return (v > 0) ? "good" : "warn";
    return "good";
  }

  const sensorIcons = {
    "DS18B20 Temperature":"🌡️","Watermark CB Value":"💧","NPK Moisture":"💦",
    "NPK pH":"⚗️","Nitrogen (N)":"🌿","Phosphorus (P)":"🧪","Potassium (K)":"🍃","Watermark Moisture":"💦"
  };

  async function loadSensors() {
    try {
      const r = await fetch(BASE + "/api/sensor-data");
      const d = await r.json();
      const grid = document.getElementById("sensorGrid");
      const ts   = document.getElementById("sensorTs");
      if (d.error) { ts.textContent = "Sensor error: " + d.error; return; }

      // Format timestamp
      const raw = d._timestamp || "unknown";
      if (raw !== "unknown") {
        const dt = new Date(raw);
        const diff = Math.floor((Date.now() - dt.getTime()) / 60000);
        ts.textContent = diff < 1 ? "Updated just now" : `Updated ${diff} min ago · ${dt.toLocaleTimeString()}`;
      } else {
        ts.textContent = "Last updated: " + raw;
      }

      grid.innerHTML = "";
      for (const [k, v] of Object.entries(d)) {
        if (k.startsWith("_")) continue;
        const isNA = v === "N/A";
        const numVal = v.replace(/[^\d.-]/g, '');
        const unit = v.replace(/[\d.-]/g, '').trim();
        const status = isNA ? "na" : getSensorStatus(k, numVal);
        grid.innerHTML += `<div class="sensor-card status-${status}">
          <div class="card-icon">${sensorIcons[k]||"📈"}</div>
          <div class="card-label">${k.replace("DS18B20 ","").replace("Watermark ","WM ").replace("NPK ","")}</div>
          <div class="card-value">${isNA ? "N/A" : numVal}</div>
          <div class="card-unit">${isNA ? "offline" : unit}</div>
        </div>`;
      }
    } catch(e) {
      document.getElementById("sensorTs").textContent = "Could not fetch sensors.";
    }
  }

  function addMessage(role, text) {
    const msgs = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message " + role;
    div.innerHTML = `<div class="avatar">${role==="user"?"👤":"<img src='/logo.png' style='width:100%; height:100%; border-radius:10px; object-fit:cover;'>"}</div><div class="bubble">${text.replace(/\n/g,"<br>")}</div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function showTyping() {
    const msgs = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message bot"; div.id = "typing";
    div.innerHTML = `<div class="avatar"><img src='/logo.png' style='width:100%; height:100%; border-radius:10px; object-fit:cover;'></div><div class="bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function removeTyping() { const t = document.getElementById("typing"); if(t) t.remove(); }

  async function sendMessage() {
    const input = document.getElementById("userInput");
    const btn   = document.getElementById("sendBtn");
    const msg   = input.value.trim();
    if (!msg) return;

    const cropInfo = document.getElementById("cropSelect").value;
    const langInfo = document.getElementById("langSelect").value;

    input.value = ""; btn.disabled = true;
    document.getElementById("suggestions").style.display = "none";
    addMessage("user", msg);
    history.push({role:"user", text:msg});
    showTyping();
    try {
      const r = await fetch(BASE + "/api/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message: msg, history: history, crop_info: cropInfo, language: langInfo})
      });
      const d = await r.json();
      removeTyping();
      if (d.error) { addMessage("bot", "Error: " + d.error); }
      else {
        addMessage("bot", d.reply);
        history.push({role:"bot", text: d.reply});
        readOutLoud(d.reply);
      }
    } catch(e) {
      removeTyping();
      addMessage("bot", "Connection error — is the server running?");
    }
    btn.disabled = false;
    input.focus();
  }

  function sendSuggestion(el) {
    document.getElementById("userInput").value = el.textContent.replace(/^[^\s]+\s/, '');
    sendMessage();
  }

  let voiceOutputEnabled = true;
  function toggleVoiceOutput() {
    voiceOutputEnabled = !voiceOutputEnabled;
    const btn = document.getElementById("voiceToggleBtn");
    btn.textContent = voiceOutputEnabled ? "🔊" : "🔈";
    btn.style.opacity = voiceOutputEnabled ? "1" : "0.5";
    if(!voiceOutputEnabled) window.speechSynthesis.cancel();
  }

  function readOutLoud(text) {
    if(!voiceOutputEnabled) return;
    window.speechSynthesis.cancel();
    const cleanText = text.replace(/[*#_`~]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    const selectedLang = document.getElementById("langSelect").value;
    utterance.lang = langCodes[selectedLang] || "en-US";
    window.speechSynthesis.speak(utterance);
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = langCodes[document.getElementById("langSelect").value] || 'en-US';
    recognition.onresult = function(event) {
      const transcript = event.results[0][0].transcript;
      document.getElementById("userInput").value = transcript;
      sendMessage();
    };
    recognition.onstart = function() {
      document.getElementById("micBtn").classList.add("recording");
      document.getElementById("userInput").placeholder = "Listening...";
    };
    recognition.onend = function() {
      document.getElementById("micBtn").classList.remove("recording");
      document.getElementById("userInput").placeholder = "Ask about your soil...";
    };
  }

  function toggleMic() {
    if(!recognition) { alert("Voice input is not supported in this browser."); return; }
    if(document.getElementById("micBtn").classList.contains("recording")) {
      recognition.stop();
    } else {
      recognition.start();
    }
  }

  loadSensors();
  setInterval(loadSensors, 30000);
</script>
</body>
</html>
"""

@app.route("/")
def serve_index():
    return HTML_CONTENT

import os
@app.route("/logo.png")
def serve_logo():
    import mimetypes
    from flask import send_file
    # absolute path to current dir
    path = os.path.join(os.path.dirname(__file__), "..", "soilbot_logo.png")
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/png")

@app.route("/api/sensor-data")

def get_sensor_data():
    return jsonify(fetch_sensor_data())

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
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply, "sensor_data": sensor_data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


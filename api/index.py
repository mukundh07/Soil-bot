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
        lines = [f"  - {k}: {v}" for k, v in sensor_data.items() if not k.startswith("_")]
        sensor_text = f"Last updated: {sensor_data.get('_timestamp','')}\n" + "\n".join(lines)
        if "_trend" in sensor_data:
            sensor_text += f"\n\nRECENT TRENDS:\n{sensor_data['_trend']}"
        if "_history" in sensor_data and sensor_data["_history"]:
            sensor_text += f"\n\nHISTORICAL DATA POINTS:\n{sensor_data['_history']}"

    return f"""You are SoilBot, an expert AI assistant for soil health and precision agriculture.
You have access to LIVE sensor readings from an IoT soil monitoring system (ESP32 + LoRaWAN).

CURRENT LIVE SENSOR DATA:
{sensor_text}

WEATHER FORECAST (Local Area):
{weather_text}

Crop Config and Container Size:
{crop_info}

SENSOR GUIDE:
- DS18B20 Temperature: Soil temperature probe
- Watermark CB: Water tension. 0-10=saturated, 10-30=optimal, 30-60=drying, above 60=dry stress
- NPK pH: Ideal 6.0-7.0 for most crops
- Nitrogen: Ideal 140-200 mg/kg for spinach. Phosphorus: 30-60. Potassium: 150-250.

INSTRUCTIONS:
- Answer questions concisely and directly. Do not add unnecessary preamble.
- Give practical farming advice based on live data and the current crop.
- If the user has a specific crop configuration, tailor advice to that geometry.
- If sensor shows N/A, say it is temporarily unavailable.
- When predicting future values, use any historical trends provided to give specific numbers.

CRITICAL FORMATTING RULES:
1. NEVER use the asterisk or star symbol anywhere in your response. Not for bold, not for bullets.
2. NEVER use markdown formatting of any kind (no **, no ##, no __, no backticks).
3. Structure your answers using plain paragraphs with blank lines between them.
4. For lists, use hyphens (-) or numbers (1., 2., 3.).
5. Keep responses short and to the point. Maximum 6-8 lines unless a full summary is requested.
6. Respond completely in: {language}.
"""

HTML_CONTENT = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SoilBot — Smart Soil Dashboard</title>
<meta name="description" content="AI-powered precision agriculture dashboard with live IoT sensor monitoring">
<link rel="icon" type="image/png" href="/logo.png">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Inter", -apple-system, sans-serif;
    background: #080b14;
    color: #e2e8f0;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ═══════ ALERT BANNER ═══════ */
  .alert-banner {
    display: none;
    align-items: center;
    gap: 12px;
    padding: 12px 28px;
    background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border-bottom: 1px solid rgba(239,68,68,0.3);
    font-size: 13px;
    font-weight: 500;
    color: #fca5a5;
    animation: slideDown 0.4s ease;
  }
  .alert-banner.show { display: flex; }
  @keyframes slideDown { from{transform:translateY(-100%);opacity:0} to{transform:translateY(0);opacity:1} }
  .alert-banner .alert-icon { font-size: 18px; }
  .alert-banner .alert-close {
    margin-left: auto; cursor: pointer;
    background: none; border: none; color: #fca5a5; font-size: 18px; padding: 0 4px;
  }

  /* ═══════ NAVBAR ═══════ */
  .navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px;
    background: rgba(8,11,20,0.9);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    position: sticky; top: 0; z-index: 100;
  }
  .navbar-brand { display: flex; align-items: center; gap: 12px; }
  .navbar-brand img { width: 38px; height: 38px; border-radius: 10px; box-shadow: 0 0 20px rgba(74,222,128,0.3); }
  .navbar-brand h1 {
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; letter-spacing: -0.5px;
  }
  .navbar-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .nav-select {
    padding: 6px 12px; border-radius: 8px;
    background: rgba(255,255,255,0.06); color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.1);
    font-family: 'Inter', sans-serif; font-size: 13px; outline: none; cursor: pointer;
    transition: border-color 0.2s;
  }
  .nav-select:hover { border-color: rgba(74,222,128,0.4); }
  .nav-select option { color: #111; background: #1e293b; }
  .nav-btn {
    padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 600;
    cursor: pointer; border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05); color: #e2e8f0;
    font-family: 'Inter', sans-serif; transition: all 0.2s; white-space: nowrap;
  }
  .nav-btn:hover { background: rgba(74,222,128,0.12); border-color: rgba(74,222,128,0.35); color: #4ade80; }
  .nav-btn.pdf { border-color: rgba(239,68,68,0.3); }
  .nav-btn.pdf:hover { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.5); color: #fca5a5; }
  .live-badge { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; color: #4ade80; text-transform: uppercase; letter-spacing: 1px; }
  .live-dot { width: 8px; height: 8px; background: #4ade80; border-radius: 50%; animation: livePulse 2s infinite; box-shadow: 0 0 8px rgba(74,222,128,0.6); }
  @keyframes livePulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }

  /* ═══════ DASHBOARD ═══════ */
  .dashboard { max-width: 1200px; margin: 0 auto; padding: 24px 20px; display: flex; flex-direction: column; gap: 20px; }

  /* ═══════ STATUS + COUNTDOWN BAR ═══════ */
  .status-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; gap: 10px; flex-wrap: wrap;
  }
  .status-bar .system-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: rgba(255,255,255,0.3); font-weight: 600; }
  .status-bar .ts { font-size: 12px; color: rgba(255,255,255,0.4); }
  .countdown-badge {
    display: flex; align-items: center; gap: 6px; font-size: 12px;
    color: rgba(255,255,255,0.5); background: rgba(255,255,255,0.04);
    padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.07);
  }
  .countdown-badge span { color: #4ade80; font-weight: 700; font-variant-numeric: tabular-nums; }

  /* ═══════ HEALTH SCORE ═══════ */
  .health-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  .health-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 20px 24px;
    display: flex; flex-direction: column; gap: 10px; position: relative; overflow: hidden;
  }
  .health-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #4ade80, #22d3ee); border-radius: 18px 18px 0 0;
  }
  .health-score-wrap { display: flex; align-items: center; gap: 16px; }
  .health-circle {
    width: 72px; height: 72px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    background: conic-gradient(#4ade80 0%, rgba(255,255,255,0.08) 0%);
    position: relative;
  }
  .health-circle .score-num { font-size: 24px; font-weight: 800; color: #fff; line-height: 1; display: flex; align-items: baseline; }
  .health-circle .score-num span { font-size: 14px; opacity: 0.5; margin-left: 2px; }
  .health-circle .score-lbl { font-size: 8px; color: rgba(255,255,255,0.4); letter-spacing: 1px; text-transform: uppercase; }
  .health-info .score-title { font-size: 13px; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }
  .health-info .score-desc { font-size: 11px; color: rgba(255,255,255,0.4); line-height: 1.5; }

  .weather-card {
    background: linear-gradient(135deg, rgba(34,211,238,0.06), rgba(74,222,128,0.04));
    border: 1px solid rgba(34,211,238,0.15); border-radius: 18px; padding: 20px 24px;
    display: flex; flex-direction: column; gap: 6px; position: relative; overflow: hidden;
  }
  .weather-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #22d3ee, #38bdf8); border-radius: 18px 18px 0 0; }
  .weather-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; color: rgba(255,255,255,0.35); font-weight: 600; }
  .weather-main { font-size: 28px; font-weight: 800; color: #22d3ee; letter-spacing: -1px; }
  .weather-sub { font-size: 12px; color: rgba(255,255,255,0.45); line-height: 1.6; }
  .weather-rain { display: flex; gap: 14px; margin-top: 4px; }
  .weather-rain-item { font-size: 11px; color: rgba(255,255,255,0.4); }
  .weather-rain-item span { color: #38bdf8; font-weight: 600; }

  .alert-log-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 18px 20px; display: flex; flex-direction: column; gap: 10px;
    position: relative; overflow: hidden;
  }
  .alert-log-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #f59e0b, #ef4444); border-radius: 18px 18px 0 0; }
  .alert-log-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: rgba(255,255,255,0.35); font-weight: 600; }
  .alert-log-list { display: flex; flex-direction: column; gap: 6px; max-height: 90px; overflow-y: auto; }
  .alert-log-item { display: flex; gap: 8px; align-items: flex-start; font-size: 11px; color: rgba(255,255,255,0.5); }
  .alert-log-item .log-dot { width: 6px; height: 6px; border-radius: 50%; background: #f59e0b; margin-top: 3px; flex-shrink: 0; }
  .alert-log-item.log-red .log-dot { background: #ef4444; }
  .alert-log-empty { font-size: 11px; color: rgba(255,255,255,0.2); font-style: italic; }

  /* ═══════ SENSOR GRID ═══════ */
  .sensor-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; }
  .sensor-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 18px 16px; text-align: center;
    position: relative; overflow: hidden;
    transition: transform 0.25s, border-color 0.3s, box-shadow 0.3s;
  }
  .sensor-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #4ade80, #22d3ee); border-radius: 16px 16px 0 0;
    opacity: 0; transition: opacity 0.3s;
  }
  .sensor-card:hover { transform: translateY(-3px); border-color: rgba(74,222,128,0.25); box-shadow: 0 8px 30px rgba(74,222,128,0.08); }
  .sensor-card:hover::before { opacity: 1; }
  .sensor-card .card-icon { font-size: 22px; margin-bottom: 8px; }
  .sensor-card .card-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px; color: rgba(255,255,255,0.4); font-weight: 600; margin-bottom: 6px; }
  .sensor-card .card-value { font-size: 26px; font-weight: 800; color: #fff; letter-spacing: -1px; line-height: 1; }
  .sensor-card .card-unit { font-size: 11px; color: rgba(255,255,255,0.35); margin-top: 4px; font-weight: 500; }
  .sensor-card.status-good { border-color: rgba(74,222,128,0.2); }
  .sensor-card.status-good .card-value { color: #4ade80; }
  .sensor-card.status-warn { border-color: rgba(250,204,21,0.25); }
  .sensor-card.status-warn .card-value { color: #facc15; }
  .sensor-card.status-bad { border-color: rgba(239,68,68,0.25); }
  .sensor-card.status-bad .card-value { color: #ef4444; }
  .sensor-card.status-na { border-color: rgba(255,255,255,0.06); }
  .sensor-card.status-na .card-value { color: rgba(255,255,255,0.2); font-size: 16px; }

  /* ═══════ CHART PANEL ═══════ */
  .chart-panel {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px;
  }
  .chart-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
  .chart-title { font-size: 14px; font-weight: 700; color: #e2e8f0; }
  .chart-sub { font-size: 11px; color: rgba(255,255,255,0.3); margin-top: 2px; }
  .chart-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .chart-tab {
    padding: 5px 14px; border-radius: 20px; font-size: 11px; font-weight: 600;
    cursor: pointer; border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.5);
    font-family: 'Inter', sans-serif; transition: all 0.2s;
  }
  .chart-tab.active { background: rgba(74,222,128,0.15); border-color: rgba(74,222,128,0.4); color: #4ade80; }
  .chart-tab:hover:not(.active) { background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.8); }
  .chart-canvas-wrap { position: relative; height: 220px; }

  /* ═══════ CHAT PANEL ═══════ */
  .chat-panel {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; display: flex; flex-direction: column; height: 480px; overflow: hidden;
  }
  .chat-topbar { padding: 14px 20px; border-bottom: 1px solid rgba(255,255,255,0.06); display: flex; align-items: center; gap: 10px; }
  .chat-topbar .ai-dot { width: 9px; height: 9px; background: #4ade80; border-radius: 50%; animation: livePulse 2s infinite; box-shadow: 0 0 6px rgba(74,222,128,0.5); }
  .chat-topbar .ai-label { font-size: 14px; font-weight: 600; color: #e2e8f0; }
  .chat-topbar .ai-sub { font-size: 11px; color: rgba(255,255,255,0.3); margin-left: auto; margin-right: 10px; }
  .voice-toggle { background: none; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: white; cursor: pointer; font-size: 16px; padding: 5px 8px; transition: background 0.2s, border-color 0.2s; }
  .voice-toggle:hover { background: rgba(255,255,255,0.06); border-color: rgba(74,222,128,0.3); }
  .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.1) transparent; }
  .message { display: flex; gap: 10px; animation: msgIn 0.35s ease; }
  @keyframes msgIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  .message.user { flex-direction: row-reverse; }
  .avatar { width: 32px; height: 32px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; overflow: hidden; }
  .message.bot .avatar { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.15); }
  .message.user .avatar { background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.2); }
  .bubble { max-width: 78%; padding: 12px 16px; border-radius: 14px; font-size: 13.5px; line-height: 1.65; color: #e2e8f0; }
  .message.bot .bubble { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.06); border-top-left-radius: 4px; }
  .message.user .bubble { background: linear-gradient(135deg, rgba(79,70,229,0.4), rgba(124,58,237,0.4)); border: 1px solid rgba(124,58,237,0.3); border-top-right-radius: 4px; }
  .typing-dot { display: inline-block; width: 7px; height: 7px; background: #4ade80; border-radius: 50%; margin: 0 2px; animation: bounce 1.2s infinite; }
  .typing-dot:nth-child(2){animation-delay:.2s} .typing-dot:nth-child(3){animation-delay:.4s}
  @keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}
  .quick-actions { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 20px 14px; }
  .quick-btn { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 50px; padding: 7px 16px; color: rgba(255,255,255,0.6); font-size: 12px; font-family: 'Inter', sans-serif; cursor: pointer; transition: all 0.2s; }
  .quick-btn:hover { background: rgba(74,222,128,0.1); border-color: rgba(74,222,128,0.25); color: #4ade80; }
  .input-area { padding: 14px 18px; border-top: 1px solid rgba(255,255,255,0.06); display: flex; gap: 10px; align-items: center; }
  .input-area input { flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 12px 18px; color: #e2e8f0; font-size: 13.5px; font-family: "Inter", sans-serif; outline: none; transition: border-color 0.2s, box-shadow 0.2s; }
  .input-area input::placeholder { color: rgba(255,255,255,0.25); }
  .input-area input:focus { border-color: rgba(74,222,128,0.4); box-shadow: 0 0 0 3px rgba(74,222,128,0.08); }
  .mic-btn { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 8px; font-size: 20px; cursor: pointer; transition: all 0.2s; outline: none; }
  .mic-btn:hover { background: rgba(255,255,255,0.08); border-color: rgba(74,222,128,0.3); }
  .mic-btn.recording { background: rgba(239,68,68,0.15); border-color: rgba(239,68,68,0.4); animation: pulseMic 1s infinite; box-shadow: 0 0 12px rgba(239,68,68,0.3); }
  @keyframes pulseMic { 0%{transform:scale(1)} 50%{transform:scale(1.08)} 100%{transform:scale(1)} }
  .send-btn { background: linear-gradient(135deg, #4ade80, #22c55e); border: none; border-radius: 12px; width: 44px; height: 44px; cursor: pointer; font-size: 17px; display: flex; align-items: center; justify-content: center; transition: transform 0.2s, opacity 0.2s; flex-shrink: 0; box-shadow: 0 4px 16px rgba(74,222,128,0.25); }
  .send-btn:hover { transform: scale(1.06); }
  .send-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; box-shadow: none; }

  /* ═══════ RESPONSIVE ═══════ */
  @media (max-width: 900px) { .health-row { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 768px) {
    .navbar { padding: 12px 16px; } .navbar-brand h1 { font-size: 17px; }
    .dashboard { padding: 16px 12px; }
    .sensor-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .sensor-card .card-value { font-size: 22px; }
    .chat-panel { height: 440px; } .nav-select { font-size: 12px; padding: 5px 8px; }
    .health-row { grid-template-columns: 1fr; }
  }
  @media (max-width: 480px) {
    .sensor-grid { grid-template-columns: repeat(2, 1fr); }
    .navbar-controls { gap: 6px; } .live-badge span { display: none; }
    .nav-btn span { display: none; }
  }
</style>
</head>
<body>

<!-- ═══ ALERT BANNER ═══ -->
<div class="alert-banner" id="alertBanner">
  <span class="alert-icon">🚨</span>
  <span id="alertText">Soil moisture is critically low — irrigation required!</span>
  <button class="alert-close" onclick="document.getElementById('alertBanner').classList.remove('show')">✕</button>
</div>

<!-- ═══ NAVBAR ═══ -->
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
    <button class="nav-btn" onclick="downloadCSV()" title="Download 7-day CSV">📥 <span>CSV</span></button>
    <button class="nav-btn pdf" onclick="downloadPDF()" title="Download PDF Report">📄 <span>PDF</span></button>
    <div class="live-badge">
      <div class="live-dot"></div>
      <span>LIVE</span>
    </div>
  </div>
</nav>

<!-- ═══ DASHBOARD ═══ -->
<main class="dashboard">

  <!-- Status Bar -->
  <div class="status-bar">
    <span class="system-label">Soil Monitoring System</span>
    <span class="ts" id="sensorTs">Connecting to sensors...</span>
    <div class="countdown-badge">⏱ Next refresh in <span id="countdown">600</span>s</div>
  </div>

  <!-- Health Score Row -->
  <div class="health-row">

    <!-- Crop Health Score -->
    <div class="health-card" id="healthCard">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:rgba(255,255,255,0.35);font-weight:600;">Crop Health Score</div>
      <div class="health-score-wrap">
        <div class="health-circle" id="healthCircle">
          <div class="score-num" id="healthNum">--</div>
        </div>
        <div class="health-info">
          <div class="score-title" id="healthTitle">Calculating...</div>
          <div class="score-desc" id="healthDesc">Waiting for sensor data to compute soil health index.</div>
        </div>
      </div>
    </div>

    <!-- Weather Widget -->
    <div class="weather-card" id="weatherCard">
      <div class="weather-title">🌤 Local Weather — CDAC Hyderabad</div>
      <div class="weather-main" id="wTemp">--°C</div>
      <div class="weather-sub" id="wDesc">Loading weather data...</div>
      <div class="weather-rain" id="wRain"></div>
    </div>

    <!-- Alert Log -->
    <div class="alert-log-card">
      <div class="alert-log-title">⚠️ Alert History</div>
      <div class="alert-log-list" id="alertLog">
        <div class="alert-log-empty">No alerts yet — system monitoring</div>
      </div>
    </div>

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

  <!-- 7-Day History Chart -->
  <div class="chart-panel">
    <div class="chart-header">
      <div>
        <div class="chart-title">📈 7-Day Sensor History</div>
        <div class="chart-sub">Live data from ThingSpeak · Last 7 days</div>
      </div>
      <div class="chart-tabs" id="chartTabs">
        <button class="chart-tab active" onclick="switchChart('temp',this)">🌡️ Temp</button>
        <button class="chart-tab" onclick="switchChart('cb',this)">💧 CB</button>
        <button class="chart-tab" onclick="switchChart('moisture',this)">💦 Moisture</button>
        <button class="chart-tab" onclick="switchChart('ph',this)">⚗️ pH</button>
        <button class="chart-tab" onclick="switchChart('npk',this)">🌿 NPK</button>
      </div>
    </div>
    <div class="chart-canvas-wrap">
      <canvas id="historyChart"></canvas>
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
        <div class="bubble">Welcome to SoilBot 🌱 I have access to your live sensor data and 7-day history. Ask me anything about soil health, irrigation timing, fertilizer recommendations or future predictions.</div>
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
  let chatHistory = [];
  let sensorData = {};
  let historyFeeds = [];
  let alertLog = [];
  let chart = null;
  let countdownVal = 600;
  const langCodes = { "English": "en-US", "Hindi": "hi-IN", "Telugu": "te-IN" };

  // ═══ Countdown Timer ═══
  setInterval(() => {
    countdownVal--;
    if (countdownVal <= 0) countdownVal = 600;
    document.getElementById("countdown").textContent = countdownVal;
  }, 1000);

  function updateVoiceLang() {
    const selectedLang = document.getElementById("langSelect").value;
    if (recognition) recognition.lang = langCodes[selectedLang] || "en-US";
  }

  // ═══ Sensor Status ═══
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

  // ═══ Health Score Calculation ═══
  function computeHealthScore(d) {
    let score = 0; let count = 0;
    const cb = parseFloat(d["Watermark CB Value"]);
    const ph = parseFloat(d["NPK pH"]);
    const moist = parseFloat(d["NPK Moisture"]);
    const temp = parseFloat(d["DS18B20 Temperature"]);
    const n = parseFloat(d["Nitrogen (N)"]);
    const p = parseFloat(d["Phosphorus (P)"]);
    const k = parseFloat(d["Potassium (K)"]);

    if (!isNaN(cb)) { score += cb <= 30 ? 100 : cb <= 60 ? 70 : cb <= 100 ? 40 : 10; count++; }
    if (!isNaN(ph)) { score += (ph >= 6.0 && ph <= 7.0) ? 100 : (ph >= 5.5 && ph <= 7.5) ? 70 : 30; count++; }
    if (!isNaN(moist)) { score += moist >= 40 ? 100 : moist >= 25 ? 65 : 20; count++; }
    if (!isNaN(temp)) { score += (temp >= 18 && temp <= 30) ? 100 : (temp >= 10 && temp <= 38) ? 70 : 30; count++; }
    if (!isNaN(n) && n > 0) { score += n >= 140 ? 100 : n >= 80 ? 65 : 30; count++; }
    if (!isNaN(p) && p > 0) { score += p >= 30 ? 100 : p >= 15 ? 65 : 30; count++; }
    if (!isNaN(k) && k > 0) { score += k >= 150 ? 100 : k >= 80 ? 65 : 30; count++; }

    return count > 0 ? Math.round(score / count) : null;
  }

  function updateHealthCard(d) {
    const score = computeHealthScore(d);
    if (score === null) return;
    const num = document.getElementById("healthNum");
    const circle = document.getElementById("healthCircle");
    const title = document.getElementById("healthTitle");
    const desc = document.getElementById("healthDesc");

    num.textContent = score;
    let color, label, description;
    if (score >= 80) { color = "#4ade80"; label = "Excellent 🌟"; description = "Your soil is in great condition. Maintain current practices."; }
    else if (score >= 60) { color = "#22d3ee"; label = "Good 👍"; description = "Soil health is acceptable. Minor adjustments may help."; }
    else if (score >= 40) { color = "#facc15"; label = "Fair ⚠️"; description = "Some parameters need attention. Check alerts below."; }
    else { color = "#ef4444"; label = "Poor 🚨"; description = "Soil health is critically low. Immediate action required!"; }

    const pct = score / 100;
    circle.style.background = `conic-gradient(${color} ${pct * 360}deg, rgba(255,255,255,0.06) 0deg)`;
    num.innerHTML = `${score}<span>/100</span>`;
    num.style.color = color;
    title.textContent = label;
    desc.textContent = description;

    // Check alerts
    const cb = parseFloat(d["Watermark CB Value"]);
    const phVal = parseFloat(d["NPK pH"]);
    if (!isNaN(cb) && cb > 60) triggerAlert(`Watermark CB is ${cb} — soil is drying out, irrigation recommended!`, "red");
    if (!isNaN(phVal) && (phVal < 5.5 || phVal > 7.5)) triggerAlert(`pH is ${phVal} — outside optimal range (6.0–7.0)!`, "red");
  }

  function triggerAlert(msg, type="warn") {
    const banner = document.getElementById("alertBanner");
    document.getElementById("alertText").textContent = msg;
    banner.classList.add("show");

    const now = new Date();
    const timeStr = new Intl.DateTimeFormat("en-IN",{timeZone:"Asia/Kolkata",hour:"2-digit",minute:"2-digit",hour12:true}).format(now);
    if (!alertLog.find(a => a.msg === msg)) {
      alertLog.unshift({ msg, type, time: timeStr });
      if (alertLog.length > 10) alertLog.pop();
      renderAlertLog();
    }
  }

  function renderAlertLog() {
    const el = document.getElementById("alertLog");
    if (alertLog.length === 0) { el.innerHTML = '<div class="alert-log-empty">No alerts yet — system monitoring</div>'; return; }
    el.innerHTML = alertLog.map(a =>
      `<div class="alert-log-item ${a.type === 'red' ? 'log-red' : ''}">
        <div class="log-dot"></div>
        <span><b>${a.time}</b> — ${a.msg}</span>
      </div>`
    ).join("");
  }

  // ═══ Weather ═══
  async function loadWeather() {
    try {
      const r = await fetch("https://api.open-meteo.com/v1/forecast?latitude=17.188&longitude=78.468&current=temperature_2m,precipitation,weathercode&daily=precipitation_probability_max,temperature_2m_max,temperature_2m_min&timezone=Asia%2FKolkata&forecast_days=3");
      const d = await r.json();
      document.getElementById("wTemp").textContent = `${d.current.temperature_2m}°C`;
      const codes = {0:"☀️ Clear",1:"🌤 Mostly Clear",2:"⛅ Partly Cloudy",3:"☁️ Overcast",45:"🌫 Foggy",51:"🌦 Drizzle",61:"🌧 Rain",80:"🌦 Showers",95:"⛈ Thunderstorm"};
      const code = d.current.weathercode;
      const desc = codes[code] || codes[Math.floor(code/10)*10] || "Variable";
      document.getElementById("wDesc").textContent = `${desc} · Rain now: ${d.current.precipitation}mm`;
      document.getElementById("wRain").innerHTML =
        `<div class="weather-rain-item">Today: <span>${d.daily.precipitation_probability_max[0]}%</span> rain</div>
         <div class="weather-rain-item">Tomorrow: <span>${d.daily.precipitation_probability_max[1]}%</span> rain</div>
         <div class="weather-rain-item">Day 3: <span>${d.daily.precipitation_probability_max[2]}%</span> rain</div>`;
    } catch(e) {
      document.getElementById("wDesc").textContent = "Weather data unavailable.";
    }
  }

  // ═══ Sensor Load ═══
  async function loadSensors() {
    countdownVal = 30;
    try {
      const r = await fetch(BASE + "/api/sensor-data");
      const d = await r.json();
      const grid = document.getElementById("sensorGrid");
      const ts = document.getElementById("sensorTs");
      if (d.error) { ts.textContent = "Sensor error: " + d.error; return; }
      sensorData = d;

      // Timestamp IST
      const raw = d._timestamp || "unknown";
      if (raw !== "unknown") {
        const dt = new Date(raw);
        const diff = Math.floor((Date.now() - dt.getTime()) / 60000);
        const istTime = new Intl.DateTimeFormat("en-IN",{timeZone:"Asia/Kolkata",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:true}).format(dt);
        const istDate = new Intl.DateTimeFormat("en-IN",{timeZone:"Asia/Kolkata",day:"2-digit",month:"short",year:"numeric"}).format(dt);
        const agoText = diff < 1 ? "just now" : diff < 60 ? `${diff} min ago` : `${Math.floor(diff/60)}h ago`;
        ts.textContent = `Updated ${agoText} · ${istDate}, ${istTime} IST`;
      }

      grid.innerHTML = "";
      for (const [k, v] of Object.entries(d)) {
        if (k.startsWith("_")) continue;
        const vStr = String(v);
        const isNA = vStr === "N/A";
        const numVal = vStr.replace(/[^\d.-]/g, '');
        const unit = vStr.replace(/[\d.-]/g, '').trim();
        const status = isNA ? "na" : getSensorStatus(k, numVal);
        grid.innerHTML += `<div class="sensor-card status-${status}">
          <div class="card-icon">${sensorIcons[k]||"📈"}</div>
          <div class="card-label">${k.replace("DS18B20 ","").replace("Watermark ","WM ").replace("NPK ","")}</div>
          <div class="card-value">${isNA ? "N/A" : numVal}</div>
          <div class="card-unit">${isNA ? "offline" : unit}</div>
        </div>`;
      }
      updateHealthCard(d);
    } catch(e) {
      document.getElementById("sensorTs").textContent = "Could not fetch sensors.";
    }
  }

  // ═══ Chart ═══
  async function loadHistory() {
    try {
      const r = await fetch(BASE + "/api/history");
      const d = await r.json();
      if (d.feeds) { historyFeeds = d.feeds; renderChart("temp"); }
    } catch(e) { console.warn("History unavailable", e); }
  }

  const chartConfigs = {
    temp:    { label: "Soil Temp (°C)", field: "field1", color: "#f97316", scale: 1 },
    cb:      { label: "Watermark CB", field: "field2", color: "#22d3ee", scale: 1 },
    moisture:{ label: "Moisture (%)", field: "field3", color: "#4ade80", scale: 1 },
    ph:      { label: "NPK pH", field: "field4", color: "#a78bfa", scale: 1 },
    npk:     { label: ["N", "P", "K"], field: ["field5","field6","field7"], color: ["#4ade80","#f59e0b","#38bdf8"], scale: 1 }
  };

  function toIST(utcStr) {
    const dt = new Date(utcStr);
    return new Intl.DateTimeFormat("en-IN",{timeZone:"Asia/Kolkata",month:"short",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false}).format(dt);
  }

  function renderChart(type) {
    const cfg = chartConfigs[type];
    const labels = historyFeeds.map(f => toIST(f.created_at));

    let datasets;
    if (type === "npk") {
      datasets = cfg.field.map((f, i) => ({
        label: cfg.label[i],
        data: historyFeeds.map(feed => parseFloat(feed[f]) || null),
        borderColor: cfg.color[i], backgroundColor: cfg.color[i] + "22",
        tension: 0.4, pointRadius: 2, fill: false, borderWidth: 2
      }));
    } else {
      datasets = [{
        label: cfg.label,
        data: historyFeeds.map(f => parseFloat(f[cfg.field]) || null),
        borderColor: cfg.color, backgroundColor: cfg.color + "18",
        tension: 0.4, pointRadius: 2, fill: true, borderWidth: 2
      }];
    }

    if (chart) chart.destroy();
    const ctx = document.getElementById("historyChart").getContext("2d");
    chart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: "rgba(255,255,255,0.6)", font: { family: "Inter", size: 11 }, boxWidth: 12, padding: 16 } },
          tooltip: {
            backgroundColor: "rgba(8,11,20,0.95)", borderColor: "rgba(255,255,255,0.1)", borderWidth: 1,
            titleColor: "rgba(255,255,255,0.5)", bodyColor: "#e2e8f0",
            padding: 12, titleFont: { family: "Inter" }, bodyFont: { family: "Inter" }
          }
        },
        scales: {
          x: { ticks: { color: "rgba(255,255,255,0.3)", font: { family: "Inter", size: 10 }, maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
          y: { ticks: { color: "rgba(255,255,255,0.3)", font: { family: "Inter", size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } }
        }
      }
    });
  }

  function switchChart(type, btn) {
    document.querySelectorAll(".chart-tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    if (historyFeeds.length > 0) renderChart(type);
  }

  // ═══ CSV Download ═══
  async function downloadCSV() {
    try {
      const r = await fetch(BASE + "/api/history");
      const d = await r.json();
      if (!d.feeds || d.feeds.length === 0) { alert("No history data available yet."); return; }

      const headers = ["Timestamp (IST)","Temp (C)","WM CB","NPK Moisture (%)","NPK pH","Nitrogen (mg/kg)","Phosphorus (mg/kg)","Potassium (mg/kg)","WM Moisture (%)"];
      const rows = d.feeds.map(f => [
        toIST(f.created_at),
        f.field1||"", f.field2||"", f.field3||"", f.field4||"",
        f.field5||"", f.field6||"", f.field7||"", f.field8||""
      ]);

      const csv = [headers, ...rows].map(r => r.join(",")).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `soilbot_7day_${new Date().toISOString().slice(0,10)}.csv`;
      a.click(); URL.revokeObjectURL(url);
    } catch(e) { alert("Could not download CSV: " + e.message); }
  }

  // ═══ PDF Report ═══
  function downloadPDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const now = new Intl.DateTimeFormat("en-IN",{timeZone:"Asia/Kolkata",dateStyle:"full",timeStyle:"short"}).format(new Date());
    const score = computeHealthScore(sensorData);

    doc.setFillColor(8, 11, 20);
    doc.rect(0, 0, 210, 297, "F");

    doc.setTextColor(74, 222, 128);
    doc.setFontSize(22); doc.setFont("helvetica","bold");
    doc.text("SoilBot — Soil Health Report", 20, 25);

    doc.setTextColor(180, 200, 220);
    doc.setFontSize(10); doc.setFont("helvetica","normal");
    doc.text(`Generated: ${now} IST`, 20, 34);
    doc.text(`Crop: ${document.getElementById("cropSelect").value}`, 20, 41);

    doc.setDrawColor(74, 222, 128); doc.setLineWidth(0.5);
    doc.line(20, 46, 190, 46);

    doc.setTextColor(74, 222, 128); doc.setFontSize(13); doc.setFont("helvetica","bold");
    doc.text(`Overall Health Score: ${score !== null ? score + "/100" : "N/A"}`, 20, 56);

    doc.setTextColor(200, 220, 240); doc.setFontSize(11); doc.setFont("helvetica","bold");
    doc.text("Current Sensor Readings:", 20, 68);

    doc.setFont("helvetica","normal"); doc.setFontSize(10);
    let y = 76;
    const fields = [
      ["DS18B20 Temperature", sensorData["DS18B20 Temperature"]],
      ["Watermark CB Value",  sensorData["Watermark CB Value"]],
      ["Watermark Moisture",  sensorData["Watermark Moisture"]],
      ["NPK Moisture",        sensorData["NPK Moisture"]],
      ["NPK pH",              sensorData["NPK pH"]],
      ["Nitrogen (N)",        sensorData["Nitrogen (N)"]],
      ["Phosphorus (P)",      sensorData["Phosphorus (P)"]],
      ["Potassium (K)",       sensorData["Potassium (K)"]],
    ];
    fields.forEach(([k, v]) => {
      doc.setTextColor(150, 180, 200); doc.text(k + ":", 20, y);
      doc.setTextColor(230, 240, 255); doc.text(v || "N/A", 100, y);
      y += 8;
    });

    if (alertLog.length > 0) {
      doc.setDrawColor(239, 68, 68); doc.line(20, y + 4, 190, y + 4); y += 12;
      doc.setTextColor(239, 68, 68); doc.setFontSize(11); doc.setFont("helvetica","bold");
      doc.text("Alert History:", 20, y); y += 8;
      doc.setFont("helvetica","normal"); doc.setFontSize(10); doc.setTextColor(240, 180, 180);
      alertLog.slice(0, 6).forEach(a => { doc.text(`• [${a.time}] ${a.msg}`, 20, y, { maxWidth: 170 }); y += 8; });
    }

    doc.setDrawColor(74, 222, 128); doc.line(20, 280, 190, 280);
    doc.setTextColor(100, 130, 100); doc.setFontSize(8);
    doc.text("SoilBot — ESP32 LoRaWAN Precision Agriculture System · CDAC Hyderabad", 20, 286);

    doc.save(`soilbot_report_${new Date().toISOString().slice(0,10)}.pdf`);
  }

  // ═══ Chat ═══
  function cleanMarkdown(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, '$1')
      .replace(/\*(.+?)\*/g, '$1')
      .replace(/^#{1,6}\s*/gm, '')
      .replace(/^[*-]\s+/gm, '- ')
      .replace(/\*/g, '')
      .replace(/_{2}/g, '')
      .replace(/`/g, '')
      .trim();
  }

  function addMessage(role, text) {
    const msgs = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message " + role;
    const displayText = role === "bot" ? cleanMarkdown(text) : text;
    div.innerHTML = `<div class="avatar">${role==="user"?"👤":"<img src='/logo.png' style='width:100%; height:100%; border-radius:10px; object-fit:cover;'>"}</div><div class="bubble">${displayText.replace(/\n/g,"<br>")}</div>`;
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
    const btn = document.getElementById("sendBtn");
    const msg = input.value.trim();
    if (!msg) return;

    const cropInfo = document.getElementById("cropSelect").value;
    const langInfo = document.getElementById("langSelect").value;

    input.value = ""; btn.disabled = true;
    document.getElementById("suggestions").style.display = "none";
    addMessage("user", msg);
    chatHistory.push({role:"user", text:msg});
    showTyping();
    try {
      const r = await fetch(BASE + "/api/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message: msg, history: chatHistory, crop_info: cropInfo, language: langInfo})
      });
      const d = await r.json();
      removeTyping();
      if (d.error) { addMessage("bot", "Error: " + d.error); }
      else {
        addMessage("bot", d.reply);
        chatHistory.push({role:"bot", text: d.reply});
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
      document.getElementById("userInput").value = event.results[0][0].transcript;
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
    if(document.getElementById("micBtn").classList.contains("recording")) { recognition.stop(); }
    else { recognition.start(); }
  }

  // ═══ Init ═══
  loadSensors();
  loadHistory();
  loadWeather();
  setInterval(loadSensors, 600000);
  setInterval(loadHistory, 120000);
  setInterval(loadWeather, 300000);
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
    response = jsonify(fetch_sensor_data())
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response

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

@app.route("/api/history")
def get_sensor_history():
    try:
        import requests as req
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CH_ID}/feeds.json?results=2000"
        r = req.get(url, timeout=10)
        data = r.json()
        feeds = data.get("feeds", [])
        filtered = []
        last_ts = None
        from datetime import datetime
        for feed in feeds:
            ts_str = feed.get("created_at")
            if not ts_str:
                continue
            ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
            if last_ts is None or (ts - last_ts).total_seconds() >= 1800:
                filtered.append(feed)
                last_ts = ts
        return jsonify({"feeds": filtered})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

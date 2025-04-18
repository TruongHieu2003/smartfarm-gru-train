
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ GRU Auto-Training is running on Render (Web Service Plan)."

def send_onesignal_notification():
    try:
        onesignal_app_id = os.environ.get("5f24656f-1db1-4b72-8739-f28ed1c77979")
        onesignal_api_key = os.environ.get("yq5u4ict7umxnywdqx552vp6q")

        if not onesignal_app_id or not onesignal_api_key:
            print("⚠️ Thiếu thông tin OneSignal.")
            return

        headers = {
            "Authorization": f"Basic {onesignal_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "app_id": onesignal_app_id,
            "included_segments": ["All"],
            "headings": {"en": "SmartFarm - Dự báo thời tiết"},
            "contents": {"en": "✅ Dự báo thời tiết mới đã được cập nhật."}
        }

        response = requests.post("https://onesignal.com/api/v1/notifications", headers=headers, json=payload)
        if response.status_code == 200:
            print("📲 Đã gửi thông báo OneSignal")
        else:
            print("❌ Gửi OneSignal thất bại:", response.text)
    except Exception as e:
        print("❌ Lỗi gửi OneSignal:", str(e))

def run_training_and_forecast():
    print("🔁 Bắt đầu kiểm tra và huấn luyện...")

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    google_key = os.environ.get("C:/Users/hieuv/smartfarm-train/focal-grin-455408-m0-c0013e6015d9.json")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(google_key), scope)
    client = gspread.authorize(creds)

    sheet_url = "https://docs.google.com/spreadsheets/d/19qBwHPrIes6PeGAyIzMORPVB-7utQpaZG7RHrdRfoNI"
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.worksheet("DATA")
    data = pd.DataFrame(worksheet.get_all_records())

    data['timestamp'] = pd.to_datetime(data['NGÀY'] + ' ' + data['GIỜ'], format='%d/%m/%Y %H:%M:%S')
    data = data.sort_values('timestamp')
    data.rename(columns={
        'temperature': 'temp', 'humidity': 'humid', 'soil_moisture': 'soil', 'wind': 'wind', 'rain': 'rain'
    }, inplace=True)

    saved_timestamp = None
    if os.path.exists("last_timestamp.json"):
        with open("last_timestamp.json", "r") as f:
            saved_timestamp = pd.to_datetime(json.load(f)["last_timestamp"])
    latest_timestamp = data['timestamp'].iloc[-1]

    features = ['temp', 'humid', 'soil', 'wind', 'rain']
    dataset = data[features].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    model_path = "gru_weather_model.h5"
    window_size = 6

    if not os.path.exists(model_path):
        print("⚠️ Chưa có mô hình .h5, tạo mới từ đầu.")
        model = Sequential([
            Input(shape=(window_size, len(features))),
            GRU(units=64),
            Dense(5)
        ])
        model.compile(optimizer='adam', loss=MeanSquaredError())
        model.save(model_path)
    else:
        print("✅ Đã có mô hình .h5, tiến hành load...")
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss=MeanSquaredError())

    n_steps = 25
    forecast = []
    current_seq = scaled_data[-window_size:].copy()

    for _ in range(n_steps):
        x_input = current_seq.reshape(1, window_size, len(features))
        y_pred = model.predict(x_input, verbose=0)
        forecast.append(y_pred[0])
        current_seq = np.vstack([current_seq[1:], y_pred])

    forecast_original = scaler.inverse_transform(np.array(forecast))
    forecast_df = pd.DataFrame(forecast_original, columns=features).clip(lower=0).round(2)
    base_time_tomorrow = (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0)
    forecast_df.insert(0, "time", [(base_time_tomorrow + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M") for i in range(n_steps)])
    forecast_df.to_json("latest_prediction.json", orient="records", indent=2)
    print("📤 Đã lưu latest_prediction.json")

    if not firebase_admin._apps:
        firebase_key = os.environ.get("C:/Users/hieuv/smartfarm-train/smart-farm-6e42d-firebase-adminsdk-fbsvc-9f6b7c2379.json")
        cred = credentials.Certificate(json.loads(firebase_key))
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://smart-farm-6e42d-default-rtdb.firebaseio.com/'
        })
    ref = db.reference("forecast/tomorrow")
    ref.set(forecast_df.to_dict(orient="records"))
    print("🔥 Đã đẩy dữ liệu lên Firebase")

    send_onesignal_notification()

    if saved_timestamp is not None and latest_timestamp <= saved_timestamp:
        print("🟡 Không có dữ liệu mới.")
        return

    print("🟢 Có dữ liệu mới. Đang huấn luyện...")
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])
    X, y = np.array(X), np.array(y)

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
    model.save(model_path)

    with open("last_timestamp.json", "w") as f:
        json.dump({"last_timestamp": str(latest_timestamp)}, f)
    print("✅ Đã huấn luyện xong.")

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_training_and_forecast, 'interval', minutes=10)
    scheduler.start()
    print("🌀 Đang chạy script tự động mỗi 10 phút...")
    run_training_and_forecast()
    app.run(host="0.0.0.0", port=8080)

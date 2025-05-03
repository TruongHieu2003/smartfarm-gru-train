import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
import gspread
from google.oauth2.service_account import Credentials
import firebase_admin
from firebase_admin import credentials, db
import smtplib
from email.mime.text import MIMEText
from pytz import timezone

def send_email_notification(message):
    try:
        smtp_user = os.environ.get("EMAIL_USER")
        smtp_pass = os.environ.get("EMAIL_PASSWORD")
        receivers = os.environ.get("Email_Receiver")

        if not smtp_user or not smtp_pass or not receivers:
            print("⚠️ Thiếu thông tin SMTP.")
            return

        receiver_list = [email.strip() for email in receivers.split(",")]

        subject = "SmartFarm - Trạng thái cập nhật dữ liệu"
        body = f"{message}\n\nThời gian: {datetime.now(timezone('Asia/Ho_Chi_Minh')).strftime('%H:%M %d/%m/%Y')}"

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = ", ".join(receiver_list)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, receiver_list, msg.as_string())
        server.quit()
        print("📧 Đã gửi email thông báo")
    except Exception as e:
        print("❌ Lỗi gửi mail:", e)

# (Đầu code không đổi, giữ nguyên các import và định nghĩa hàm send_email_notification)

def run_training_and_forecast():
    print("\n🔁 Bắt đầu kiểm tra và huấn luyện...\n")

    # 🧪 Kiểm tra biến môi trường
    json_str = os.environ.get("GOOGLE_SERVICE_JSON")
    sheet_url = os.environ.get("SHEET_URL")
    print("🔍 Kiểm tra biến môi trường:")
    print("GOOGLE_SERVICE_JSON tồn tại:", bool(json_str))
    print("SHEET_URL:", sheet_url)

    if not json_str or not sheet_url:
        print("❌ Thiếu GOOGLE_SERVICE_JSON hoặc SHEET_URL")
        return

    # 🧪 Kiểm tra JSON hợp lệ
    try:
        service_account_info = json.loads(json_str)
        print("✅ GOOGLE_SERVICE_JSON hợp lệ")
    except Exception as e:
        print("❌ Lỗi parse GOOGLE_SERVICE_JSON:", e)
        return

    try:
        creds = Credentials.from_service_account_info(service_account_info)
        client = gspread.authorize(creds)

        sheet = client.open_by_url(sheet_url.strip())
        worksheet = sheet.worksheet("DATA")
        data = pd.DataFrame(worksheet.get_all_records())
        print("✅ Lấy dữ liệu từ Google Sheets thành công")
        print(data.head(3))
    except Exception as e:
        print("❌ Lỗi truy cập Google Sheets:", e)
        return

    data.columns = data.columns.str.strip()

    required_columns = ["NGÀY", "GIỜ", "temperature", "humidity", "soil_moisture", "wind", "rain"]
    if not all(col in data.columns for col in required_columns):
        print("❌ Dữ liệu thiếu cột cần thiết:", set(required_columns) - set(data.columns))
        return

    try:
        data['timestamp'] = pd.to_datetime(data['NGÀY'] + ' ' + data['GIỜ'], format='%d/%m/%Y %H:%M:%S')
        print("✅ Đã tạo cột timestamp")
    except Exception as e:
        print("❌ Lỗi chuyển đổi thời gian:", e)
        return

    data = data.sort_values('timestamp')
    data.rename(columns={
        'temperature': 'temp', 'humidity': 'humid', 'soil_moisture': 'soil',
        'wind': 'wind', 'rain': 'rain'
    }, inplace=True)

    if os.path.exists("training_data.csv"):
        try:
            old_data = pd.read_csv("training_data.csv", parse_dates=["timestamp"])
            data = pd.concat([old_data, data])
            print(f"🔁 Đã nối dữ liệu cũ: {len(old_data)} dòng + dữ liệu mới: {len(data) - len(old_data)} dòng")
        except Exception as e:
            print("⚠️ Lỗi đọc training_data.csv:", e)

    data.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    print("✅ Dữ liệu tổng sau xử lý:", data.shape)

    saved_timestamp = None
    if os.path.exists("last_timestamp.json"):
        try:
            with open("last_timestamp.json", "r") as f:
                saved_timestamp = pd.to_datetime(json.load(f)["last_timestamp"])
                print("🧪 Lần cuối huấn luyện lúc:", saved_timestamp)
        except Exception as e:
            print("⚠️ Lỗi đọc last_timestamp.json:", e)

    latest_timestamp = data["timestamp"].iloc[-1]
    print("🆕 Timestamp mới nhất trong dữ liệu:", latest_timestamp)

    if saved_timestamp is not None and latest_timestamp <= saved_timestamp:
        print("🟡 KHÔNG có dữ liệu mới.")
        send_email_notification("🟡 KHÔNG có dữ liệu mới để huấn luyện.")
        return

    features = ['temp', 'humid', 'soil', 'wind', 'rain']
    if data[features].isnull().any().any():
        print("⚠️ Có giá trị thiếu trong dữ liệu. Loại bỏ dòng lỗi.")
        data.dropna(subset=features, inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    print("✅ Đã scale dữ liệu:", scaled_data.shape)

    model_path = "gru_weather_model.h5"
    window_size = 6

    if not os.path.exists(model_path):
        print("📦 Chưa có model, tạo mới GRU...")
        model = Sequential([
            Input(shape=(window_size, len(features))),
            GRU(units=64),
            Dense(len(features))
        ])
        model.compile(optimizer="adam", loss=MeanSquaredError())
    else:
        print("📦 Đã có model, load lại từ file...")
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss=MeanSquaredError())

    print("🟢 Có dữ liệu mới. Đang huấn luyện...")
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])

    X, y = np.array(X), np.array(y)
    print(f"📊 Dữ liệu huấn luyện: X={X.shape}, y={y.shape}")
    model.fit(X, y, epochs=100, batch_size=16,
              callbacks=[EarlyStopping(monitor="loss", patience=5)], verbose=0)
    model.save(model_path)
    print("✅ Huấn luyện xong, model đã lưu")

    data.to_csv("training_data.csv", index=False)
    with open("last_timestamp.json", "w") as f:
        json.dump({"last_timestamp": str(latest_timestamp)}, f)

    # 🔮 Dự báo 24 giờ tiếp theo
    current_seq = scaled_data[-window_size:].copy()
    forecast = []
    for _ in range(24):
        x_input = current_seq.reshape(1, window_size, len(features))
        y_pred = model.predict(x_input, verbose=0)
        forecast.append(y_pred[0])
        current_seq = np.vstack([current_seq[1:], y_pred])

    forecast_original = scaler.inverse_transform(np.array(forecast))
    forecast_df = pd.DataFrame(forecast_original, columns=features).clip(lower=0).round(2)
    print("🔮 Dự báo xong, kích thước:", forecast_df.shape)

    # Tạo mốc thời gian
    base_time = datetime.now(timezone("Asia/Ho_Chi_Minh")) + timedelta(days=1)
    base_time = base_time.replace(hour=0, minute=0)
    forecast_df.insert(0, "time", [(base_time + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M") for i in range(24)])

    # Lưu local
    forecast_df.to_json("latest_prediction.json", orient="records", indent=2)
    print("📁 Đã lưu dự báo vào latest_prediction.json")

    try:
        if not firebase_admin._apps:
            firebase_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
            print("🧪 Kiểm tra FIREBASE_SERVICE_ACCOUNT_JSON:", bool(firebase_key))
            if not firebase_key:
                raise Exception("❌ FIREBASE_SERVICE_ACCOUNT_JSON is missing")
            firebase_key = json.loads(firebase_key)
            cred = credentials.Certificate(firebase_key)
            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://smart-farm-6e42d-default-rtdb.firebaseio.com/"
            })
        db.reference("forecast/tomorrow").set(forecast_df.to_dict(orient="records"))
        print("📡 Đã cập nhật Firebase.")
    except Exception as e:
        print("❌ Lỗi cập nhật Firebase:", e)

    send_email_notification("🟢 Dự báo mới đã được huấn luyện và cập nhật.")
    print("✅ XONG lúc", datetime.now(timezone("Asia/Ho_Chi_Minh")).strftime("%H:%M:%S %d/%m/%Y"))

if __name__ == "__main__":
    while True:
        run_training_and_forecast()
        time.sleep(300)


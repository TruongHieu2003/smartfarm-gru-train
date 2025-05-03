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

def run_training_and_forecast():
    print("\n🔁 Bắt đầu kiểm tra và huấn luyện...\n")

    # 🧪 Kiểm tra biến môi trường cho Google Sheets
    raw_json = os.environ.get("GOOGLE_SERVICE_JSON")
    if raw_json:
        # convert literal '\n' into actual newline
        raw_json = raw_json.replace('\\n', '\n')
    sheet_url = os.environ.get("SHEET_URL", "").strip()

    print("🔍 Kiểm tra biến môi trường:")
    print("GOOGLE_SERVICE_JSON tồn tại:", bool(raw_json))
    print("SHEET_URL:", sheet_url)

    if not raw_json or not sheet_url:
        print("❌ Thiếu GOOGLE_SERVICE_JSON hoặc SHEET_URL")
        return

    # parse JSON và authorize gspread
    try:
        service_account_info = json.loads(raw_json)
        creds = Credentials.from_service_account_info(service_account_info)
        client = gspread.authorize(creds)
        print("✅ GOOGLE_SERVICE_JSON hợp lệ và authorized")
    except Exception as e:
        print("❌ Lỗi xử lý GOOGLE_SERVICE_JSON hoặc authorize:", e)
        return

    # Lấy dữ liệu từ Sheets
    try:
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.worksheet("DATA")
        data = pd.DataFrame(worksheet.get_all_records())
        print("✅ Lấy dữ liệu từ Google Sheets thành công")
    except Exception as e:
        print("❌ Lỗi truy cập Google Sheets:", e)
        return

    # Tiền xử lý dữ liệu...
    data.columns = data.columns.str.strip()
    required_columns = ["NGÀY", "GIỜ", "temperature", "humidity", "soil_moisture", "wind", "rain"]
    if not all(col in data.columns for col in required_columns):
        missing = set(required_columns) - set(data.columns)
        print("❌ Dữ liệu thiếu cột cần thiết:", missing)
        return

    try:
        data['timestamp'] = pd.to_datetime(
            data['NGÀY'] + ' ' + data['GIỜ'],
            format='%d/%m/%Y %H:%M:%S'
        )
        print("✅ Đã tạo cột timestamp")
    except Exception as e:
        print("❌ Lỗi chuyển đổi thời gian:", e)
        return

    data = data.sort_values('timestamp').reset_index(drop=True)
    data.rename(columns={
        'temperature': 'temp', 'humidity': 'humid',
        'soil_moisture': 'soil', 'wind': 'wind', 'rain': 'rain'
    }, inplace=True)

    # Nối với data cũ nếu có
    if os.path.exists("training_data.csv"):
        try:
            old_data = pd.read_csv("training_data.csv", parse_dates=["timestamp"])
            data = pd.concat([old_data, data], ignore_index=True)
            print(f"🔁 Nối dữ liệu cũ: {len(old_data)} + mới: {len(data)-len(old_data)}")
        except Exception as e:
            print("⚠️ Lỗi đọc training_data.csv:", e)

    data.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    print("✅ Dữ liệu tổng sau xử lý:", data.shape)

    # Lấy last_timestamp
    saved_ts = None
    if os.path.exists("last_timestamp.json"):
        try:
            with open("last_timestamp.json", "r") as f:
                saved_ts = pd.to_datetime(json.load(f)["last_timestamp"])
                print("🧪 Lần cuối huấn luyện lúc:", saved_ts)
        except Exception as e:
            print("⚠️ Lỗi đọc last_timestamp.json:", e)

    latest_ts = data["timestamp"].iloc[-1]
    print("🆕 Timestamp mới nhất:", latest_ts)
    if saved_ts and latest_ts <= saved_ts:
        print("🟡 KHÔNG có dữ liệu mới.")
        send_email_notification("🟡 KHÔNG có dữ liệu mới để huấn luyện.")
        return

    # Chuẩn bị dữ liệu huấn luyện
    features = ['temp', 'humid', 'soil', 'wind', 'rain']
    data.dropna(subset=features, inplace=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[features])
    print("✅ Scale dữ liệu:", scaled.shape)

    # Xây dựng hoặc load model
    model_path = "gru_weather_model.h5"
    window_size = 6
    if not os.path.exists(model_path):
        print("📦 Tạo mới GRU model...")
        model = Sequential([
            Input(shape=(window_size, len(features))),
            GRU(64),
            Dense(len(features))
        ])
        model.compile(optimizer="adam", loss=MeanSquaredError())
    else:
        print("📦 Load model hiện có...")
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss=MeanSquaredError())

    # Tạo dữ liệu X,y
    X, y = [], []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i+window_size])
        y.append(scaled[i+window_size])
    X, y = np.array(X), np.array(y)
    print(f"📊 Dữ liệu huấn luyện: X={X.shape}, y={y.shape}")

    # Huấn luyện
    print("🟢 Huấn luyện model...")
    model.fit(
        X, y,
        epochs=100, batch_size=16,
        callbacks=[EarlyStopping(monitor="loss", patience=5)],
        verbose=0
    )
    model.save(model_path)
    print("✅ Lưu model xong")

    # Cập nhật training_data & last_timestamp
    data.to_csv("training_data.csv", index=False)
    with open("last_timestamp.json", "w") as f:
        json.dump({"last_timestamp": str(latest_ts)}, f)

    # Dự báo 24h tiếp theo
    seq = scaled[-window_size:].copy()
    preds = []
    for _ in range(24):
        p = model.predict(seq.reshape(1, window_size, -1), verbose=0)[0]
        preds.append(p)
        seq = np.vstack([seq[1:], p])
    orig = scaler.inverse_transform(preds)
    df_fore = pd.DataFrame(orig, columns=features).clip(lower=0).round(2)

    # Thêm cột time
    base = datetime.now(timezone("Asia/Ho_Chi_Minh")) + timedelta(days=1)
    base = base.replace(hour=0, minute=0)
    df_fore.insert(
        0, "time",
        [(base + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M") for i in range(24)]
    )
    df_fore.to_json("latest_prediction.json", orient="records", indent=2)
    print("📁 Lưu dự báo vào latest_prediction.json")

    # Cập nhật Firebase
    try:
        if not firebase_admin._apps:
            raw_fb = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
            if raw_fb:
                raw_fb = raw_fb.replace('\\n', '\n')
            if not raw_fb:
                raise ValueError("Missing FIREBASE_SERVICE_ACCOUNT_JSON")
            fb_info = json.loads(raw_fb)
            cred = credentials.Certificate(fb_info)
            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://smart-farm-6e42d-default-rtdb.firebaseio.com/"
            })
        db.reference("forecast/tomorrow").set(df_fore.to_dict(orient="records"))
        print("📡 Đã cập nhật Firebase.")
    except Exception as e:
        print("❌ Lỗi cập nhật Firebase:", e)

    # Gửi thông báo email
    send_email_notification("🟢 Dự báo mới đã được huấn luyện và cập nhật.")
    print("✅ XONG lúc", datetime.now(timezone("Asia/Ho_Chi_Minh")).strftime("%H:%M:%S %d/%m/%Y"))

if __name__ == "__main__":
    while True:
        run_training_and_forecast()
        time.sleep(300)

import traceback
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
import firebase_admin
from firebase_admin import credentials, db
import smtplib
from email.mime.text import MIMEText
from google.oauth2 import service_account
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
    print("\n🔁 Bắt đầu kiểm tra và huấn luyện...")

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    json_str = os.environ.get("google_service_json")
    sheet_url = os.environ.get("SHEET_URL")

    print("🧪 GOOGLE_SERVICE_JSON is valid:", json_str[:100] if json_str else "None")
    print("📄 SHEET_URL =", sheet_url)

    if not json_str or not sheet_url:
        print("❌ Thiếu GOOGLE_SERVICE_JSON hoặc SHEET_URL!")
        return

    try:
        print("🔍 Đang kiểm tra định dạng JSON...")
        google_key = json.loads(json_str)
        if "private_key" not in google_key:
            raise ValueError("⚠️ Không tìm thấy khóa 'private_key' trong GOOGLE_SERVICE_JSON")

        google_key["private_key"] = google_key["private_key"].replace("\\n", "\n")

        print("✅ JSON nạp thành công.")

        creds = service_account.Credentials.from_service_account_info(google_key, scopes=scope)
        print("🔑 Đã tạo credentials.")

        client = gspread.authorize(creds)
        print("📡 Đã xác thực gspread.")

        sheet = client.open_by_url(sheet_url.strip())
        worksheet = sheet.worksheet("DATA")
        data = pd.DataFrame(worksheet.get_all_records())

    except Exception as e:
        print("❌ Lỗi truy cập Google Sheets:")
        traceback.print_exc()
        print("📌 Gợi ý:")
        print("- Đảm bảo biến GOOGLE_SERVICE_JSON đúng định dạng JSON một dòng.")
        print("- Chuỗi private_key phải có dạng '\\n' thay vì dòng mới.")
        print("- Email service account phải được chia sẻ quyền chỉnh sửa Google Sheets.")
        return

    data.columns = data.columns.str.strip()
    try:
        data["timestamp"] = pd.to_datetime(data["NGÀY"] + " " + data["GIỜ"], format="%d/%m/%Y %H:%M:%S")
    except KeyError as e:
        print("❌ Lỗi cột thiếu:", e)
        print("📋 Danh sách cột:", data.columns.tolist())
        return

    data = data.sort_values("timestamp")
    data.rename(columns={
        "temperature": "temp", "humidity": "humid", "soil_moisture": "soil",
        "wind": "wind", "rain": "rain"
    }, inplace=True)

    if os.path.exists("training_data.csv"):
        old_data = pd.read_csv("training_data.csv", parse_dates=["timestamp"])
        data = pd.concat([old_data, data])
        data.drop_duplicates(subset="timestamp", keep="last", inplace=True)
        data = data.sort_values("timestamp")

    saved_timestamp = None
    if os.path.exists("last_timestamp.json"):
        with open("last_timestamp.json", "r") as f:
            saved_timestamp = pd.to_datetime(json.load(f)["last_timestamp"])

    latest_timestamp = data["timestamp"].iloc[-1]

    if saved_timestamp is not None and latest_timestamp <= saved_timestamp:
        print("🟡 KHÔNG có dữ liệu mới.")
        send_email_notification("🟡 KHÔNG có dữ liệu mới để huấn luyện.")
        return

    features = ["temp", "humid", "soil", "wind", "rain"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    model_path = "gru_weather_model.h5"
    window_size = 6

    if not os.path.exists(model_path):
        model = Sequential([
            Input(shape=(window_size, len(features))),
            GRU(units=64),
            Dense(5)
        ])
        model.compile(optimizer="adam", loss=MeanSquaredError())
    else:
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss=MeanSquaredError())

    print("🟢 Có dữ liệu mới. Đang huấn luyện...")
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])
    model.fit(np.array(X), np.array(y), epochs=100, batch_size=16,
              callbacks=[EarlyStopping(monitor="loss", patience=5)], verbose=0)
    model.save(model_path)

    data.to_csv("training_data.csv", index=False)
    with open("last_timestamp.json", "w") as f:
        json.dump({"last_timestamp": str(latest_timestamp)}, f)

    current_seq = scaled_data[-window_size:].copy()
    forecast = []
    for _ in range(24):
        x_input = current_seq.reshape(1, window_size, len(features))
        y_pred = model.predict(x_input, verbose=0)
        forecast.append(y_pred[0])
        current_seq = np.vstack([current_seq[1:], y_pred])

    forecast_original = scaler.inverse_transform(np.array(forecast))
    forecast_df = pd.DataFrame(forecast_original, columns=features).clip(lower=0).round(2)
    base_time = datetime.now(timezone("Asia/Ho_Chi_Minh")) + timedelta(days=1)
    base_time = base_time.replace(hour=0, minute=0)
    forecast_df.insert(0, "time", [(base_time + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M") for i in range(24)])
    forecast_df.to_json("latest_prediction.json", orient="records", indent=2)

    try:
        if not firebase_admin._apps:
            firebase_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
            if not firebase_key:
                raise Exception("❌ FIREBASE_SERVICE_ACCOUNT_JSON is missing")
            firebase_key = json.loads(firebase_key)
            cred = credentials.Certificate(firebase_key)
            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://smart-farm-6e42d-default-rtdb.firebaseio.com/"
            })
        db.reference("forecast/tomorrow").set(forecast_df.to_dict(orient="records"))
    except Exception as e:
        print("❌ Lỗi cập nhật Firebase:", e)

    send_email_notification("🟢 Dự báo mới đã được huấn luyện và cập nhật.")
    print("✅ XONG lúc", datetime.now(timezone("Asia/Ho_Chi_Minh")).strftime("%H:%M:%S %d/%m/%Y"))

if __name__ == "__main__":
    while True:
        run_training_and_forecast()
        time.sleep(300)
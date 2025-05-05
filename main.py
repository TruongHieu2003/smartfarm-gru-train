import os, json, time, base64
# Suppress TensorFlow and absl warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import pandas as pd
import numpy as np
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

        receiver_list = [e.strip() for e in receivers.split(",")]
        subject = "SmartFarm - Trạng thái cập nhật dữ liệu"
        body = f"{message}\n\nThời gian: {datetime.now(timezone('Asia/Ho_Chi_Minh')).strftime('%H:%M %d/%m/%Y')}"
        msg = MIMEText(body)
        msg["Subject"], msg["From"], msg["To"] = subject, smtp_user, ", ".join(receiver_list)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, receiver_list, msg.as_string())
        server.quit()
        print("📧 Đã gửi email thông báo")
    except Exception as e:
        print("❌ Lỗi gửi mail:", e)

def load_json_from_env(b64_var, txt_var):
    """
    Ưu tiên decode từ <*_B64>, nếu không có thì dùng <*_JSON> và replace '\\n' → '\n'
    """
    b64 = os.environ.get(b64_var)
    if b64:
        try:
            return base64.b64decode(b64).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Không decode được {b64_var}:", e)
    raw = os.environ.get(txt_var, "")
    return raw.replace('\\n', '\n')

def run_training_and_forecast():
    print("\n🔁 Bắt đầu kiểm tra và huấn luyện...\n")

    # --- Google Sheets credentials & data load ---
    raw_google = load_json_from_env("GOOGLE_SERVICE_JSON_B64", "GOOGLE_SERVICE_JSON")
    sheet_url = os.environ.get("SHEET_URL", "").strip()
    print("🔍 Kiểm tra biến môi trường:")
    print("GOOGLE_SERVICE_JSON tồn tại:", bool(raw_google))
    print("SHEET_URL:", sheet_url)
    if not raw_google or not sheet_url:
        print("❌ Thiếu GOOGLE_SERVICE_JSON hoặc SHEET_URL")
        return

    try:
        info = json.loads(raw_google)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
        print("✅ Authorized Google Sheets với scopes:", scopes)
    except Exception as e:
        print("❌ Lỗi xử lý GOOGLE_SERVICE_JSON hoặc authorize:", e)
        return

    try:
        sheet = client.open_by_url(sheet_url)
        df = pd.DataFrame(sheet.worksheet("DATA").get_all_records())
        print("✅ Lấy dữ liệu từ Google Sheets:", df.shape)
    except Exception as e:
        print("❌ Lỗi truy cập Google Sheets:", e)
        return

    # --- Preprocess ---
    df.columns = df.columns.str.strip()
    need = ["NGÀY","GIỜ","temperature","humidity","soil_moisture","wind","rain"]
    if not all(c in df.columns for c in need):
        print("❌ Thiếu cột:", set(need)-set(df.columns))
        return

    try:
        df['timestamp'] = pd.to_datetime(
            df['NGÀY'] + ' ' + df['GIỜ'], format='%d/%m/%Y %H:%M:%S'
        )
        print("✅ Tạo cột timestamp")
    except Exception as e:
        print("❌ Lỗi format timestamp:", e)
        return

    df = df.sort_values('timestamp').reset_index(drop=True)
    df.rename(columns={'temperature':'temp','humidity':'humid','soil_moisture':'soil'}, inplace=True)

    # --- Merge old data ---
    if os.path.exists("training_data.csv"):
        try:
            old = pd.read_csv("training_data.csv", parse_dates=["timestamp"])
            df = pd.concat([old, df], ignore_index=True)
            print(f"🔁 Nối: cũ {len(old)} + mới {len(df)-len(old)}")
        except Exception as e:
            print("⚠️ Lỗi đọc training_data.csv:", e)
    df.drop_duplicates("timestamp", keep="last", inplace=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("✅ Data sau xử lý:", df.shape)

    # --- Check last_timestamp ---
    last_ts = None
    if os.path.exists("last_timestamp.json"):
        try:
            with open("last_timestamp.json") as f:
                last_ts = pd.to_datetime(json.load(f)["last_timestamp"])
                print("🧪 Lần cuối huấn luyện:", last_ts)
        except Exception as e:
            print("⚠️ Lỗi đọc last_timestamp.json:", e)

    newest = df["timestamp"].iloc[-1]
    print("🆕 Timestamp mới nhất:", newest)
    if last_ts and newest <= last_ts:
        print("🟡 KHÔNG có dữ liệu mới.")
        send_email_notification("🟡 KHÔNG có dữ liệu mới để huấn luyện.")
        return

    # --- Scale & windowing ---
    feats = ['temp','humid','soil','wind','rain']
    df.dropna(subset=feats, inplace=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feats])
    print("✅ Scale xong:", scaled.shape)

    # --- Model setup ---
    mdl_file = "gru_weather_model.keras"
    win = 6
    if not os.path.exists(mdl_file):
        print("📦 Tạo model GRU mới")
        model = Sequential([Input((win,len(feats))), GRU(64), Dense(len(feats))])
        model.compile("adam", MeanSquaredError())
    else:
        print("📦 Load model hiện có")
        model = load_model(mdl_file, compile=False)
        model.compile("adam", MeanSquaredError())

    # --- Prepare X,y ---
    X,y = [],[]
    for i in range(len(scaled)-win):
        X.append(scaled[i:i+win])
        y.append(scaled[i+win])
    X,y = np.array(X), np.array(y)
    print(f"📊 X={X.shape}, y={y.shape}")

    # --- Train & save ---
    print("🟢 Training...")
    model.fit(X, y,
              epochs=100, batch_size=16,
              callbacks=[EarlyStopping("loss", patience=5)],
              verbose=0)
    model.save(mdl_file)
    print("✅ Model saved")

    # --- Lưu data & timestamp ---
    df.to_csv("training_data.csv", index=False)
    with open("last_timestamp.json","w") as f:
        json.dump({"last_timestamp": str(newest)}, f)

    # --- Forecast next 24h ---
    seq = scaled[-win:].copy()
    preds=[]
    for _ in range(24):
        p = model.predict(seq.reshape(1,win,-1), verbose=0)[0]
        preds.append(p)
        seq = np.vstack([seq[1:], p])
    orig = scaler.inverse_transform(preds)
    fut = pd.DataFrame(orig, columns=feats).clip(lower=0).round(2)
    base = datetime.now(timezone("Asia/Ho_Chi_Minh")) + timedelta(days=1)
    base = base.replace(hour=0, minute=0)
    fut.insert(0, "time", [(base+timedelta(hours=i)).strftime("%d/%m/%Y %H:%M") for i in range(24)])
    fut.to_json("latest_prediction.json", orient="records", indent=2)
    print("📁 Saved latest_prediction.json")

    # --- Firebase update ---
    try:
        if not firebase_admin._apps:
            raw_fb = load_json_from_env("FIREBASE_SERVICE_ACCOUNT_JSON_B64","FIREBASE_SERVICE_ACCOUNT_JSON")
            info_fb = json.loads(raw_fb)
            cred = credentials.Certificate(info_fb)
            firebase_admin.initialize_app(cred, {
                "databaseURL":"https://smart-farm-6e42d-default-rtdb.firebaseio.com/"
            })
        db.reference("forecast/tomorrow").set(fut.to_dict("records"))
        print("📡 Đã cập nhật Firebase.")
    except Exception as e:
        print("❌ Lỗi cập nhật Firebase:", e)

    # --- Gửi email ---
    send_email_notification("🟢 Dự báo mới đã được huấn luyện và cập nhật.")
    print("✅ XONG lúc", datetime.now(timezone("Asia/Ho_Chi_Minh")).strftime("%H:%M:%S %d/%m/%Y"))

if __name__ == "__main__":
    while True:
        run_training_and_forecast()
        time.sleep(600)

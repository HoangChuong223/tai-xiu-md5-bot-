import json
import time
import threading
import websocket
import ssl
from flask import Flask, render_template, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# =================== CONFIG WEBSOCKET ===================
messages_to_send = [
    [1, "MiniGame", "SC_dungtrong1205", "hoangchuong", {
        "info": '{"ipAddress":"125.235.239.187","userId":"cada855a-eff2-4494-8d69-c2a521331d97","username":"SC_dungtrong1205","timestamp":1749396871580,"refreshToken":"e0fdf802d5d24b5aab719bb77ec58fdf.4ddd9fe305264e38b6a0e82d873b56c5"}',
        "signature": "13E24C3BDBADC4C543536D3E86697E56E14FB2486D42ADB52D2D22020A200C6D85ECEA2D589903C8016C245D87628D64263132B70C39B395B27DF08F33ED05530766F68100872B423556EA1528DD57128C48578404FE5288A00E274899AACD1C4CD0FDA2A4B26ED62018409AC9E263667DB5A84C75B657A91A3E1FBB6945A63D"
    }],
    [6, "MiniGame", "taixiuPlugin", {"cmd": 1005}],
    [6, "MiniGame", "lobbyPlugin", {"cmd": 10001}]
]

# =================== FLASK APP ===================
app = Flask(__name__)
id_phien = None
ket_qua = []
last_prediction = None

# =================== AI MODEL ===================
X_train = []
y_train = []

vectorizer = CountVectorizer()
model = MultinomialNB()

def predict_from_pattern(history):
    if len(history) < 5:
        return "t"  # Mặc định nếu chưa có đủ mẫu
    pattern = "".join(history[-5:])
    X_new = vectorizer.transform([pattern])
    return model.predict(X_new)[0]

def update_patterns(history, result):
    if len(history) >= 5:
        pattern = "".join(history[-5:])
        X_train.append(pattern)
        y_train.append(result)
        X_vec = vectorizer.fit_transform(X_train)
        model.fit(X_vec, y_train)

# =================== WEBSOCKET HANDLER ===================
def on_message(ws, message):
    global id_phien, ket_qua, last_prediction
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        print("Không thể parse message:", message)
        return

    if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], dict):
        cmd = data[1].get("cmd")
        if cmd == 1008 and "sid" in data[1]:
            new_id = data[1]["sid"]
            if new_id != id_phien:
                id_phien = new_id
                last_prediction = predict_from_pattern(ket_qua)

        elif cmd == 1003 and all(key in data[1] for key in ["d1", "d2", "d3"]):
            d1, d2, d3 = data[1]["d1"], data[1]["d2"], data[1]["d3"]
            total = d1 + d2 + d3
            result_tx = "t" if total > 10 else "x"
            ket_qua.append(result_tx)
            if len(ket_qua) > 20:
                ket_qua.pop(0)
            update_patterns(ket_qua, result_tx)

def on_error(ws, error):
    print(f"Lỗi WebSocket: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Kết nối đóng: {close_status_code}, {close_msg}")

def on_open(ws):
    for msg in messages_to_send:
        ws.send(json.dumps(msg))
    print("Gửi thông tin xác thực...")

def run_websocket():
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://websocket.azhkthg1.net/websocket",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        except Exception as e:
            print(f"Lỗi chạy WebSocket: {e}")
            time.sleep(5)

# =================== FLASK ROUTES ===================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        'id_phien': id_phien,
        'lich_su': ket_qua[-10:],
        'du_doan': last_prediction
    })

if __name__ == "__main__":
    ws_thread = threading.Thread(target=run_websocket)
    ws_thread.daemon = True
    ws_thread.start()

    app.run(debug=True, port=5000)

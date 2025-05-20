import os
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import joblib
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram
from flask import Flask, request

# ================== CONFIG ==================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "default_token_for_local_testing")
PORT = int(os.environ.get("PORT", 5000))
APP_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://your-bot.onrender.com ")
bot = telegram.Bot(token=BOT_TOKEN)

# ================== HASHING ==================
def generate_md5(text):
    return hashlib.md5(text.encode()).hexdigest()

# ================== TẠO DỮ LIỆU GIẢ LẬP - PHÂN LOẠI CHUỖI ==================
def create_dataset(num_samples=5000):
    X, y = [], []
    chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*')
    for _ in range(num_samples):
        label = np.random.choice(['number', 'word', 'mixed'])
        if label == 'number':
            text = ''.join(np.random.choice(list('0123456789'), 10))
        elif label == 'word':
            text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 10))
        else:
            text = ''.join(np.random.choice(chars, 10))
        md5_hash = generate_md5(text)
        bin_vec = bin(int(md5_hash, 16))[2:].zfill(128)
        X.append([int(b) for b in bin_vec])
        y.append(label)
    return np.array(X), np.array(y)

# ================== TRAIN RANDOM FOREST MODEL ==================
RF_PATH = 'rf_md5_classifier.pkl'
if not os.path.exists(RF_PATH):
    print("Training RF model...")
    X, y = create_dataset()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    rf_model = RandomForestClassifier(n_estimators=20, max_depth=5)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, RF_PATH)
else:
    rf_model = joblib.load(RF_PATH)

# ================== TRAIN KNN SEARCH ==================
known_strings = ['password', 'admin123', 'letmein', '123456', 'hello123']
known_hashes = [generate_md5(s) for s in known_strings]
X_knn = np.array([[int(b) for b in bin(int(h, 16))[2:].zfill(128)] for h in known_hashes])

KNN_PATH = 'knn_md5_searcher.pkl'
if not os.path.exists(KNN_PATH):
    print("Training KNN model...")
    knn_model = NearestNeighbors(n_neighbors=1, metric='hamming')
    knn_model.fit(X_knn)
    joblib.dump(knn_model, KNN_PATH)
else:
    knn_model = joblib.load(KNN_PATH)

# ================== TRAIN MODEL DỰ ĐOÁN TÀI/XỈU ==================
TX_MODEL_PATH = 'tai_xiu_model.pkl'

def train_tai_xiu_model():
    test_strings = [
        *[str(x) for x in range(1000)],
        *['a'*i for i in range(5, 15)],
        *['abc123'] * 100,
        *['xyz999'] * 100,
        *['hitclub'] * 100
    ]

    X_train, y_train = [], []

    for _ in range(5000):  # Tăng số lượng sample
        text = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*'), 10))
        md5_hash = generate_md5(text)
        byte_sum = sum(bytes.fromhex(md5_hash))
        first_two = int(md5_hash[:2], 16)
        last_two = int(md5_hash[-2:], 16)
        hash_int = int(md5_hash, 16)
        bit_count = bin(hash_int).count('1')

        # Dùng trung vị để chọn ngưỡng động
        label = 'Tài' if byte_sum > 280 else 'Xỉu'

        X_train.append([byte_sum, first_two, last_two, hash_int % 256, bit_count])
        y_train.append(label)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, TX_MODEL_PATH)
    return model

# Load hoặc train mô hình Tài/Xỉu
if not os.path.exists(TX_MODEL_PATH):
    tai_xiu_model = train_tai_xiu_model()
else:
    tai_xiu_model = joblib.load(TX_MODEL_PATH)

# Hàm dự đoán Tài/Xỉu bằng AI
def predict_tai_xiu_ai(md5_hash):
    byte_sum = sum(bytes.fromhex(md5_hash))
    first_two = int(md5_hash[:2], 16)
    last_two = int(md5_hash[-2:], 16)
    hash_int = int(md5_hash, 16)
    bit_count = bin(hash_int).count('1')

    prediction = tai_xiu_model.predict([[byte_sum, first_two, last_two, hash_int % 256, bit_count]])
    return prediction[0]

# ================== HÀM PHÂN TÍCH MD5 ==================
def analyze_md5(text_input):
    if len(text_input) > 100:
        return "⚠️ Chuỗi quá dài. Vui lòng nhập dưới 100 ký tự."

    try:
        md5_hash = generate_md5(text_input)
        bin_vec = bin(int(md5_hash, 16))[2:].zfill(128)
        bin_array = np.array([[int(b) for b in bin_vec]])

        # Dự đoán loại chuỗi
        predicted_type = rf_model.predict(bin_array)[0]

        # Tìm chuỗi gần giống
        dist, idx = knn_model.kneighbors(bin_array)
        similar_str = known_strings[idx[0][0]] if dist[0][0] < 0.2 else "Không tìm thấy"

        # Dự đoán Tài/Xỉu
        tai_xiu = predict_tai_xiu_ai(md5_hash)

        result = (
            f"🔹 *Chuỗi đầu vào:* `{text_input}`\n"
            f"🔹 *MD5 Hash:* `{md5_hash}`\n\n"
            f"[AI🤖] Dự đoán loại chuỗi: *{predicted_type}*\n"
            f"[KNN⚡] Gần giống với: *{similar_str}*\n"
            f"[🎲] Dự đoán kết quả: **{tai_xiu}**"
        )
        return result
    except Exception as e:
        return f"❌ Có lỗi xảy ra: {str(e)}"

# ================== FLASK SERVER ==================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot đang hoạt động!"

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    updater = setup_updater()
    updater.dispatcher.process_update(update)
    return 'OK'

def setup_updater():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    def start(update: telegram.Update, context: telegram.ext.CallbackContext):
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="⚡ Tool Soi Mã MD5 Mạnh Mẽ\n"
                                      "ToolBy@Cskhtx1210 Chuyên HitClub 🎲",
                                 parse_mode=telegram.ParseMode.MARKDOWN)

    def handle_message(update: telegram.Update, context: telegram.ext.CallbackContext):
        user_input = update.message.text.strip()
        response = analyze_md5(user_input)
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=response,
                                 parse_mode=telegram.ParseMode.MARKDOWN)

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    return updater

# ================== START SERVER ==================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

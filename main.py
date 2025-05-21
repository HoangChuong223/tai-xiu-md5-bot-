import os
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram
from flask import Flask, request
from collections import Counter

# ================== CONFIG ==================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "default_token_for_local_testing")
PORT = int(os.environ.get("PORT", 5000))
APP_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://your-bot.onrender.com ")
bot = telegram.Bot(token=BOT_TOKEN)

# ================== HASHING ==================
def generate_md5(text):
    return hashlib.md5(text.encode()).hexdigest()

# ================== TẠO DỮ LIỆU GIẢ LẬP ==================
def generate_tai_xiu_data(num_samples=5000):
    X, y = [], []
    chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*')
    
    for _ in range(num_samples):
        text = ''.join(np.random.choice(chars, 10))
        md5_hash = generate_md5(text)
        byte_sum = sum(bytes.fromhex(md5_hash))
        first_two = int(md5_hash[:2], 16)
        last_two = int(md5_hash[-2:], 16)
        hash_int = int(md5_hash, 16)
        bit_count = bin(hash_int).count('1')

        # Ngưỡng động dựa trên trung vị tổng byte
        label = 'Tài' if byte_sum > 280 else 'Xỉu'

        X.append([byte_sum, first_two, last_two, hash_int % 256, bit_count])
        y.append(label)
    return np.array(X), np.array(y)

# ================== TRAIN RANDOM FOREST MODEL ==================
RF_PATH = 'rf_md5_classifier.pkl'
if not os.path.exists(RF_PATH):
    print("Training RF model...")
    X, y = generate_tai_xiu_data(5000)
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

# ================== TRAIN NHIỀU MÔ HÌNH DỰ ĐOÁN TÀI/XỈU ==================
TX_MODEL_DIR = 'models/'
os.makedirs(TX_MODEL_DIR, exist_ok=True)

# Huấn luyện nhiều mô hình
def train_multiple_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=50),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50),
        "SVM": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"[+] {name} đã được train")

    return trained_models

# Tạo dữ liệu giả lập & train mô hình
X, y = generate_tai_xiu_data(5000)
trained_models = train_multiple_models(X, y)

# Lưu tất cả mô hình
for name, model in trained_models.items():
    joblib.dump(model, f'{TX_MODEL_DIR}{name.lower()}_tai_xiu_model.pkl')

# Load lại mô hình
def load_all_models():
    model_paths = {
        "randomforest": f"{TX_MODEL_DIR}randomforest_tai_xiu_model.pkl",
        "gradientboosting": f"{TX_MODEL_DIR}gradientboosting_tai_xiu_model.pkl",
        "svm": f"{TX_MODEL_DIR}svm_tai_xiu_model.pkl",
        "knn": f"{TX_MODEL_DIR}knn_tai_xiu_model.pkl",
        "logisticregression": f"{TX_MODEL_DIR}logisticregression_tai_xiu_model.pkl",
        "neuralnet": f"{TX_MODEL_DIR}neuralnet_tai_xiu_model.pkl"
    }

    loaded_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                loaded_models[name.capitalize()] = joblib.load(path)
            except Exception as e:
                print(f"[!] Không thể load {name}: {str(e)}")
    return loaded_models

all_models = load_all_models()

# Hàm dự đoán với nhiều mô hình
def predict_with_all_models(md5_hash):
    byte_sum = sum(bytes.fromhex(md5_hash))
    first_two = int(md5_hash[:2], 16)
    last_two = int(md5_hash[-2:], 16)
    hash_int = int(md5_hash, 16)
    bit_count = bin(hash_int).count('1')
    features = [[byte_sum, first_two, last_two, hash_int % 256, bit_count]]

    results = {}
    for name, model in all_models.items():
        pred = model.predict(features)[0]
        results[name] = pred

    final_prediction = Counter(results.values()).most_common(1)[0][0]
    return results, final_prediction

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
        individual_preds, final_pred = predict_with_all_models(md5_hash)

        result = (
            f"🔹 *Chuỗi đầu vào:* `{text_input}`\n"
            f"🔹 *MD5 Hash:* `{md5_hash}`\n\n"
            f"[AI🤖] Dự đoán loại chuỗi: *{predicted_type}*\n"
            f"[KNN⚡] Gần giống với: *{similar_str}*\n\n"
            f"[🎲] **Dự đoán từng mô hình:**\n"
            "\n".join([f"- {k}: {v}" for k, v in individual_preds.items()]) + "\n\n"
            f"[🏆] Kết luận cuối cùng: **{final_pred}**"
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

# =============================================================================
# Giao diện Web Phát hiện Tin nhắn Spam/Phishing - Sử dụng Streamlit
# Ứng dụng cho phép người dùng nhập tin nhắn và phân loại bằng 3 mô hình
# Tác giả: Đồ án Phát hiện Tin nhắn Spam/Phishing
# =============================================================================

import streamlit as st     # Thư viện tạo giao diện web
import pandas as pd        # Xử lý dữ liệu bảng
import numpy as np         # Tính toán số học
import joblib              # Tải mô hình scikit-learn
import pickle              # Tải đối tượng Python nhị phân
import re                  # Biểu thức chính quy
import os                  # Tương tác hệ điều hành
import json                # Đọc/ghi JSON

# Đường dẫn thư mục chứa mô hình đã huấn luyện
MODEL_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model"

# --- Cấu hình trang web ---
st.set_page_config(
    page_title="Spam Detection System",   # Tiêu đề tab trình duyệt
    page_icon="🛡️",                       # Icon tab
    layout="wide"                          # Bố cục rộng
)

# --- CSS tùy chỉnh giao diện ---
st.markdown("""
<style>
    .main-title { text-align: center; color: #1E88E5; font-size: 2.5rem; font-weight: bold; margin-bottom: 0; }
    .sub-title { text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .spam-result { background-color: #FFEBEE; color: #C62828; border: 2px solid #C62828; }
    .ham-result { background-color: #E8F5E9; color: #2E7D32; border: 2px solid #2E7D32; }
</style>
""", unsafe_allow_html=True)

# --- Hàm tiền xử lý văn bản (giống khi huấn luyện) ---
def preprocess_text(text):
    """Làm sạch văn bản: chữ thường, xóa URL, xóa ký tự đặc biệt, xóa khoảng trắng thừa."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Lớp SimpleTokenizer cho CNN (cần khai báo lại để unpickle) ---
from collections import Counter

class SimpleTokenizer:
    def __init__(self, num_words=10000, oov_token='<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: 1}
        
    def fit_on_texts(self, texts):
        """Xây dựng từ điển từ danh sách văn bản."""
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        most_common = word_counts.most_common(self.num_words - 1)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_index[word] = idx
    
    def texts_to_sequences(self, texts):
        """Chuyển văn bản thành chuỗi số nguyên."""
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 1) for word in text.split()]
            sequences.append(seq)
        return sequences


# --- Tải mô hình (cache để chỉ tải 1 lần) ---
@st.cache_resource
def load_models():
    """Tải tất cả mô hình đã huấn luyện: Naive Bayes, SVM, CNN."""
    models = {}
    
    # Tải Naive Bayes
    try:
        nb_model = joblib.load(os.path.join(MODEL_PATH, "nb_model.pkl"))
        nb_tfidf = joblib.load(os.path.join(MODEL_PATH, "nb_tfidf_vectorizer.pkl"))
        models['Naive Bayes'] = {'model': nb_model, 'vectorizer': nb_tfidf, 'type': 'sklearn'}
    except:
        pass
    
    # Tải SVM
    try:
        svm_model = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
        svm_tfidf = joblib.load(os.path.join(MODEL_PATH, "svm_tfidf_vectorizer.pkl"))
        models['SVM'] = {'model': svm_model, 'vectorizer': svm_tfidf, 'type': 'sklearn'}
    except:
        pass
    
    # Tải CNN (PyTorch)
    try:
        import torch
        import torch.nn as nn
        
        # Định nghĩa lại kiến trúc CNN (phải giống khi huấn luyện)
        class SpamCNN(nn.Module):
            def __init__(self, vocab_size=10000, embedding_dim=128, max_len=150):
                super(SpamCNN, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5)
                self.pool = nn.AdaptiveMaxPool1d(1)
                self.fc1 = nn.Linear(128, 64)
                self.dropout1 = nn.Dropout(0.5)
                self.fc2 = nn.Linear(64, 32)
                self.dropout2 = nn.Dropout(0.3)
                self.fc3 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.embedding(x)
                x = x.permute(0, 2, 1)
                x = self.relu(self.conv1(x))
                x = self.pool(x).squeeze(2)
                x = self.dropout1(self.relu(self.fc1(x)))
                x = self.dropout2(self.relu(self.fc2(x)))
                x = self.sigmoid(self.fc3(x))
                return x.squeeze(1)
        
        cnn_model = SpamCNN()
        cnn_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "cnn_best_model.pth"), map_location='cpu', weights_only=True))
        cnn_model.eval()  # Chế độ đánh giá (tắt Dropout)
        
        # Tải tokenizer
        import sys
        sys.modules['__main__'].SimpleTokenizer = SimpleTokenizer
        with open(os.path.join(MODEL_PATH, "cnn_tokenizer.pkl"), 'rb') as f:
            cnn_tokenizer = pickle.load(f)
        
        models['CNN'] = {'model': cnn_model, 'tokenizer': cnn_tokenizer, 'type': 'pytorch'}
        st.sidebar.success("✅ CNN loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ CNN không load được: {str(e)[:50]}")
    
    return models

# --- Tải lịch sử huấn luyện (cache dữ liệu) ---
@st.cache_data
def load_histories():
    """Tải lịch sử huấn luyện từ file JSON."""
    histories = {}
    files = {'Naive Bayes': 'nb_history.json', 'SVM': 'svm_history.json', 'CNN': 'cnn_history.json'}
    for name, filename in files.items():
        try:
            with open(os.path.join(MODEL_PATH, filename), 'r') as f:
                histories[name] = json.load(f)
        except:
            pass
    return histories

# --- Hàm dự đoán ---
def predict(text, model_name, models):
    """
    Phân loại tin nhắn Ham/Spam.
    Trả về: (prediction, ham_prob, spam_prob)
    """
    cleaned_text = preprocess_text(text)
    model_data = models[model_name]
    
    # Mô hình sklearn (NB, SVM)
    if model_data['type'] == 'sklearn':
        vectorizer = model_data['vectorizer']
        model = model_data['model']
        text_vec = vectorizer.transform([cleaned_text])  # Vector hóa TF-IDF
        prediction = model.predict(text_vec)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vec)[0]  # Xác suất trực tiếp (NB)
            spam_prob = proba[1]
            ham_prob = proba[0]
        else:
            # SVM: chuyển decision_function thành xác suất bằng sigmoid
            decision = model.decision_function(text_vec)[0]
            spam_prob = 1 / (1 + np.exp(-decision))
            ham_prob = 1 - spam_prob
    
    # Mô hình PyTorch (CNN)
    elif model_data['type'] == 'pytorch':
        import torch
        tokenizer = model_data['tokenizer']
        model = model_data['model']
        
        seq = tokenizer.texts_to_sequences([cleaned_text])  # Token hóa
        padded = np.zeros((1, 150), dtype=np.int64)          # Padding
        if len(seq[0]) > 0:
            padded[0, :min(len(seq[0]), 150)] = seq[0][:150]
        
        with torch.no_grad():
            tensor = torch.LongTensor(padded)
            output = model(tensor).item()  # Xác suất spam (0-1)
        
        spam_prob = output
        ham_prob = 1 - output
        prediction = 1 if spam_prob > 0.5 else 0
    
    return prediction, ham_prob, spam_prob


# --- Tải mô hình và lịch sử ---
models = load_models()
histories = load_histories()

# =============================================================================
# THANH BÊN (Sidebar) - Thông tin hệ thống
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/spam.png", width=80)
    st.markdown("### 📊 Thông tin hệ thống")
    st.markdown("---")
    
    st.markdown("**📁 Dataset:**")
    st.info("SMS Spam Collection\n\n5,572 tin nhắn")
    
    st.markdown("**📈 Phân bố dữ liệu:**")
    st.write("- Ham (hợp lệ): 86.6%")
    st.write("- Spam (giả mạo): 13.4%")
    
    st.markdown("---")
    st.markdown("**🤖 Mô hình đã huấn luyện:**")
    
    # Hiển thị hiệu suất từng mô hình
    for model_name in models.keys():
        if model_name in histories:
            h = histories[model_name]
            with st.expander(f"📌 {model_name}"):
                st.write(f"Accuracy: {h['accuracy']:.2%}")
                st.write(f"Precision: {h['precision']:.2%}")
                st.write(f"Recall: {h['recall']:.2%}")
                st.write(f"F1-Score: {h['f1_score']:.2%}")
    
    st.markdown("---")
    st.markdown("**👨‍💻 Đồ án môn học**")
    st.caption("Spam/Phishing Detection")

# =============================================================================
# NỘI DUNG CHÍNH
# =============================================================================
st.markdown('<p class="main-title">🛡️ HỆ THỐNG PHÁT HIỆN TIN NHẮN GIẢ MẠO</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Spam / Phishing Detection System</p>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 10px; background-color: #E3F2FD; border-radius: 10px; margin-bottom: 20px;">
    📌 Ứng dụng sử dụng các mô hình <b>Machine Learning</b> và <b>Deep Learning</b> 
    (Naive Bayes, SVM, CNN) để phân loại tin nhắn/email là <b>HAM</b> (hợp lệ) hay <b>SPAM</b> (giả mạo).
</div>
""", unsafe_allow_html=True)

# --- Chọn mô hình ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if len(models) > 0:
        selected_model = st.selectbox(
            "🤖 Chọn mô hình phân loại:",
            list(models.keys()),
            index=list(models.keys()).index('CNN') if 'CNN' in models else 0
        )
    else:
        st.error("❌ Không tìm thấy mô hình! Vui lòng train model trước.")
        st.stop()

# --- Chọn phương thức nhập ---
st.markdown("### 📝 Nhập nội dung tin nhắn / email:")
input_method = st.radio("Chọn phương thức nhập:", ["✍️ Nhập trực tiếp", "📁 Upload file .csv"], horizontal=True)

user_input = ""
csv_data = None

if input_method == "✍️ Nhập trực tiếp":
    user_input = st.text_area("", height=150, placeholder="Nhập nội dung tin nhắn hoặc email tại đây...\n\nVí dụ: Congratulations! You have won a $1000 Walmart gift card. Click here now!")
else:
    uploaded_file = st.file_uploader("Chọn file .csv (cột 'text' chứa nội dung tin nhắn)", type=['csv'])
    if uploaded_file is not None:
        csv_data = pd.read_csv(uploaded_file)
        st.write(f"📊 Đã tải {len(csv_data)} dòng dữ liệu")
        
        # Tự động phát hiện cột văn bản
        text_col = None
        for col in ['text', 'message', 'content', 'v2', 'Text', 'MESSAGE']:
            if col in csv_data.columns:
                text_col = col
                break
        if text_col is None:
            text_col = st.selectbox("Chọn cột chứa nội dung tin nhắn:", csv_data.columns)
        st.dataframe(csv_data.head(10), use_container_width=True)

# --- Ví dụ tin nhắn mẫu ---
with st.expander("📋 Xem ví dụ tin nhắn mẫu"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 Ví dụ HAM (hợp lệ):**")
        st.code("Hey, are you coming to the party tonight?")
        st.code("I'll call you later when I'm free.")
        st.code("Can you pick up some groceries on your way home?")
    with col2:
        st.markdown("**🔴 Ví dụ SPAM (giả mạo):**")
        st.code("WINNER! You have been selected to receive a $1000 prize!")
        st.code("FREE entry to win cash! Text WIN to 80888 now!")
        st.code("Urgent! Your account has been compromised. Click here immediately!")

# --- Nút phân loại ---
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔍 Phân loại tin nhắn", type="primary", use_container_width=True)

# =============================================================================
# KẾT QUẢ DỰ ĐOÁN
# =============================================================================
if predict_btn:
    # Chế độ CSV (hàng loạt)
    if input_method == "📁 Upload file .csv" and csv_data is not None:
        with st.spinner("🔄 Đang phân tích tất cả tin nhắn..."):
            results = []
            for idx, row in csv_data.iterrows():
                text = str(row[text_col])
                pred, ham_p, spam_p = predict(text, selected_model, models)
                results.append({
                    'Text': text[:100] + '...' if len(text) > 100 else text,
                    'Prediction': 'SPAM' if pred == 1 else 'HAM',
                    'Ham %': f"{ham_p:.1%}",
                    'Spam %': f"{spam_p:.1%}"
                })
            results_df = pd.DataFrame(results)
        
        st.markdown("---")
        st.markdown("### 📊 Kết quả phân loại:")
        
        spam_count = sum(1 for r in results if r['Prediction'] == 'SPAM')
        ham_count = len(results) - spam_count
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📧 Tổng tin nhắn", len(results))
        col2.metric("✅ HAM", ham_count, delta=f"{ham_count/len(results):.1%}")
        col3.metric("🚨 SPAM", spam_count, delta=f"{spam_count/len(results):.1%}")
        
        st.dataframe(results_df, use_container_width=True)
        
        csv_result = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải kết quả (.csv)", csv_result, "spam_detection_results.csv", "text/csv")
    
    # Chế độ đơn
    elif user_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung tin nhắn!")
    else:
        with st.spinner("🔄 Đang phân tích..."):
            prediction, ham_prob, spam_prob = predict(user_input, selected_model, models)
        
        st.markdown("---")
        st.markdown("### 📊 Kết quả phân loại:")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="result-box spam-result">
                    🚨 SPAM / PHISHING<br>
                    <span style="font-size: 1rem;">Tin nhắn giả mạo</span>
                </div>
                """, unsafe_allow_html=True)
                st.error(f"📊 Xác suất SPAM: **{spam_prob:.1%}**")
            else:
                st.markdown("""
                <div class="result-box ham-result">
                    ✅ HAM<br>
                    <span style="font-size: 1rem;">Tin nhắn hợp lệ</span>
                </div>
                """, unsafe_allow_html=True)
                st.success(f"📊 Xác suất hợp lệ: **{ham_prob:.1%}**")
        
        with col2:
            # Biểu đồ xác suất (Plotly)
            st.markdown("**📈 Biểu đồ xác suất:**")
            chart_data = pd.DataFrame({
                'Loại': ['Ham (Hợp lệ)', 'Spam (Giả mạo)'],
                'Xác suất': [ham_prob * 100, spam_prob * 100]
            })
            
            import plotly.express as px
            fig = px.bar(chart_data, x='Loại', y='Xác suất', color='Loại',
                color_discrete_map={'Ham (Hợp lệ)': '#4CAF50', 'Spam (Giả mạo)': '#F44336'},
                text=chart_data['Xác suất'].apply(lambda x: f'{x:.1f}%'))
            fig.update_layout(showlegend=False, yaxis_title="Xác suất (%)", xaxis_title="", yaxis_range=[0, 100], height=300)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Chi tiết phân tích
        with st.expander("🔍 Chi tiết phân tích"):
            st.write(f"**Mô hình sử dụng:** {selected_model}")
            st.write(f"**Văn bản gốc:** {user_input[:200]}{'...' if len(user_input) > 200 else ''}")
            st.write(f"**Văn bản sau xử lý:** {preprocess_text(user_input)[:200]}")
            st.write(f"**Xác suất Ham:** {ham_prob:.4f}")
            st.write(f"**Xác suất Spam:** {spam_prob:.4f}")

# --- Chân trang ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    🎓 Đồ án môn học - Spam/Phishing Detection System<br>
    Sử dụng Machine Learning & Deep Learning
</div>
""", unsafe_allow_html=True)

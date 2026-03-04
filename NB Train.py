# =============================================================================
# Mô hình Naive Bayes (NB) cho Phát hiện Tin nhắn Spam
# Naive Bayes là thuật toán phân loại dựa trên định lý Bayes
# với giả định các đặc trưng độc lập với nhau
# Tác giả: Đồ án Phát hiện Tin nhắn Spam/Phishing
# =============================================================================

# --- Import các thư viện cần thiết ---
import pandas as pd          # Thư viện xử lý dữ liệu dạng bảng (DataFrame)
import numpy as np           # Thư viện tính toán số học với mảng đa chiều
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ và đồ thị
import seaborn as sns        # Thư viện trực quan hóa dữ liệu nâng cao
from sklearn.model_selection import train_test_split  # Hàm chia dữ liệu thành tập train/test
from sklearn.feature_extraction.text import TfidfVectorizer  # Bộ chuyển đổi văn bản thành vector TF-IDF
from sklearn.naive_bayes import MultinomialNB  # Mô hình Naive Bayes dạng Multinomial (phù hợp cho text)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# accuracy_score: Tính độ chính xác tổng thể
# precision_score: Tính độ chính xác dương (precision)
# recall_score: Tính độ nhạy (recall) - tỉ lệ phát hiện đúng
# f1_score: Tính điểm F1 - trung bình điều hòa của precision và recall
# confusion_matrix: Tạo ma trận nhầm lẫn
# classification_report: Tạo báo cáo phân loại chi tiết
from sklearn.model_selection import learning_curve  # Hàm tính đường cong học tập (learning curve)
import re      # Thư viện xử lý biểu thức chính quy (regex)
import json    # Thư viện đọc/ghi dữ liệu định dạng JSON
import joblib  # Thư viện lưu/đọc mô hình scikit-learn hiệu quả
import os      # Thư viện tương tác với hệ điều hành

# --- Đường dẫn đến các thư mục ---
DATASET_PATH = r"C:\Users\princ\Desktop\Đồ án\dataset\spam.csv"  # File dữ liệu CSV
MODEL_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model"          # Thư mục lưu mô hình
IMG_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model IMG"        # Thư mục lưu hình ảnh biểu đồ

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(IMG_PATH, exist_ok=True)

# =============================================================================
# BƯỚC 1: TẢI DỮ LIỆU (Load Dataset)
# =============================================================================
print("=" * 50)
print("NAIVE BAYES - SPAM DETECTION")
print("=" * 50)

# Đọc file CSV chứa bộ dữ liệu SMS Spam Collection
# encoding='latin-1': mã hóa ký tự phù hợp với bộ dữ liệu gốc
df = pd.read_csv(DATASET_PATH, encoding='latin-1')
# Chỉ lấy 2 cột cần thiết: 'v1' (nhãn) và 'v2' (nội dung tin nhắn)
df = df[['v1', 'v2']]
# Đổi tên cột cho dễ hiểu
df.columns = ['label', 'text']

# In thông tin cơ bản về bộ dữ liệu
print(f"\nDataset shape: {df.shape}")       # Kích thước (số hàng x số cột)
print(f"\nLabel distribution:")              # Phân bố số lượng theo nhãn
print(df['label'].value_counts())
print(f"\nLabel percentage:")                # Tỉ lệ phần trăm mỗi nhãn
print(df['label'].value_counts(normalize=True) * 100)

# =============================================================================
# BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN (Text Preprocessing)
# =============================================================================
def preprocess_text(text):
    """
    Hàm làm sạch và tiền xử lý văn bản đầu vào.
    Các bước xử lý:
    1. Chuyển toàn bộ văn bản về chữ thường
    2. Xóa tất cả các đường dẫn URL
    3. Xóa các ký tự đặc biệt và số, chỉ giữ lại chữ cái
    4. Xóa khoảng trắng thừa
    """
    text = str(text).lower()                                    # Chuyển thành chữ thường
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)        # Xóa URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)                    # Xóa ký tự đặc biệt và số
    text = re.sub(r'\s+', ' ', text).strip()                   # Xóa khoảng trắng thừa
    return text

# Áp dụng hàm tiền xử lý cho toàn bộ cột 'text'
df['cleaned_text'] = df['text'].apply(preprocess_text)

# =============================================================================
# BƯỚC 3: MÃ HÓA NHÃN (Encode Labels)
# Chuyển nhãn từ dạng chữ sang dạng số: 'ham' = 0, 'spam' = 1
# =============================================================================
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

# =============================================================================
# BƯỚC 4: CHIA DỮ LIỆU HUẤN LUYỆN / KIỂM TRA (Train/Test Split)
# Chia theo tỉ lệ 80% train, 20% test với phân tầng (stratified)
# =============================================================================
X = df['cleaned_text']       # Dữ liệu đầu vào (văn bản đã làm sạch)
y = df['label_encoded']      # Nhãn đã mã hóa
original_text = df['text']   # Văn bản gốc (để lưu lại)
labels = df['label']         # Nhãn gốc dạng chữ

# Chia dữ liệu: stratify=y đảm bảo tỉ lệ ham/spam giống nhau ở cả train và test
X_train, X_test, y_train, y_test, text_train, text_test, label_train, label_test = train_test_split(
    X, y, original_text, labels, test_size=0.2, random_state=42, stratify=y
)

# In thông tin về tập dữ liệu đã chia
print(f"\nTrain set: {len(X_train)} samples")    # Số mẫu tập huấn luyện
print(f"Test set: {len(X_test)} samples")        # Số mẫu tập kiểm tra
print(f"\nTrain label distribution:")             # Phân bố nhãn tập train
print(y_train.value_counts())
print(f"\nTest label distribution:")              # Phân bố nhãn tập test
print(y_test.value_counts())

# --- Lưu tập dữ liệu train và test ra file CSV ---
DATASET_DIR = r"C:\Users\princ\Desktop\Đồ án\dataset"
train_df = pd.DataFrame({'label': label_train.values, 'text': text_train.values})
test_df = pd.DataFrame({'label': label_test.values, 'text': text_test.values})
train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)  # Lưu file train.csv
test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)    # Lưu file test.csv
print(f"\nSaved train.csv ({len(train_df)} samples) and test.csv ({len(test_df)} samples) to dataset folder")


# =============================================================================
# BƯỚC 5: VECTOR HÓA VĂN BẢN BẰNG TF-IDF (Text Vectorization)
# TF-IDF (Term Frequency - Inverse Document Frequency):
# - TF: Tần suất từ xuất hiện trong văn bản
# - IDF: Nghịch đảo tần suất xuất hiện trong toàn bộ tập dữ liệu
# - Từ xuất hiện nhiều trong 1 văn bản nhưng ít trong tập dữ liệu -> trọng số cao
# =============================================================================
print("\nApplying TF-IDF Vectorization...")
# max_features=5000: chỉ giữ 5000 từ quan trọng nhất
# stop_words='english': loại bỏ các từ dừng tiếng Anh (the, is, at, which, ...)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)  # Học từ điển từ tập train và chuyển đổi
X_test_tfidf = tfidf.transform(X_test)        # Chỉ chuyển đổi tập test (dùng từ điển đã học)

print(f"TF-IDF features: {X_train_tfidf.shape[1]}")  # In số lượng đặc trưng (features)

# =============================================================================
# BƯỚC 6: HUẤN LUYỆN MÔ HÌNH NAIVE BAYES (Train Model)
# MultinomialNB: Naive Bayes dạng Multinomial - phù hợp với dữ liệu đếm và TF-IDF
# =============================================================================
print("\nTraining Naive Bayes model...")
nb_model = MultinomialNB()              # Tạo mô hình Naive Bayes
nb_model.fit(X_train_tfidf, y_train)    # Huấn luyện mô hình trên tập train

# =============================================================================
# BƯỚC 7: DỰ ĐOÁN (Predictions)
# =============================================================================
y_pred = nb_model.predict(X_test_tfidf)  # Dự đoán nhãn cho tập kiểm tra

# =============================================================================
# BƯỚC 8: ĐÁNH GIÁ MÔ HÌNH (Evaluation Metrics)
# =============================================================================
accuracy = accuracy_score(y_test, y_pred)    # Độ chính xác tổng thể
precision = precision_score(y_test, y_pred)  # Precision: tỉ lệ dự đoán spam đúng
recall = recall_score(y_test, y_pred)        # Recall: tỉ lệ spam được phát hiện
f1 = f1_score(y_test, y_pred)                # F1-Score: trung bình điều hòa

# In kết quả đánh giá
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")   # Độ chính xác
print(f"Precision: {precision:.4f}")  # Độ chính xác dương
print(f"Recall:    {recall:.4f}")     # Độ nhạy
print(f"F1-Score:  {f1:.4f}")         # Điểm F1

# In báo cáo phân loại chi tiết
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# =============================================================================
# BƯỚC 9: MA TRẬN NHẦM LẪN (Confusion Matrix)
# Ma trận cho biết số lượng dự đoán đúng/sai cho mỗi lớp
# =============================================================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =============================================================================
# BƯỚC 10: LƯU LỊCH SỬ HUẤN LUYỆN (Save History)
# =============================================================================
history = {
    "model": "Naive Bayes",                # Tên mô hình
    "accuracy": float(accuracy),           # Độ chính xác
    "precision": float(precision),         # Precision
    "recall": float(recall),               # Recall
    "f1_score": float(f1),                 # F1-Score
    "confusion_matrix": cm.tolist(),       # Ma trận nhầm lẫn (chuyển sang list cho JSON)
    "train_samples": len(X_train),         # Số mẫu huấn luyện
    "test_samples": len(X_test)            # Số mẫu kiểm tra
}

# Lưu lịch sử ra file JSON
with open(os.path.join(MODEL_PATH, "nb_history.json"), 'w') as f:
    json.dump(history, f, indent=4)

# =============================================================================
# BƯỚC 11: LƯU MÔ HÌNH VÀ BỘ VECTOR HÓA (Save Model & Vectorizer)
# =============================================================================
# Lưu mô hình Naive Bayes đã huấn luyện ra file .pkl
joblib.dump(nb_model, os.path.join(MODEL_PATH, "nb_model.pkl"))
# Lưu bộ TF-IDF Vectorizer (cần thiết khi dự đoán dữ liệu mới)
joblib.dump(tfidf, os.path.join(MODEL_PATH, "nb_tfidf_vectorizer.pkl"))
print(f"\nModel saved to: {MODEL_PATH}")

# =============================================================================
# BƯỚC 12: TRỰC QUAN HÓA KẾT QUẢ (Visualizations)
# =============================================================================

# --- 12.0: Đường cong học tập (Learning Curve) ---
# Learning Curve cho thấy mô hình học tốt cỡ nào khi tăng dần lượng dữ liệu huấn luyện
# Dùng để phát hiện overfitting (học thuộc) hoặc underfitting (học chưa đủ)
print("\nGenerating Learning Curve...")
train_sizes, train_scores, val_scores = learning_curve(
    MultinomialNB(), X_train_tfidf, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 mức kích thước tập train (10% -> 100%)
    cv=5,                                      # Cross-validation 5 fold
    scoring='accuracy',                        # Đánh giá bằng accuracy
    n_jobs=-1                                  # Dùng tất cả CPU cores để tăng tốc
)

# Tính giá trị trung bình và độ lệch chuẩn qua 5 fold
train_mean = train_scores.mean(axis=1)   # Accuracy trung bình trên tập train
train_std = train_scores.std(axis=1)     # Độ lệch chuẩn tập train
val_mean = val_scores.mean(axis=1)       # Accuracy trung bình trên tập validation
val_std = val_scores.std(axis=1)         # Độ lệch chuẩn tập validation

fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Tạo 2 subplot

# Biểu đồ Learning Curve - Accuracy
axes[0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
axes[0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
# Vẽ vùng ±1 độ lệch chuẩn (confidence band)
axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
axes[0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
axes[0].set_xlabel('Training Samples')    # Số lượng mẫu huấn luyện
axes[0].set_ylabel('Accuracy')            # Độ chính xác
axes[0].set_title('Naive Bayes - Learning Curve (Accuracy)')
axes[0].legend(loc='lower right')
axes[0].grid(True)

# Biểu đồ Learning Curve - Loss (dùng 1 - Accuracy làm xấp xỉ cho loss)
axes[1].plot(train_sizes, 1 - train_mean, 'o-', color='blue', label='Training Loss')
axes[1].plot(train_sizes, 1 - val_mean, 'o-', color='red', label='Validation Loss')
axes[1].fill_between(train_sizes, 1 - train_mean - train_std, 1 - train_mean + train_std, alpha=0.1, color='blue')
axes[1].fill_between(train_sizes, 1 - val_mean - val_std, 1 - val_mean + val_std, alpha=0.1, color='red')
axes[1].set_xlabel('Training Samples')
axes[1].set_ylabel('Loss (1 - Accuracy)')
axes[1].set_title('Naive Bayes - Learning Curve (Loss)')
axes[1].legend(loc='upper right')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "nb_learning_curve.png"), dpi=150)  # Lưu hình ảnh
plt.close()

# Lưu dữ liệu learning curve vào lịch sử
history["learning_curve"] = {
    "train_sizes": train_sizes.tolist(),        # Kích thước tập train ở mỗi mức
    "train_accuracy": train_mean.tolist(),      # Accuracy tập train
    "val_accuracy": val_mean.tolist()           # Accuracy tập validation
}

# --- 12.1: Biểu đồ phân bố nhãn (Label Distribution) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Biểu đồ cột: phân bố nhãn toàn bộ dataset
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Label Distribution (Full Dataset)')   # Phân bố nhãn
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Ham', 'Spam'], rotation=0)

# Biểu đồ cột nhóm: so sánh tập Train và Test
train_dist = y_train.value_counts()    # Phân bố nhãn tập train
test_dist = y_test.value_counts()      # Phân bố nhãn tập test
x = np.arange(2)
width = 0.35
axes[1].bar(x - width/2, train_dist.values, width, label='Train', color='blue')
axes[1].bar(x + width/2, test_dist.values, width, label='Test', color='orange')
axes[1].set_title('Stratified Train/Test Split')  # Phân chia theo tầng
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Count')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Ham (0)', 'Spam (1)'])
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "nb_label_distribution.png"), dpi=150)
plt.close()

# --- 12.2: Ma trận nhầm lẫn dạng Heatmap (Confusion Matrix) ---
plt.figure(figsize=(8, 6))
# Vẽ heatmap: annot=True hiển thị số, fmt='d' định dạng số nguyên, cmap='Blues' bảng màu xanh
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Naive Bayes - Confusion Matrix')
plt.xlabel('Predicted')   # Nhãn dự đoán
plt.ylabel('Actual')      # Nhãn thực tế
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "nb_confusion_matrix.png"), dpi=150)
plt.close()

# --- 12.3: Biểu đồ các chỉ số đánh giá (Metrics Bar Chart) ---
plt.figure(figsize=(8, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']  # Tên các chỉ số
values = [accuracy, precision, recall, f1]                   # Giá trị tương ứng
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']       # Màu sắc cho mỗi cột
bars = plt.bar(metrics, values, color=colors)                # Vẽ biểu đồ cột
plt.ylim(0, 1.1)                                             # Giới hạn trục Y
plt.title('Naive Bayes - Performance Metrics')               # Tiêu đề
plt.ylabel('Score')                                           # Nhãn trục Y
# Hiển thị giá trị số trên đỉnh mỗi cột
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "nb_metrics.png"), dpi=150)
plt.close()

# In thông báo hoàn tất
print(f"\nVisualization images saved to: {IMG_PATH}")
print("\n" + "=" * 50)
print("NAIVE BAYES TRAINING COMPLETED!")  # HUẤN LUYỆN NAIVE BAYES HOÀN TẤT!
print("=" * 50)

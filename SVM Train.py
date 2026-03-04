# =============================================================================
# Mô hình Máy Vector Hỗ trợ (SVM - Support Vector Machine) cho Phát hiện Spam
# SVM tìm siêu phẳng (hyperplane) tối ưu để phân tách 2 lớp dữ liệu
# với khoảng cách margin lớn nhất
# Tác giả: Đồ án Phát hiện Tin nhắn Spam/Phishing
# =============================================================================

# --- Import các thư viện cần thiết ---
import pandas as pd          # Thư viện xử lý dữ liệu dạng bảng (DataFrame)
import numpy as np           # Thư viện tính toán số học với mảng đa chiều
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ và đồ thị
import seaborn as sns        # Thư viện trực quan hóa dữ liệu nâng cao
from sklearn.model_selection import train_test_split  # Hàm chia dữ liệu train/test
from sklearn.feature_extraction.text import TfidfVectorizer  # Bộ chuyển đổi văn bản thành vector TF-IDF
from sklearn.svm import SVC  # Mô hình SVM phân loại (Support Vector Classifier)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# accuracy_score: Độ chính xác tổng thể
# precision_score: Độ chính xác dương
# recall_score: Độ nhạy (tỉ lệ phát hiện đúng)
# f1_score: Trung bình điều hòa precision và recall
# confusion_matrix: Ma trận nhầm lẫn
# classification_report: Báo cáo phân loại chi tiết
from sklearn.model_selection import learning_curve  # Hàm tính đường cong học tập
import re      # Thư viện biểu thức chính quy (regex)
import json    # Thư viện đọc/ghi JSON
import joblib  # Thư viện lưu/đọc mô hình scikit-learn
import os      # Thư viện tương tác hệ điều hành

# --- Đường dẫn đến các thư mục ---
DATASET_PATH = r"C:\Users\princ\Desktop\Đồ án\dataset\spam.csv"  # File dữ liệu CSV
MODEL_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model"          # Thư mục lưu mô hình
IMG_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model IMG"        # Thư mục lưu biểu đồ

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(IMG_PATH, exist_ok=True)

# =============================================================================
# BƯỚC 1: TẢI DỮ LIỆU (Load Dataset)
# =============================================================================
print("=" * 50)
print("SUPPORT VECTOR MACHINE (SVM) - SPAM DETECTION")
print("=" * 50)

# Đọc file CSV với mã hóa 'latin-1'
df = pd.read_csv(DATASET_PATH, encoding='latin-1')
# Chỉ lấy 2 cột cần thiết
df = df[['v1', 'v2']]
# Đổi tên cột cho dễ hiểu
df.columns = ['label', 'text']

# In thông tin cơ bản về bộ dữ liệu
print(f"\nDataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nLabel percentage:")
print(df['label'].value_counts(normalize=True) * 100)

# =============================================================================
# BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN (Text Preprocessing)
# =============================================================================
def preprocess_text(text):
    """
    Hàm làm sạch và tiền xử lý văn bản.
    Các bước:
    1. Chuyển về chữ thường
    2. Xóa URL
    3. Xóa ký tự đặc biệt và số
    4. Xóa khoảng trắng thừa
    """
    text = str(text).lower()                                    # Chuyển chữ thường
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)        # Xóa URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)                    # Xóa ký tự đặc biệt và số
    text = re.sub(r'\s+', ' ', text).strip()                   # Xóa khoảng trắng thừa
    return text

# Áp dụng hàm tiền xử lý cho toàn bộ cột text
df['cleaned_text'] = df['text'].apply(preprocess_text)

# =============================================================================
# BƯỚC 3: MÃ HÓA NHÃN (Encode Labels)
# 'ham' (tin hợp lệ) = 0, 'spam' (tin rác) = 1
# =============================================================================
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

# =============================================================================
# BƯỚC 4: CHIA DỮ LIỆU HUẤN LUYỆN / KIỂM TRA (Stratified 80/20 Split)
# =============================================================================
X = df['cleaned_text']       # Dữ liệu đầu vào
y = df['label_encoded']      # Nhãn đã mã hóa
original_text = df['text']   # Văn bản gốc
labels = df['label']         # Nhãn gốc

# Chia dữ liệu với phân tầng (stratify) để đảm bảo tỉ lệ ham/spam như nhau
X_train, X_test, y_train, y_test, text_train, text_test, label_train, label_test = train_test_split(
    X, y, original_text, labels, test_size=0.2, random_state=42, stratify=y
)

# In thông tin tập dữ liệu đã chia
print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"\nTrain label distribution:")
print(y_train.value_counts())
print(f"\nTest label distribution:")
print(y_test.value_counts())

# --- Lưu tập train và test ra file CSV ---
DATASET_DIR = r"C:\Users\princ\Desktop\Đồ án\dataset"
train_df = pd.DataFrame({'label': label_train.values, 'text': text_train.values})
test_df = pd.DataFrame({'label': label_test.values, 'text': text_test.values})
train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)
print(f"\nSaved train.csv ({len(train_df)} samples) and test.csv ({len(test_df)} samples) to dataset folder")


# =============================================================================
# BƯỚC 5: VECTOR HÓA VĂN BẢN BẰNG TF-IDF
# TF-IDF: đánh giá mức độ quan trọng của từ trong văn bản so với toàn bộ tập dữ liệu
# =============================================================================
print("\nApplying TF-IDF Vectorization...")
# max_features=5000: giữ 5000 từ quan trọng nhất
# stop_words='english': loại bỏ từ dừng tiếng Anh
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)  # Học từ điển và chuyển đổi tập train
X_test_tfidf = tfidf.transform(X_test)        # Chuyển đổi tập test (dùng từ điển đã học)

print(f"TF-IDF features: {X_train_tfidf.shape[1]}")  # Số đặc trưng TF-IDF

# =============================================================================
# BƯỚC 6: TÍNH TRỌNG SỐ LỚP (Class Weights)
# Do dữ liệu mất cân bằng (ham >> spam), cần cân bằng trọng số
# để mô hình không thiên lệch về lớp đa số (ham)
# =============================================================================
from sklearn.utils.class_weight import compute_class_weight
# Tính trọng số cân bằng tự động
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}  # Dictionary trọng số cho mỗi lớp
print(f"\nClass weights: {class_weight_dict}")

# =============================================================================
# BƯỚC 7: HUẤN LUYỆN MÔ HÌNH SVM (Train SVM Model)
# SVC: Support Vector Classifier
# kernel='linear': sử dụng kernel tuyến tính (phù hợp cho dữ liệu text)
# class_weight='balanced': tự động cân bằng trọng số lớp
# =============================================================================
print("\nTraining SVM model (this may take a while)...")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_tfidf, y_train)  # Huấn luyện mô hình

# =============================================================================
# BƯỚC 8: DỰ ĐOÁN TRÊN TẬP KIỂM TRA (Predictions)
# =============================================================================
y_pred = svm_model.predict(X_test_tfidf)  # Dự đoán nhãn cho tập test

# =============================================================================
# BƯỚC 9: ĐÁNH GIÁ MÔ HÌNH (Evaluation Metrics)
# =============================================================================
accuracy = accuracy_score(y_test, y_pred)    # Độ chính xác tổng thể
precision = precision_score(y_test, y_pred)  # Precision
recall = recall_score(y_test, y_pred)        # Recall
f1 = f1_score(y_test, y_pred)                # F1-Score

# In kết quả đánh giá
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# In báo cáo phân loại chi tiết
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# =============================================================================
# BƯỚC 10: MA TRẬN NHẦM LẪN (Confusion Matrix)
# =============================================================================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =============================================================================
# BƯỚC 11: LƯU LỊCH SỬ HUẤN LUYỆN (Save History)
# =============================================================================
history = {
    "model": "Support Vector Machine (SVM)",  # Tên mô hình
    "kernel": "linear",                        # Loại kernel sử dụng
    "class_weight": "balanced",                # Cách cân bằng trọng số lớp
    "accuracy": float(accuracy),               # Độ chính xác
    "precision": float(precision),             # Precision
    "recall": float(recall),                   # Recall
    "f1_score": float(f1),                     # F1-Score
    "confusion_matrix": cm.tolist(),           # Ma trận nhầm lẫn
    "train_samples": len(X_train),             # Số mẫu huấn luyện
    "test_samples": len(X_test)                # Số mẫu kiểm tra
}

# Lưu ra file JSON
with open(os.path.join(MODEL_PATH, "svm_history.json"), 'w') as f:
    json.dump(history, f, indent=4)

# =============================================================================
# BƯỚC 12: LƯU MÔ HÌNH VÀ BỘ VECTOR HÓA (Save Model & Vectorizer)
# =============================================================================
joblib.dump(svm_model, os.path.join(MODEL_PATH, "svm_model.pkl"))          # Lưu mô hình SVM
joblib.dump(tfidf, os.path.join(MODEL_PATH, "svm_tfidf_vectorizer.pkl"))   # Lưu bộ TF-IDF
print(f"\nModel saved to: {MODEL_PATH}")

# =============================================================================
# BƯỚC 13: TRỰC QUAN HÓA KẾT QUẢ (Visualizations)
# =============================================================================

# --- 13.0: Đường cong học tập (Learning Curve) ---
# Cho thấy hiệu suất mô hình thay đổi khi tăng dần lượng dữ liệu huấn luyện
print("\nGenerating Learning Curve (this may take a while)...")
train_sizes, train_scores, val_scores = learning_curve(
    SVC(kernel='linear', class_weight='balanced', random_state=42), 
    X_train_tfidf, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 mức kích thước (10% -> 100%)
    cv=5,                                      # 5-fold cross-validation
    scoring='accuracy',                        # Đánh giá bằng accuracy
    n_jobs=-1                                  # Song song hóa trên tất cả CPU cores
)

# Tính trung bình và độ lệch chuẩn
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Biểu đồ Learning Curve - Accuracy
axes[0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
axes[0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
# Vùng tin cậy (±1 độ lệch chuẩn)
axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
axes[0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
axes[0].set_xlabel('Training Samples')    # Số mẫu huấn luyện
axes[0].set_ylabel('Accuracy')            # Độ chính xác
axes[0].set_title('SVM - Learning Curve (Accuracy)')
axes[0].legend(loc='lower right')
axes[0].grid(True)

# Biểu đồ Learning Curve - Loss (1 - Accuracy)
axes[1].plot(train_sizes, 1 - train_mean, 'o-', color='blue', label='Training Loss')
axes[1].plot(train_sizes, 1 - val_mean, 'o-', color='red', label='Validation Loss')
axes[1].fill_between(train_sizes, 1 - train_mean - train_std, 1 - train_mean + train_std, alpha=0.1, color='blue')
axes[1].fill_between(train_sizes, 1 - val_mean - val_std, 1 - val_mean + val_std, alpha=0.1, color='red')
axes[1].set_xlabel('Training Samples')
axes[1].set_ylabel('Loss (1 - Accuracy)')
axes[1].set_title('SVM - Learning Curve (Loss)')
axes[1].legend(loc='upper right')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "svm_learning_curve.png"), dpi=150)
plt.close()

# Lưu dữ liệu learning curve vào lịch sử
history["learning_curve"] = {
    "train_sizes": train_sizes.tolist(),
    "train_accuracy": train_mean.tolist(),
    "val_accuracy": val_mean.tolist()
}

# --- 13.1: Biểu đồ phân bố nhãn (Label Distribution) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Phân bố nhãn toàn bộ dataset
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Label Distribution (Full Dataset)')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Ham', 'Spam'], rotation=0)

# So sánh tập Train và Test
train_dist = y_train.value_counts()
test_dist = y_test.value_counts()
x = np.arange(2)
width = 0.35
axes[1].bar(x - width/2, train_dist.values, width, label='Train', color='blue')
axes[1].bar(x + width/2, test_dist.values, width, label='Test', color='orange')
axes[1].set_title('Stratified Train/Test Split')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Count')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Ham (0)', 'Spam (1)'])
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "svm_label_distribution.png"), dpi=150)
plt.close()

# --- 13.2: Ma trận nhầm lẫn dạng Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',   # Bảng màu xanh lá (Greens)
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('SVM - Confusion Matrix')
plt.xlabel('Predicted')   # Nhãn dự đoán
plt.ylabel('Actual')      # Nhãn thực tế
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "svm_confusion_matrix.png"), dpi=150)
plt.close()

# --- 13.3: Biểu đồ các chỉ số đánh giá (Metrics Bar Chart) ---
plt.figure(figsize=(8, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.title('SVM - Performance Metrics')
plt.ylabel('Score')
# Hiển thị giá trị trên đỉnh mỗi cột
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "svm_metrics.png"), dpi=150)
plt.close()

# In thông báo hoàn tất
print(f"\nVisualization images saved to: {IMG_PATH}")
print("\n" + "=" * 50)
print("SVM TRAINING COMPLETED!")  # HUẤN LUYỆN SVM HOÀN TẤT!
print("=" * 50)

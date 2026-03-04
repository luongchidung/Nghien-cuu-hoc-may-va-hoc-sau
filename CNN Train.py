# =============================================================================
# Mô hình Mạng Nơ-ron Tích chập (CNN) cho Phát hiện Spam
# Sử dụng thư viện PyTorch (tương thích với Python 3.12)
# Tác giả: Đồ án Phát hiện Tin nhắn Spam/Phishing
# =============================================================================

# --- Import các thư viện cần thiết ---
import pandas as pd          # Thư viện xử lý dữ liệu dạng bảng (DataFrame)
import numpy as np           # Thư viện tính toán số học với mảng đa chiều
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ và đồ thị
import seaborn as sns        # Thư viện trực quan hóa dữ liệu nâng cao (dựa trên matplotlib)
from sklearn.model_selection import train_test_split  # Hàm chia dữ liệu thành tập huấn luyện và kiểm tra
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# accuracy_score: Tính độ chính xác tổng thể
# precision_score: Tính độ chính xác dương (precision)
# recall_score: Tính độ nhạy (recall) - tỉ lệ phát hiện đúng
# f1_score: Tính điểm F1 (trung bình điều hòa của precision và recall)
# confusion_matrix: Tạo ma trận nhầm lẫn
# classification_report: Tạo báo cáo phân loại chi tiết
import re      # Thư viện xử lý biểu thức chính quy (regex) để làm sạch văn bản
import json    # Thư viện đọc/ghi dữ liệu định dạng JSON
import os      # Thư viện tương tác với hệ điều hành (tạo thư mục, đường dẫn file, ...)
import pickle  # Thư viện lưu/đọc đối tượng Python dưới dạng nhị phân (binary)

# --- Import các thư viện PyTorch ---
import torch               # Thư viện Deep Learning PyTorch - framework chính
import torch.nn as nn       # Module chứa các lớp mạng nơ-ron (Neural Network layers)
import torch.optim as optim # Module chứa các thuật toán tối ưu hóa (optimizer)
from torch.utils.data import DataLoader, TensorDataset  # Công cụ tải và quản lý dữ liệu theo batch
from collections import Counter  # Công cụ đếm tần suất xuất hiện của các phần tử

# --- Đường dẫn đến các thư mục ---
DATASET_PATH = r"C:\Users\princ\Desktop\Đồ án\dataset\spam.csv"  # Đường dẫn file dữ liệu CSV chứa tin nhắn spam
MODEL_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model"          # Thư mục lưu mô hình đã huấn luyện
IMG_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model IMG"        # Thư mục lưu hình ảnh biểu đồ

# Tạo các thư mục nếu chưa tồn tại (exist_ok=True: không báo lỗi nếu thư mục đã có)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(IMG_PATH, exist_ok=True)

# --- Siêu tham số (Hyperparameters) - Các tham số cấu hình cho mô hình ---
MAX_WORDS = 10000      # Số lượng từ tối đa trong từ điển (chỉ giữ 10,000 từ phổ biến nhất)
MAX_LEN = 150          # Độ dài tối đa của mỗi chuỗi đầu vào (padding/cắt ngắn về 150 từ)
EMBEDDING_DIM = 128    # Số chiều của vector embedding (mỗi từ được biểu diễn bằng vector 128 chiều)
EPOCHS = 20            # Số lượt huấn luyện tối đa trên toàn bộ dữ liệu
BATCH_SIZE = 32        # Số mẫu dữ liệu xử lý trong mỗi lần cập nhật trọng số
LEARNING_RATE = 0.001  # Tốc độ học - bước nhảy khi cập nhật trọng số mô hình

# --- Thiết lập thiết bị tính toán ---
# Kiểm tra xem máy có GPU (CUDA) không, nếu có thì dùng GPU để tăng tốc, không thì dùng CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# BƯỚC 1: TẢI DỮ LIỆU (Load Dataset)
# =============================================================================
print("=" * 50)
print("CNN (PYTORCH) - SPAM DETECTION")
print("=" * 50)

# Đọc file CSV với mã hóa 'latin-1' (phù hợp với bộ dữ liệu SMS Spam Collection)
df = pd.read_csv(DATASET_PATH, encoding='latin-1')
# Chỉ lấy 2 cột cần thiết: 'v1' (nhãn) và 'v2' (nội dung tin nhắn)
df = df[['v1', 'v2']]
# Đổi tên cột cho dễ hiểu: v1 -> label (nhãn), v2 -> text (văn bản)
df.columns = ['label', 'text']

# In thông tin cơ bản về bộ dữ liệu
print(f"\nDataset shape: {df.shape}")       # Kích thước bộ dữ liệu (số hàng x số cột)
print(f"\nLabel distribution:")              # Phân bố nhãn (số lượng ham và spam)
print(df['label'].value_counts())
print(f"\nLabel percentage:")                # Tỉ lệ phần trăm của mỗi nhãn
print(df['label'].value_counts(normalize=True) * 100)


# =============================================================================
# BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN (Text Preprocessing)
# =============================================================================
def preprocess_text(text):
    """
    Hàm làm sạch và tiền xử lý văn bản đầu vào.
    Các bước xử lý:
    1. Chuyển toàn bộ văn bản về chữ thường (lowercase)
    2. Xóa tất cả các đường dẫn URL (http, https, www)
    3. Xóa các ký tự đặc biệt và số, chỉ giữ lại chữ cái và khoảng trắng
    4. Xóa khoảng trắng thừa và khoảng trắng ở đầu/cuối chuỗi
    """
    text = str(text).lower()                                    # Bước 1: Chuyển thành chữ thường
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)        # Bước 2: Xóa URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)                    # Bước 3: Xóa ký tự đặc biệt và số
    text = re.sub(r'\s+', ' ', text).strip()                   # Bước 4: Xóa khoảng trắng thừa
    return text

# Áp dụng hàm tiền xử lý cho toàn bộ cột 'text', lưu kết quả vào cột mới 'cleaned_text'
df['cleaned_text'] = df['text'].apply(preprocess_text)
# Chuyển đổi nhãn từ chữ sang số: 'ham' (tin hợp lệ) = 0, 'spam' (tin rác) = 1
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

# =============================================================================
# BƯỚC 3: BỘ TOKENIZER ĐƠN GIẢN (Simple Tokenizer)
# Tokenizer có nhiệm vụ chuyển đổi văn bản thành chuỗi số nguyên (sequences)
# Mỗi từ được gán một chỉ số (index) duy nhất trong từ điển
# =============================================================================
class SimpleTokenizer:
    def __init__(self, num_words=10000, oov_token='<OOV>'):
        """
        Khởi tạo Tokenizer.
        - num_words: Số lượng từ tối đa trong từ điển
        - oov_token: Token đại diện cho các từ ngoài từ điển (Out-Of-Vocabulary)
          được gán chỉ số = 1
        """
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: 1}  # Từ điển ban đầu chỉ chứa token OOV
        
    def fit_on_texts(self, texts):
        """
        Xây dựng từ điển từ tập dữ liệu huấn luyện.
        - Đếm tần suất xuất hiện của tất cả các từ
        - Chỉ giữ lại (num_words - 1) từ phổ biến nhất
        - Gán chỉ số cho mỗi từ, bắt đầu từ 2 (vì 0 dùng cho padding, 1 dùng cho OOV)
        """
        word_counts = Counter()  # Bộ đếm tần suất từ
        for text in texts:
            word_counts.update(text.split())  # Tách từ và đếm
        # Lấy (num_words - 1) từ xuất hiện nhiều nhất
        most_common = word_counts.most_common(self.num_words - 1)
        # Gán chỉ số cho mỗi từ (bắt đầu từ 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_index[word] = idx
    
    def texts_to_sequences(self, texts):
        """
        Chuyển đổi danh sách văn bản thành danh sách chuỗi số nguyên.
        - Mỗi từ được thay thế bằng chỉ số tương ứng trong từ điển
        - Từ không có trong từ điển sẽ được thay bằng chỉ số OOV (= 1)
        """
        sequences = []
        for text in texts:
            # Tra cứu chỉ số của mỗi từ, nếu không tìm thấy thì dùng chỉ số 1 (OOV)
            seq = [self.word_index.get(word, 1) for word in text.split()]
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, maxlen, padding='post'):
    """
    Đệm (padding) các chuỗi về cùng một độ dài cố định.
    - Nếu chuỗi ngắn hơn maxlen: thêm số 0 vào cuối (post padding)
    - Nếu chuỗi dài hơn maxlen: cắt bớt phần cuối
    Điều này đảm bảo tất cả đầu vào có cùng kích thước cho mô hình CNN.
    """
    padded = np.zeros((len(sequences), maxlen), dtype=np.int64)  # Tạo ma trận toàn số 0
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = seq[:maxlen]       # Cắt ngắn nếu quá dài
        else:
            padded[i, :len(seq)] = seq     # Đệm số 0 vào phần còn lại
    return padded

# =============================================================================
# BƯỚC 4: CHIA DỮ LIỆU HUẤN LUYỆN / KIỂM TRA (Train/Test Split)
# =============================================================================
X = df['cleaned_text'].values       # Dữ liệu đầu vào (văn bản đã làm sạch)
y = df['label_encoded'].values      # Nhãn đã mã hóa (0 = ham, 1 = spam)
original_text = df['text'].values   # Văn bản gốc (chưa xử lý) - dùng để lưu lại
labels = df['label'].values         # Nhãn gốc dạng chữ ('ham'/'spam')

# Chia dữ liệu: 80% huấn luyện (train), 20% kiểm tra (test)
# stratify=y: đảm bảo tỉ lệ ham/spam giống nhau ở cả tập train và test
# random_state=42: đặt seed để kết quả có thể tái lập (reproducible)
X_train, X_test, y_train, y_test, text_train, text_test, label_train, label_test = train_test_split(
    X, y, original_text, labels, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")  # In số lượng mẫu tập huấn luyện
print(f"Test set: {len(X_test)} samples")      # In số lượng mẫu tập kiểm tra

# --- Lưu tập dữ liệu train và test ra file CSV ---
DATASET_DIR = r"C:\Users\princ\Desktop\Đồ án\dataset"
train_df = pd.DataFrame({'label': label_train, 'text': text_train})  # Tạo DataFrame cho tập train
test_df = pd.DataFrame({'label': label_test, 'text': text_test})     # Tạo DataFrame cho tập test
train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False) # Lưu file train.csv
test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)   # Lưu file test.csv
print(f"\nSaved train.csv ({len(train_df)} samples) and test.csv ({len(test_df)} samples) to dataset folder")

# =============================================================================
# BƯỚC 5: TOKEN HÓA VÀ PADDING (Tokenization and Padding)
# Chuyển đổi văn bản thành chuỗi số và đệm về cùng độ dài
# =============================================================================
print("\nTokenizing and padding sequences...")
tokenizer = SimpleTokenizer(num_words=MAX_WORDS)  # Tạo tokenizer với từ điển tối đa 10,000 từ
tokenizer.fit_on_texts(X_train)                    # Xây dựng từ điển từ tập huấn luyện

# Chuyển văn bản thành chuỗi số nguyên
X_train_seq = tokenizer.texts_to_sequences(X_train)  # Token hóa tập huấn luyện
X_test_seq = tokenizer.texts_to_sequences(X_test)    # Token hóa tập kiểm tra

# Đệm (padding) các chuỗi về cùng độ dài MAX_LEN = 150
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)  # Padding tập huấn luyện
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)    # Padding tập kiểm tra

print(f"Vocabulary size: {len(tokenizer.word_index)}")  # In kích thước từ điển

# =============================================================================
# BƯỚC 6: TÍNH TRỌNG SỐ LỚP (Class Weights)
# Do dữ liệu mất cân bằng (ham >> spam), cần tính trọng số để mô hình
# chú ý hơn đến lớp thiểu số (spam)
# =============================================================================
from sklearn.utils.class_weight import compute_class_weight
# Tính trọng số cân bằng tự động dựa trên tỉ lệ các lớp
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# Chuyển trọng số của lớp spam thành tensor PyTorch và đưa lên thiết bị (CPU/GPU)
class_weight_tensor = torch.FloatTensor([class_weights[1]]).to(device)
print(f"\nClass weights: ham={class_weights[0]:.4f}, spam={class_weights[1]:.4f}")


# =============================================================================
# BƯỚC 7: TẠO DataLoader (Bộ tải dữ liệu theo batch)
# DataLoader giúp chia dữ liệu thành các batch nhỏ để huấn luyện hiệu quả
# =============================================================================
# Chuyển đổi dữ liệu numpy thành tensor PyTorch
X_train_tensor = torch.LongTensor(X_train_pad)   # Tensor đầu vào tập train (kiểu Long cho chỉ số từ)
y_train_tensor = torch.FloatTensor(y_train)       # Tensor nhãn tập train (kiểu Float cho hàm loss)
X_test_tensor = torch.LongTensor(X_test_pad)      # Tensor đầu vào tập test
y_test_tensor = torch.FloatTensor(y_test)          # Tensor nhãn tập test

# Tạo TensorDataset: gộp đầu vào và nhãn thành một dataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Tạo DataLoader: tự động chia dữ liệu thành các batch
# shuffle=True cho tập train: xáo trộn dữ liệu mỗi epoch để tránh overfitting
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =============================================================================
# BƯỚC 8: ĐỊNH NGHĨA MÔ HÌNH CNN (Convolutional Neural Network)
# Kiến trúc: Embedding -> Conv1D -> MaxPooling -> Dense -> Output
# =============================================================================
class SpamCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len):
        """
        Khởi tạo kiến trúc mô hình CNN cho phân loại spam.
        
        Các lớp trong mô hình:
        - Embedding: Chuyển chỉ số từ thành vector đặc (dense vector) có ý nghĩa
        - Conv1D: Tích chập 1 chiều - trích xuất các đặc trưng cục bộ từ chuỗi từ
        - AdaptiveMaxPool1d: Lấy giá trị lớn nhất từ mỗi bộ lọc (filter)
        - Fully Connected (fc1, fc2, fc3): Các lớp kết nối đầy đủ để phân loại
        - Dropout: Tắt ngẫu nhiên một số nơ-ron trong quá trình huấn luyện để tránh overfitting
        - Sigmoid: Hàm kích hoạt đầu ra, cho xác suất trong khoảng [0, 1]
        """
        super(SpamCNN, self).__init__()
        # Lớp Embedding: chuyển mỗi từ (chỉ số) thành vector embedding_dim chiều
        # padding_idx=0: vector của chỉ số 0 (padding) luôn là vector 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Lớp tích chập 1D: 128 bộ lọc, kích thước kernel = 5
        # Quét qua chuỗi embedding để tìm các mẫu (pattern) cục bộ của 5 từ liên tiếp
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5)
        # Lớp pooling thích nghi: giảm chiều đầu ra xuống còn 1 giá trị cho mỗi bộ lọc
        self.pool = nn.AdaptiveMaxPool1d(1)
        # Lớp kết nối đầy đủ thứ 1: 128 -> 64 nơ-ron
        self.fc1 = nn.Linear(128, 64)
        # Dropout 50%: tắt ngẫu nhiên 50% nơ-ron để chống overfitting
        self.dropout1 = nn.Dropout(0.5)
        # Lớp kết nối đầy đủ thứ 2: 64 -> 32 nơ-ron
        self.fc2 = nn.Linear(64, 32)
        # Dropout 30%: tắt ngẫu nhiên 30% nơ-ron
        self.dropout2 = nn.Dropout(0.3)
        # Lớp đầu ra: 32 -> 1 nơ-ron (phân loại nhị phân: spam hoặc ham)
        self.fc3 = nn.Linear(32, 1)
        # Hàm kích hoạt ReLU: f(x) = max(0, x) - giúp mô hình học các quan hệ phi tuyến
        self.relu = nn.ReLU()
        # Hàm kích hoạt Sigmoid: nén đầu ra về khoảng [0, 1] để biểu diễn xác suất
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Hàm lan truyền xuôi (forward pass) - định nghĩa luồng dữ liệu qua mô hình.
        
        Luồng dữ liệu:
        1. x (batch, 150) -> Embedding -> (batch, 150, 128)
        2. Hoán vị chiều -> (batch, 128, 150) [Conv1D yêu cầu channels ở chiều thứ 2]
        3. Conv1D + ReLU -> (batch, 128, 146) [146 = 150 - 5 + 1]
        4. MaxPool -> (batch, 128, 1) -> squeeze -> (batch, 128)
        5. FC1 + ReLU + Dropout -> (batch, 64)
        6. FC2 + ReLU + Dropout -> (batch, 32)
        7. FC3 + Sigmoid -> (batch, 1) -> squeeze -> (batch,)
        """
        x = self.embedding(x)              # Chuyển chỉ số từ thành vector embedding
        x = x.permute(0, 2, 1)             # Hoán vị chiều cho phù hợp với Conv1D
        x = self.relu(self.conv1(x))        # Áp dụng tích chập + ReLU
        x = self.pool(x).squeeze(2)         # Pooling và loại bỏ chiều thừa
        x = self.dropout1(self.relu(self.fc1(x)))  # Lớp FC1 + ReLU + Dropout
        x = self.dropout2(self.relu(self.fc2(x)))  # Lớp FC2 + ReLU + Dropout
        x = self.sigmoid(self.fc3(x))       # Lớp đầu ra + Sigmoid
        return x.squeeze(1)                 # Loại bỏ chiều thừa, trả về xác suất spam

# Khởi tạo mô hình CNN và đưa lên thiết bị tính toán (CPU hoặc GPU)
model = SpamCNN(MAX_WORDS, EMBEDDING_DIM, MAX_LEN).to(device)
print("\nModel Architecture:")    # In kiến trúc mô hình
print(model)

# =============================================================================
# BƯỚC 9: ĐỊNH NGHĨA HÀM MẤT MÁT VÀ BỘ TỐI ƯU (Loss & Optimizer)
# =============================================================================
# BCELoss: Binary Cross-Entropy Loss - hàm mất mát cho bài toán phân loại nhị phân
# So sánh xác suất dự đoán với nhãn thực tế
criterion = nn.BCELoss()
# Adam Optimizer: bộ tối ưu hóa thích nghi (adaptive) - tự điều chỉnh tốc độ học
# cho từng tham số dựa trên gradient
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================================================
# BƯỚC 10: HUẤN LUYỆN MÔ HÌNH (Training Loop)
# =============================================================================
print("\nTraining CNN model...")
# Lưu lịch sử huấn luyện để vẽ biểu đồ sau này
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
best_val_loss = float('inf')  # Giá trị loss tốt nhất (khởi tạo = vô cực)
patience = 3                   # Số epoch chờ trước khi dừng sớm (early stopping)
patience_counter = 0           # Bộ đếm cho early stopping

# Vòng lặp qua từng epoch
for epoch in range(EPOCHS):
    # --- Giai đoạn huấn luyện (Training Phase) ---
    model.train()  # Chuyển mô hình sang chế độ huấn luyện (bật Dropout, BatchNorm)
    train_loss = 0      # Tổng loss tập huấn luyện
    train_correct = 0   # Số dự đoán đúng trên tập huấn luyện
    train_total = 0     # Tổng số mẫu đã xử lý
    
    # Lặp qua từng batch dữ liệu
    for X_batch, y_batch in train_loader:
        # Đưa dữ liệu lên thiết bị tính toán (CPU/GPU)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()        # Xóa gradient cũ từ bước trước
        outputs = model(X_batch)     # Lan truyền xuôi: tính đầu ra dự đoán
        
        # Áp dụng trọng số lớp để xử lý mất cân bằng dữ liệu
        # Mẫu spam (y=1) được nhân trọng số cao hơn để mô hình chú ý hơn
        weights = torch.where(y_batch == 1, class_weight_tensor, torch.ones_like(y_batch))
        loss = (criterion(outputs, y_batch) * weights).mean()  # Tính loss có trọng số
        
        loss.backward()     # Lan truyền ngược: tính gradient cho các tham số
        optimizer.step()    # Cập nhật trọng số mô hình dựa trên gradient
        
        train_loss += loss.item()                              # Cộng dồn loss
        predicted = (outputs > 0.5).float()                    # Chuyển xác suất thành nhãn (ngưỡng 0.5)
        train_correct += (predicted == y_batch).sum().item()   # Đếm số dự đoán đúng
        train_total += y_batch.size(0)                         # Đếm tổng số mẫu
    
    # --- Giai đoạn đánh giá (Validation Phase) ---
    model.eval()  # Chuyển mô hình sang chế độ đánh giá (tắt Dropout)
    val_loss = 0       # Tổng loss tập kiểm tra
    val_correct = 0    # Số dự đoán đúng trên tập kiểm tra
    val_total = 0      # Tổng số mẫu tập kiểm tra
    
    with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm bộ nhớ và tăng tốc
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)                            # Dự đoán
            loss = criterion(outputs, y_batch)                  # Tính loss
            val_loss += loss.item()                             # Cộng dồn loss
            predicted = (outputs > 0.5).float()                 # Chuyển xác suất thành nhãn
            val_correct += (predicted == y_batch).sum().item()  # Đếm dự đoán đúng
            val_total += y_batch.size(0)                        # Đếm tổng mẫu
    
    # Tính trung bình loss và accuracy cho epoch này
    train_loss /= len(train_loader)      # Loss trung bình trên mỗi batch (tập train)
    val_loss /= len(test_loader)          # Loss trung bình trên mỗi batch (tập test)
    train_acc = train_correct / train_total  # Accuracy tập train
    val_acc = val_correct / val_total        # Accuracy tập test
    
    # Lưu lịch sử huấn luyện
    history['loss'].append(train_loss)
    history['accuracy'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)
    
    # In kết quả của epoch hiện tại
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val_Loss: {val_loss:.4f} - Val_Acc: {val_acc:.4f}")
    
    # --- Early Stopping (Dừng sớm) ---
    # Nếu val_loss giảm -> lưu mô hình tốt nhất và reset bộ đếm
    # Nếu val_loss không giảm liên tục 3 epoch -> dừng huấn luyện để tránh overfitting
    if val_loss < best_val_loss:
        best_val_loss = val_loss         # Cập nhật loss tốt nhất
        patience_counter = 0             # Reset bộ đếm
        # Lưu trọng số mô hình tốt nhất
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, "cnn_best_model.pth"))
    else:
        patience_counter += 1            # Tăng bộ đếm
        if patience_counter >= patience:  # Nếu đã chờ đủ số epoch
            print(f"Early stopping at epoch {epoch+1}")  # Dừng sớm
            break

# Tải lại mô hình tốt nhất đã lưu (mô hình có val_loss thấp nhất)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "cnn_best_model.pth")))


# =============================================================================
# BƯỚC 11: ĐÁNH GIÁ MÔ HÌNH (Model Evaluation)
# =============================================================================
model.eval()        # Chuyển sang chế độ đánh giá
all_preds = []      # Danh sách lưu tất cả dự đoán
all_labels = []     # Danh sách lưu tất cả nhãn thực tế

# Dự đoán trên toàn bộ tập kiểm tra
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)                         # Tính xác suất spam
        predicted = (outputs > 0.5).cpu().numpy()        # Chuyển thành nhãn 0/1 trên CPU
        all_preds.extend(predicted)                      # Thêm vào danh sách dự đoán
        all_labels.extend(y_batch.numpy())               # Thêm vào danh sách nhãn thực

# Chuyển đổi sang mảng numpy kiểu int
y_pred = np.array(all_preds).astype(int)       # Nhãn dự đoán
y_test_np = np.array(all_labels).astype(int)   # Nhãn thực tế

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test_np, y_pred)    # Độ chính xác tổng thể
precision = precision_score(y_test_np, y_pred)  # Precision: tỉ lệ dự đoán spam đúng / tổng dự đoán spam
recall = recall_score(y_test_np, y_pred)        # Recall: tỉ lệ spam phát hiện đúng / tổng spam thực tế
f1 = f1_score(y_test_np, y_pred)                # F1-Score: trung bình điều hòa precision và recall

# In kết quả đánh giá
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")   # Độ chính xác
print(f"Precision: {precision:.4f}")  # Độ chính xác dương
print(f"Recall:    {recall:.4f}")     # Độ nhạy
print(f"F1-Score:  {f1:.4f}")         # Điểm F1

# In báo cáo phân loại chi tiết (precision, recall, f1 cho mỗi lớp)
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred, target_names=['Ham', 'Spam']))

# Tạo và in ma trận nhầm lẫn (Confusion Matrix)
# Ma trận cho biết: True Positive, True Negative, False Positive, False Negative
cm = confusion_matrix(y_test_np, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =============================================================================
# BƯỚC 12: LƯU LỊCH SỬ HUẤN LUYỆN VÀ TOKENIZER
# =============================================================================
# Tạo dictionary chứa toàn bộ thông tin huấn luyện
history_data = {
    "model": "CNN (PyTorch)",                                          # Tên mô hình
    "architecture": "Embedding -> Conv1D -> GlobalMaxPooling -> Dense -> Output",  # Kiến trúc
    "hyperparameters": {                                               # Siêu tham số đã sử dụng
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "epochs_trained": len(history['loss']),                        # Số epoch đã huấn luyện thực tế
        "batch_size": BATCH_SIZE
    },
    "accuracy": float(accuracy),            # Độ chính xác
    "precision": float(precision),          # Precision
    "recall": float(recall),                # Recall
    "f1_score": float(f1),                  # F1-Score
    "confusion_matrix": cm.tolist(),        # Ma trận nhầm lẫn (chuyển sang list để lưu JSON)
    "train_samples": len(X_train),          # Số mẫu huấn luyện
    "test_samples": len(X_test),            # Số mẫu kiểm tra
    "training_history": history             # Lịch sử loss và accuracy qua các epoch
}

# Lưu lịch sử huấn luyện ra file JSON
with open(os.path.join(MODEL_PATH, "cnn_history.json"), 'w') as f:
    json.dump(history_data, f, indent=4)

# Lưu tokenizer ra file pickle để sử dụng khi dự đoán
with open(os.path.join(MODEL_PATH, "cnn_tokenizer.pkl"), 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"\nModel saved to: {MODEL_PATH}")  # Thông báo đã lưu mô hình

# =============================================================================
# BƯỚC 13: TRỰC QUAN HÓA KẾT QUẢ (Visualizations)
# =============================================================================

# --- 13.1: Biểu đồ lịch sử huấn luyện (Training History) ---
# Vẽ 2 biểu đồ: Loss và Accuracy qua các epoch
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Tạo figure với 2 subplot ngang

# Biểu đồ Loss (Mất mát)
axes[0].plot(history['loss'], label='Train Loss', color='blue')       # Loss tập train
axes[0].plot(history['val_loss'], label='Val Loss', color='red')      # Loss tập validation
axes[0].set_title('CNN - Training & Validation Loss')                 # Tiêu đề
axes[0].set_xlabel('Epoch')                                            # Nhãn trục X
axes[0].set_ylabel('Loss')                                             # Nhãn trục Y
axes[0].legend()                                                       # Hiển thị chú giải
axes[0].grid(True)                                                     # Hiển thị lưới

# Biểu đồ Accuracy (Độ chính xác)
axes[1].plot(history['accuracy'], label='Train Accuracy', color='blue')     # Accuracy tập train
axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='red')    # Accuracy tập validation
axes[1].set_title('CNN - Training & Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()  # Tự động điều chỉnh khoảng cách giữa các subplot
plt.savefig(os.path.join(IMG_PATH, "cnn_training_history.png"), dpi=150)  # Lưu ảnh (150 DPI)
plt.close()  # Đóng figure để giải phóng bộ nhớ

# --- 13.2: Biểu đồ phân bố nhãn (Label Distribution) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Biểu đồ cột: phân bố nhãn trên toàn bộ dataset
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Label Distribution (Full Dataset)')  # Phân bố nhãn toàn bộ dữ liệu
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Ham', 'Spam'], rotation=0)

# Biểu đồ cột nhóm: so sánh phân bố nhãn giữa tập Train và Test
train_counts = np.bincount(y_train)    # Đếm số lượng mỗi nhãn trong tập train
test_counts = np.bincount(y_test)      # Đếm số lượng mỗi nhãn trong tập test
x = np.arange(2)
width = 0.35
axes[1].bar(x - width/2, train_counts, width, label='Train', color='blue')   # Cột tập Train
axes[1].bar(x + width/2, test_counts, width, label='Test', color='orange')   # Cột tập Test
axes[1].set_title('Stratified Train/Test Split')  # Phân chia theo tầng (stratified)
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Count')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Ham (0)', 'Spam (1)'])
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "cnn_label_distribution.png"), dpi=150)
plt.close()

# --- 13.3: Ma trận nhầm lẫn dạng heatmap (Confusion Matrix Heatmap) ---
plt.figure(figsize=(8, 6))
# Vẽ heatmap với giá trị số nguyên, bảng màu cam (Oranges)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('CNN - Confusion Matrix')
plt.xlabel('Predicted')   # Trục X: nhãn dự đoán
plt.ylabel('Actual')      # Trục Y: nhãn thực tế
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "cnn_confusion_matrix.png"), dpi=150)
plt.close()

# --- 13.4: Biểu đồ các chỉ số đánh giá (Metrics Bar Chart) ---
plt.figure(figsize=(8, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']      # Tên các chỉ số
values = [accuracy, precision, recall, f1]                       # Giá trị tương ứng
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']           # Màu sắc cho mỗi cột
bars = plt.bar(metrics, values, color=colors)                    # Vẽ biểu đồ cột
plt.ylim(0, 1.1)                                                 # Giới hạn trục Y từ 0 đến 1.1
plt.title('CNN - Performance Metrics')                           # Tiêu đề biểu đồ
plt.ylabel('Score')                                               # Nhãn trục Y
# Hiển thị giá trị số trên đỉnh mỗi cột
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "cnn_metrics.png"), dpi=150)
plt.close()

# In thông báo hoàn tất
print(f"\nVisualization images saved to: {IMG_PATH}")
print("\n" + "=" * 50)
print("CNN TRAINING COMPLETED!")  # HUẤN LUYỆN CNN HOÀN TẤT!
print("=" * 50)

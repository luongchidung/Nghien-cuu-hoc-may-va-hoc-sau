**NGHIÊN CỨU VÀ ỨNG DỤNG MÔ HÌNH CNN, SVM, NB TRONG PHÁT HIỆN TIN NHẮN/EMAIL GIẢ MẠO 
**# 🛡️ Hệ thống Phát hiện Tin nhắn Spam/Phishing

Đồ án sử dụng **Machine Learning** và **Deep Learning** để phân loại tin nhắn SMS là **Ham** (hợp lệ) hay **Spam** (giả mạo).

## 📁 Cấu trúc dự án

| File | Mô tả |
|------|--------|
| `NB Train.py` | Huấn luyện mô hình **Naive Bayes** |
| `SVM Train.py` | Huấn luyện mô hình **SVM** |
| `CNN Train.py` | Huấn luyện mô hình **CNN** (PyTorch) |
| `Compare Models.py` | So sánh hiệu suất 3 mô hình |
| `Streamlit.py` | Giao diện web phân loại tin nhắn |
| `codeimage2-1.py` | Tạo hình ảnh minh họa cho báo cáo |

## ⚙️ Cài đặt

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch streamlit plotly wordcloud joblib
```

## 🚀 Hướng dẫn sử dụng

### Bước 1: Huấn luyện mô hình (chạy lần lượt)

```bash
python "NB Train.py"
python "SVM Train.py"
python "CNN Train.py"
```

> ⚠️ Đảm bảo file `dataset/spam.csv` tồn tại trước khi chạy.

### Bước 2: So sánh mô hình

```bash
python "Compare Models.py"
```

### Bước 3: Chạy giao diện web

```bash
streamlit run Streamlit.py
```

Truy cập `http://localhost:8501` → Chọn mô hình → Nhập tin nhắn → Nhấn **Phân loại**.

### Tạo hình ảnh báo cáo (tùy chọn)

```bash
python codeimage2-1.py
```

## 📊 Mô hình

| Mô hình | Thuật toán | Đặc trưng |
|---------|-----------|-----------|
| Naive Bayes | Multinomial NB | TF-IDF |
| SVM | Linear SVC | TF-IDF |
| CNN | Conv1D + Dense | Word Embedding |

## 📂 Thư mục đầu ra

- `Train Model/` — File mô hình (.pkl, .pth) và lịch sử (.json)
- `Train Model IMG/` — Biểu đồ đánh giá (.png)
- `images/` — Hình ảnh minh họa cho báo cáo

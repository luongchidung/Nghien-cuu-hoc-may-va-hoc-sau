# =============================================================================
# So sánh Hiệu suất Tất cả Mô hình - Phát hiện Spam
# Script này tải kết quả huấn luyện từ 3 mô hình (Naive Bayes, SVM, CNN)
# và so sánh hiệu suất thông qua các biểu đồ trực quan
# Tác giả: Đồ án Phát hiện Tin nhắn Spam/Phishing
# =============================================================================

# --- Import các thư viện cần thiết ---
import json               # Thư viện đọc/ghi dữ liệu JSON
import os                 # Thư viện tương tác hệ điều hành
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
import numpy as np        # Thư viện tính toán số học
import seaborn as sns     # Thư viện trực quan hóa dữ liệu nâng cao

# --- Đường dẫn ---
MODEL_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model"      # Thư mục chứa mô hình và lịch sử
IMG_PATH = r"C:\Users\princ\Desktop\Đồ án\Train Model IMG"    # Thư mục lưu hình ảnh biểu đồ

print("=" * 60)
print("MODEL COMPARISON - SPAM DETECTION")  # SO SÁNH MÔ HÌNH - PHÁT HIỆN SPAM
print("=" * 60)

# =============================================================================
# TẢI CÁC FILE LỊCH SỬ HUẤN LUYỆN (Load History Files)
# Mỗi mô hình sau khi huấn luyện đã lưu kết quả vào file JSON
# =============================================================================
models = {}  # Dictionary lưu dữ liệu lịch sử của từng mô hình
history_files = {
    'Naive Bayes': 'nb_history.json',   # File lịch sử Naive Bayes
    'SVM': 'svm_history.json',          # File lịch sử SVM
    'CNN': 'cnn_history.json'           # File lịch sử CNN
}

# Duyệt qua từng mô hình và tải file lịch sử tương ứng
for model_name, filename in history_files.items():
    filepath = os.path.join(MODEL_PATH, filename)      # Tạo đường dẫn đầy đủ
    if os.path.exists(filepath):                        # Kiểm tra file có tồn tại không
        with open(filepath, 'r') as f:
            models[model_name] = json.load(f)           # Đọc dữ liệu JSON
        print(f"✓ Loaded {model_name} history")         # Thông báo tải thành công
    else:
        print(f"✗ {model_name} history not found")      # Thông báo không tìm thấy

# Kiểm tra: nếu không có mô hình nào được tải -> dừng chương trình
if len(models) == 0:
    print("\nNo model histories found. Please train the models first.")
    exit()

# =============================================================================
# TRÍCH XUẤT CÁC CHỈ SỐ ĐÁNH GIÁ (Extract Metrics)
# =============================================================================
metrics_data = {
    'Model': [],       # Tên mô hình
    'Accuracy': [],    # Độ chính xác tổng thể
    'Precision': [],   # Độ chính xác dương
    'Recall': [],      # Độ nhạy (tỉ lệ phát hiện spam)
    'F1-Score': []     # Trung bình điều hòa precision và recall
}

# In bảng so sánh hiệu suất
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")  # SO SÁNH HIỆU SUẤT
print("=" * 60)
print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 60)

# Duyệt qua từng mô hình, trích xuất và in các chỉ số
for model_name, data in models.items():
    metrics_data['Model'].append(model_name)
    metrics_data['Accuracy'].append(data['accuracy'])
    metrics_data['Precision'].append(data['precision'])
    metrics_data['Recall'].append(data['recall'])
    metrics_data['F1-Score'].append(data['f1_score'])
    
    # In kết quả của mỗi mô hình, căn lề trái với 4 chữ số thập phân
    print(f"{model_name:<15} {data['accuracy']:<12.4f} {data['precision']:<12.4f} {data['recall']:<12.4f} {data['f1_score']:<12.4f}")

# =============================================================================
# TÌM MÔ HÌNH TỐT NHẤT CHO TỪNG CHỈ SỐ (Find Best Model)
# =============================================================================
print("\n" + "=" * 60)
print("BEST MODEL BY METRIC")  # MÔ HÌNH TỐT NHẤT THEO TỪNG CHỈ SỐ
print("=" * 60)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    best_idx = np.argmax(metrics_data[metric])           # Tìm chỉ số (index) của giá trị lớn nhất
    best_model = metrics_data['Model'][best_idx]         # Tên mô hình tốt nhất
    best_value = metrics_data[metric][best_idx]          # Giá trị cao nhất
    print(f"{metric:<12}: {best_model} ({best_value:.4f})")

# =============================================================================
# BIỂU ĐỒ 1: Biểu đồ cột nhóm - So sánh tất cả chỉ số (Grouped Bar Chart)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_data['Model']))   # Vị trí trên trục X cho mỗi mô hình
width = 0.2                                  # Chiều rộng mỗi cột
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']  # Màu cho mỗi chỉ số

# Vẽ từng nhóm cột (mỗi nhóm = 1 chỉ số)
for i, (metric, color) in enumerate(zip(metrics_list, colors)):
    bars = ax.bar(x + i * width, metrics_data[metric], width, label=metric, color=color)
    # Hiển thị giá trị số trên đỉnh mỗi cột
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",  # Dịch lên 3 điểm
                    ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Model')                          # Nhãn trục X
ax.set_ylabel('Score')                          # Nhãn trục Y
ax.set_title('Model Comparison - All Metrics')  # Tiêu đề: So sánh tất cả chỉ số
ax.set_xticks(x + width * 1.5)                  # Đặt vị trí nhãn trục X ở giữa nhóm
ax.set_xticklabels(metrics_data['Model'])       # Nhãn trục X là tên mô hình
ax.legend()                                      # Hiển thị chú giải
ax.set_ylim(0, 1.15)                             # Giới hạn trục Y
ax.grid(axis='y', alpha=0.3)                     # Hiển thị lưới ngang mờ

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "comparison_all_metrics.png"), dpi=150)  # Lưu hình
plt.close()
print(f"\n✓ Saved: comparison_all_metrics.png")

# =============================================================================
# BIỂU ĐỒ 2: Ma trận nhầm lẫn cạnh nhau (Confusion Matrices Side by Side)
# So sánh trực quan ma trận nhầm lẫn của tất cả mô hình
# =============================================================================
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
# Nếu chỉ có 1 mô hình, axes không phải là list -> chuyển thành list
if len(models) == 1:
    axes = [axes]

cmaps = ['Blues', 'Greens', 'Oranges']  # Bảng màu cho mỗi mô hình
for idx, (model_name, data) in enumerate(models.items()):
    cm = np.array(data['confusion_matrix'])  # Lấy ma trận nhầm lẫn
    # Vẽ heatmap cho ma trận nhầm lẫn
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[idx % 3],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                ax=axes[idx])
    axes[idx].set_title(f'{model_name}\nConfusion Matrix')  # Tiêu đề
    axes[idx].set_xlabel('Predicted')   # Nhãn dự đoán
    axes[idx].set_ylabel('Actual')      # Nhãn thực tế

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "comparison_confusion_matrices.png"), dpi=150)
plt.close()
print(f"✓ Saved: comparison_confusion_matrices.png")

# =============================================================================
# BIỂU ĐỒ 3: Biểu đồ Radar (Radar Chart)
# Giúp so sánh trực quan đa chiều giữa các mô hình
# Mỗi trục đại diện cho 1 chỉ số đánh giá
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))  # Tạo biểu đồ dạng cực (polar)
categories = metrics_list         # Các chỉ số đánh giá
N = len(categories)               # Số trục
# Tính góc cho mỗi trục (chia đều 360 độ)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Đóng vòng tròn (nối điểm cuối với điểm đầu)

colors_radar = ['#3498db', '#2ecc71', '#e74c3c']  # Màu cho mỗi mô hình
for idx, model_name in enumerate(metrics_data['Model']):
    # Lấy giá trị các chỉ số của mô hình hiện tại
    values = [metrics_data[m][idx] for m in metrics_list]
    values += values[:1]  # Đóng vòng tròn
    # Vẽ đường nối và tô màu vùng bên trong
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors_radar[idx % 3])
    ax.fill(angles, values, alpha=0.25, color=colors_radar[idx % 3])  # Tô vùng bên trong (mờ 25%)

ax.set_xticks(angles[:-1])                # Đặt vị trí nhãn trục
ax.set_xticklabels(categories)            # Nhãn cho mỗi trục
ax.set_ylim(0, 1)                          # Giới hạn tầm giá trị (0 -> 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))  # Chú giải ở góc phải trên
ax.set_title('Model Comparison - Radar Chart', size=14, y=1.1)  # Tiêu đề

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "comparison_radar_chart.png"), dpi=150)
plt.close()
print(f"✓ Saved: comparison_radar_chart.png")

# =============================================================================
# BIỂU ĐỒ 4: So sánh Recall và F1-Score (Quan trọng cho Phát hiện Spam)
# Recall rất quan trọng vì việc bỏ sót spam (False Negative) nguy hiểm hơn
# việc phân loại nhầm ham thành spam (False Positive)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
recall_values = metrics_data['Recall']     # Giá trị Recall của các mô hình
f1_values = metrics_data['F1-Score']       # Giá trị F1-Score của các mô hình
model_names = metrics_data['Model']        # Tên các mô hình

x = np.arange(len(model_names))
width = 0.35

# Vẽ 2 nhóm cột: Recall (đỏ) và F1-Score (tím)
bars1 = ax.bar(x - width/2, recall_values, width, label='Recall (Spam)', color='#e74c3c')
bars2 = ax.bar(x + width/2, f1_values, width, label='F1-Score', color='#9b59b6')

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Recall & F1-Score Comparison\n(Critical Metrics for Spam Detection)')
# Tiêu đề: So sánh Recall & F1 (Các chỉ số quan trọng cho phát hiện Spam)
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.set_ylim(0, 1.1)

# Hiển thị giá trị trên đỉnh mỗi cột Recall
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
# Hiển thị giá trị trên đỉnh mỗi cột F1-Score
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "comparison_recall_f1.png"), dpi=150)
plt.close()
print(f"✓ Saved: comparison_recall_f1.png")

# =============================================================================
# LƯU BÁO CÁO TỔNG HỢP SO SÁNH (Save Comparison Summary)
# =============================================================================
summary = {
    "models_compared": list(models.keys()),   # Danh sách các mô hình được so sánh
    "metrics": metrics_data,                   # Dữ liệu chỉ số đánh giá
    "best_by_metric": {                        # Mô hình tốt nhất theo từng chỉ số
        metric: {
            "model": metrics_data['Model'][np.argmax(metrics_data[metric])],  # Tên mô hình
            "value": float(max(metrics_data[metric]))                          # Giá trị cao nhất
        }
        for metric in metrics_list
    }
}

# Lưu báo cáo ra file JSON
with open(os.path.join(MODEL_PATH, "comparison_summary.json"), 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n✓ Saved: comparison_summary.json")
print("\n" + "=" * 60)
print("COMPARISON COMPLETED!")  # SO SÁNH HOÀN TẤT!
print("=" * 60)

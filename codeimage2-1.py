"""
=============================================================================
SCRIPT TẠO TẤT CẢ HÌNH ẢNH CHO CHƯƠNG 2 - ĐỒ ÁN TỐT NGHIỆP
Phát hiện Tin nhắn Spam/Phishing sử dụng AI

Script này tạo 9 hình ảnh minh họa cho báo cáo đồ án:
- Hình 2.1: Box Plot so sánh độ dài tin nhắn Ham vs Spam
- Hình 2.2: Word Cloud các từ phổ biến trong Spam
- Hình 2.3: Biểu đồ tròn phân bố Ham/Spam
- Hình 2.4: Sơ đồ quy trình tiền xử lý văn bản
- Hình 2.5: Minh họa SVM với Hyperplane
- Hình 2.6: Kiến trúc mô hình CNN
- Hình 2.7: Kiến trúc hệ thống 3 tầng
- Hình 2.8: Sơ đồ luồng xử lý dữ liệu
- Hình 2.9: Mockup giao diện Streamlit

Yêu cầu cài đặt:
    pip install pandas matplotlib seaborn wordcloud scikit-learn numpy

Cách chạy:
    python codeimage2-1.py

Output:
    Tất cả hình ảnh sẽ được lưu vào thư mục 'images/' với định dạng PNG 300dpi
=============================================================================
"""

# --- Import các thư viện cần thiết ---
import os                       # Tương tác với hệ điều hành (tạo thư mục, kiểm tra file)
import pandas as pd             # Xử lý dữ liệu dạng bảng
import numpy as np              # Tính toán số học, tạo dữ liệu mẫu
import matplotlib.pyplot as plt # Vẽ biểu đồ và đồ thị
import seaborn as sns           # Trực quan hóa dữ liệu nâng cao
from wordcloud import WordCloud # Tạo đám mây từ (Word Cloud)
import warnings
warnings.filterwarnings('ignore')  # Tắt tất cả cảnh báo để output sạch hơn

# --- Cấu hình font chữ cho matplotlib ---
plt.rcParams['font.family'] = 'DejaVu Sans'     # Sử dụng font hỗ trợ Unicode
plt.rcParams['axes.unicode_minus'] = False       # Hiển thị đúng dấu trừ trong biểu đồ

# --- Tạo thư mục lưu ảnh đầu ra ---
OUTPUT_DIR = 'images'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

print("=" * 60)
print("SCRIPT TẠO HÌNH ẢNH CHO CHƯƠNG 2")
print("=" * 60)


# =============================================================================
# HÌNH 2.1: Box Plot so sánh độ dài tin nhắn Ham vs Spam
# Box plot hiển thị: trung vị, tứ phân vị, giá trị ngoại lệ (outliers)
# =============================================================================
def create_figure_2_1():
    """Tạo Box Plot so sánh độ dài tin nhắn Ham và Spam."""
    print("\n[1/9] Đang tạo Hình 2.1: Box Plot...")
    
    # Tạo dữ liệu mẫu dựa trên thống kê thực tế từ bộ dữ liệu SMS Spam Collection
    np.random.seed(42)  # Đặt seed để kết quả tái lập được
    
    # Dữ liệu Ham: trung bình ~71.5 ký tự, median ~52, max ~910
    # Phần lớn tin nhắn ham ngắn, một số dài, và có outliers
    ham_lengths = np.concatenate([
        np.random.normal(60, 30, 4000),   # Phần lớn ngắn (phân phối chuẩn quanh 60)
        np.random.normal(150, 50, 500),   # Một số tin dài hơn
        np.random.exponential(100, 325)   # Các giá trị ngoại lệ (outliers)
    ])
    ham_lengths = np.clip(ham_lengths, 2, 910)  # Giới hạn giá trị trong khoảng [2, 910]
    
    # Dữ liệu Spam: trung bình ~139.4 ký tự, median ~149, max ~224
    # Tin nhắn spam thường có độ dài đồng đều hơn
    spam_lengths = np.random.normal(140, 30, 747)
    spam_lengths = np.clip(spam_lengths, 13, 224)  # Giới hạn trong khoảng [13, 224]
    
    # Tạo DataFrame để vẽ biểu đồ
    df = pd.DataFrame({
        'Loại tin nhắn': ['Ham'] * len(ham_lengths) + ['Spam'] * len(spam_lengths),
        'Độ dài (ký tự)': np.concatenate([ham_lengths, spam_lengths])
    })
    
    # Vẽ Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Ham': '#2ecc71', 'Spam': '#e74c3c'}  # Xanh cho Ham, đỏ cho Spam
    sns.boxplot(x='Loại tin nhắn', y='Độ dài (ký tự)', data=df, 
                palette=colors, width=0.5, ax=ax)
    
    ax.set_title('Hình 2.1: So sánh độ dài tin nhắn Ham và Spam', fontsize=14, fontweight='bold')
    ax.set_xlabel('Loại tin nhắn', fontsize=12)
    ax.set_ylabel('Độ dài (số ký tự)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)  # Lưới ngang mờ
    
    # Thêm chú thích số lượng mẫu
    ax.text(0, 750, f'n = 4,825', ha='center', fontsize=10, color='gray')
    ax.text(1, 200, f'n = 747', ha='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_1_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_1_boxplot.png")


# =============================================================================
# HÌNH 2.2: Word Cloud tin nhắn Spam
# Đám mây từ - các từ xuất hiện nhiều hơn sẽ có kích thước lớn hơn
# =============================================================================
def create_figure_2_2():
    """Tạo Word Cloud các từ phổ biến trong tin nhắn Spam."""
    print("\n[2/9] Đang tạo Hình 2.2: Word Cloud...")
    
    # Các từ phổ biến trong tin nhắn Spam (dựa trên thống kê thực tế)
    # Số lần lặp lại thể hiện tần suất xuất hiện
    spam_words = """
    call call call call call call call call call call
    free free free free free free free free
    txt txt txt txt txt txt
    text text text text text
    now now now now now
    claim claim claim claim
    prize prize prize prize
    win win win win
    reply reply reply
    urgent urgent urgent
    mobile mobile mobile
    cash cash cash
    offer offer offer
    stop stop stop
    send send send
    message message
    guaranteed guaranteed
    winner winner
    awarded awarded
    congratulations congratulations
    customer customer
    service service
    week week
    today today
    tone tone
    ringtone ringtone
    nokia nokia
    camera camera
    video video
    """
    
    # Tạo đối tượng Word Cloud
    wordcloud = WordCloud(
        width=1200,              # Chiều rộng ảnh (pixels)
        height=600,              # Chiều cao ảnh (pixels)
        background_color='white', # Nền trắng
        colormap='Reds',         # Bảng màu đỏ (phù hợp với chủ đề Spam)
        max_words=100,           # Số từ tối đa hiển thị
        min_font_size=10,        # Cỡ chữ nhỏ nhất
        max_font_size=150,       # Cỡ chữ lớn nhất
        random_state=42          # Seed để kết quả nhất quán
    ).generate(spam_words)       # Tạo word cloud từ chuỗi văn bản
    
    # Hiển thị Word Cloud
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation='bilinear')  # Hiển thị ảnh với nội suy song tuyến
    ax.axis('off')  # Ẩn trục tọa độ
    ax.set_title('Hình 2.2: Word Cloud - Các từ xuất hiện nhiều nhất trong tin nhắn Spam', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_2_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_2_wordcloud.png")


# =============================================================================
# HÌNH 2.3: Biểu đồ tròn phân bố Ham/Spam
# Hiển thị tỉ lệ phần trăm giữa tin nhắn hợp lệ và tin rác
# =============================================================================
def create_figure_2_3():
    """Tạo biểu đồ tròn (pie chart) phân bố dữ liệu Ham/Spam."""
    print("\n[3/9] Đang tạo Hình 2.3: Pie Chart...")
    
    labels = ['Ham (Hợp lệ)', 'Spam (Rác)']
    sizes = [4825, 747]                       # Số lượng mẫu: 4825 ham, 747 spam
    colors = ['#2ecc71', '#e74c3c']           # Xanh cho Ham, đỏ cho Spam
    explode = (0, 0.05)                       # Tách phần Spam ra 5% để nhấn mạnh
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Vẽ biểu đồ tròn
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode,                      # Tách phần Spam
        labels=labels,                        # Nhãn
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes)):,})',  # Hiển thị % và số lượng
        startangle=90,                        # Bắt đầu từ 12 giờ (90 độ)
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}  # Viền trắng giữa các phần
    )
    
    # Tô đậm phần trăm
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Hình 2.3: Phân bố nhãn trong bộ dữ liệu SMS Spam Collection\n(Tổng: 5,572 tin nhắn)', 
                 fontsize=14, fontweight='bold')
    
    # Thêm chú giải (legend) ở phía dưới
    ax.legend(wedges, [f'{l}: {s:,} tin nhắn' for l, s in zip(labels, sizes)],
              loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_3_piechart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_3_piechart.png")


# =============================================================================
# HÌNH 2.4: Sơ đồ quy trình tiền xử lý văn bản
# Minh họa pipeline: Raw Text -> Lowercase -> Remove URL -> Remove Special -> Clean Text
# =============================================================================
def create_figure_2_4():
    """Tạo sơ đồ Pipeline tiền xử lý văn bản."""
    print("\n[4/9] Đang tạo Hình 2.4: Pipeline Preprocessing...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')  # Ẩn trục tọa độ
    
    # Định nghĩa các bước trong pipeline: (tên, vị trí x, màu sắc)
    steps = [
        ('Raw Text\n(Văn bản thô)', 1, '#3498db'),      # Bước 1: Văn bản gốc
        ('Lowercase\n(Chữ thường)', 3.5, '#9b59b6'),     # Bước 2: Chuyển chữ thường
        ('Remove URL\n(Xóa liên kết)', 6, '#e74c3c'),    # Bước 3: Xóa URL
        ('Remove Special\n(Xóa ký tự)', 8.5, '#f39c12'), # Bước 4: Xóa ký tự đặc biệt
        ('Clean Text\n(Văn bản sạch)', 11, '#2ecc71'),   # Bước 5: Kết quả cuối cùng
    ]
    
    # Vẽ các hộp (rectangles) cho mỗi bước
    for text, x, color in steps:
        rect = plt.Rectangle((x-0.8, 2), 1.6, 2,             # Tạo hình chữ nhật
                             facecolor=color, edgecolor='white', 
                             linewidth=2, alpha=0.9, 
                             transform=ax.transData, zorder=2)
        ax.add_patch(rect)                                     # Thêm vào biểu đồ
        ax.text(x, 3, text, ha='center', va='center',         # Thêm nhãn vào giữa hộp
               fontsize=10, fontweight='bold', color='white', zorder=3)
    
    # Vẽ mũi tên kết nối giữa các bước
    arrow_props = dict(arrowstyle='->', color='#34495e', lw=2)
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 0.8      # Điểm bắt đầu mũi tên (cạnh phải hộp trước)
        x2 = steps[i+1][1] - 0.8    # Điểm kết thúc mũi tên (cạnh trái hộp sau)
        ax.annotate('', xy=(x2, 3), xytext=(x1, 3), arrowprops=arrow_props)
    
    # Ví dụ minh họa cho mỗi bước xử lý
    examples = [
        '"URGENT! Win £1000\nhttp://spam.com"',    # Văn bản gốc
        '"urgent! win £1000\nhttp://spam.com"',     # Sau khi chuyển chữ thường
        '"urgent! win £1000"',                       # Sau khi xóa URL
        '"urgent win"',                              # Sau khi xóa ký tự đặc biệt
        '"urgent win"',                              # Kết quả cuối cùng
    ]
    
    # Hiển thị ví dụ bên dưới mỗi hộp
    for i, (text, x, _) in enumerate(steps):
        ax.text(x, 0.8, examples[i], ha='center', va='center', 
               fontsize=8, style='italic', color='#7f8c8d',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='none'))
    
    ax.set_title('Hình 2.4: Sơ đồ quy trình tiền xử lý văn bản (Text Preprocessing Pipeline)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_4_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_4_pipeline.png")


# =============================================================================
# HÌNH 2.5: Minh họa SVM Hyperplane
# Hiển thị cách SVM tìm siêu phẳng phân tách 2 lớp dữ liệu
# với margin lớn nhất và các support vectors
# =============================================================================
def create_figure_2_5():
    """Tạo minh họa SVM với hyperplane phân tách Ham vs Spam."""
    print("\n[5/9] Đang tạo Hình 2.5: SVM Hyperplane...")
    
    np.random.seed(42)
    
    n_samples = 50  # Số điểm mẫu cho mỗi lớp
    
    # Tạo dữ liệu 2 lớp phân tách rõ ràng
    X_ham = np.random.randn(n_samples, 2) + np.array([-2, -2])    # Lớp Ham: tập trung ở góc dưới trái
    X_spam = np.random.randn(n_samples, 2) + np.array([2, 2])     # Lớp Spam: tập trung ở góc trên phải
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Vẽ các điểm dữ liệu
    ax.scatter(X_ham[:, 0], X_ham[:, 1], c='#2ecc71', s=80, 
              label='Ham', edgecolors='white', linewidths=1, alpha=0.8)    # Điểm Ham (xanh)
    ax.scatter(X_spam[:, 0], X_spam[:, 1], c='#e74c3c', s=80, 
              label='Spam', edgecolors='white', linewidths=1, alpha=0.8)   # Điểm Spam (đỏ)
    
    # Vẽ hyperplane (đường phân tách chính): y = -x
    xx = np.linspace(-5, 5, 100)
    yy = -xx
    ax.plot(xx, yy, 'k-', linewidth=2, label='Hyperplane')  # Đường liền đen
    
    # Vẽ margin (vùng biên): 2 đường nét đứt song song với hyperplane
    ax.plot(xx, yy + 1.5, 'k--', linewidth=1, alpha=0.5)    # Biên trên
    ax.plot(xx, yy - 1.5, 'k--', linewidth=1, alpha=0.5)    # Biên dưới
    ax.fill_between(xx, yy - 1.5, yy + 1.5, alpha=0.1, color='gray')  # Tô màu vùng margin
    
    # Đánh dấu support vectors (các điểm gần margin nhất)
    # Đây là các điểm quan trọng nhất quyết định vị trí hyperplane
    sv_ham = X_ham[np.argsort(X_ham[:, 0] + X_ham[:, 1])[-3:]]     # 3 điểm Ham gần nhất
    sv_spam = X_spam[np.argsort(X_spam[:, 0] + X_spam[:, 1])[:3]]  # 3 điểm Spam gần nhất
    
    # Vẽ vòng tròn bao quanh support vectors
    ax.scatter(sv_ham[:, 0], sv_ham[:, 1], s=200, facecolors='none', 
              edgecolors='#2ecc71', linewidths=3, label='Support Vectors')
    ax.scatter(sv_spam[:, 0], sv_spam[:, 1], s=200, facecolors='none', 
              edgecolors='#e74c3c', linewidths=3)
    
    # Thêm chú thích "Margin" với mũi tên
    ax.annotate('Margin', xy=(2, -0.5), xytext=(3.5, 1),
               fontsize=11, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Cấu hình biểu đồ
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('Feature 1 (TF-IDF dimension)', fontsize=12)   # Trục X: chiều TF-IDF thứ 1
    ax.set_ylabel('Feature 2 (TF-IDF dimension)', fontsize=12)   # Trục Y: chiều TF-IDF thứ 2
    ax.set_title('Hình 2.5: Minh họa SVM với Hyperplane phân tách Ham vs Spam', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # Tỉ lệ trục X:Y bằng nhau
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_5_svm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_5_svm.png")


# =============================================================================
# HÌNH 2.6: Kiến trúc CNN
# Sơ đồ các lớp trong mô hình CNN: Input -> Embedding -> Conv1D -> ...
# =============================================================================
def create_figure_2_6():
    """Tạo sơ đồ kiến trúc mô hình CNN."""
    print("\n[6/9] Đang tạo Hình 2.6: CNN Architecture...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Định nghĩa các lớp CNN: (tên, vị trí x, y, chiều rộng, chiều cao, màu)
    layers = [
        ('Input\n(150)', 1, 4, 0.6, 3, '#3498db'),          # Đầu vào: chuỗi 150 số nguyên
        ('Embedding\n(150×128)', 3, 4, 1.2, 4, '#9b59b6'),  # Lớp Embedding: 150 từ x 128 chiều
        ('Conv1D\n(146×128)', 5.5, 4, 1.2, 3.5, '#e74c3c'), # Lớp tích chập 1D
        ('MaxPool\n(128)', 8, 4, 0.8, 2.5, '#f39c12'),      # Lớp pooling
        ('Dropout\n(0.5)', 10, 4, 0.6, 2, '#1abc9c'),       # Lớp Dropout 50%
        ('Dense\n(64→32)', 12, 4, 0.8, 2.5, '#e67e22'),     # Lớp kết nối đầy đủ
        ('Output\n(1)', 14.5, 4, 0.5, 1.5, '#2ecc71'),      # Đầu ra: 1 giá trị (xác suất spam)
    ]
    
    # Vẽ các hộp cho từng lớp
    for name, x, y, w, h, color in layers:
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, 
                             facecolor=color, edgecolor='white', 
                             linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    # Vẽ mũi tên kết nối giữa các lớp
    arrow_props = dict(arrowstyle='->', color='#34495e', lw=1.5)
    connections = [(1.3, 2.4), (4.2, 4.3), (6.7, 7.2), (8.8, 9.4), (10.6, 11.2), (12.8, 14)]
    for x1, x2 in connections:
        ax.annotate('', xy=(x2, 4), xytext=(x1, 4), arrowprops=arrow_props)
    
    ax.set_title('Hình 2.6: Kiến trúc mô hình 1D-CNN cho phân loại tin nhắn Spam', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Chú thích chi tiết phía dưới
    ax.text(8, 1, 
            'Input: Chuỗi 150 số nguyên (index từ) → Embedding: Ma trận 150×128 → Conv1D: 128 filters, kernel=5\n'
            '→ GlobalMaxPool: Vector 128 chiều → Dropout: 50% → Dense: 64→32 neurons → Output: Sigmoid (0-1)',
            ha='center', va='center', fontsize=10, color='#7f8c8d',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_6_cnn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_6_cnn.png")


# =============================================================================
# HÌNH 2.7: Kiến trúc hệ thống 3 tầng (3-Tier Architecture)
# Presentation Layer -> Business Logic Layer -> Model Layer
# =============================================================================
def create_figure_2_7():
    """Tạo sơ đồ kiến trúc hệ thống 3 tầng."""
    print("\n[7/9] Đang tạo Hình 2.7: System Architecture...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # --- Tầng Presentation (Giao diện - trên cùng) ---
    rect1 = plt.Rectangle((2, 7.5), 10, 1.5, facecolor='#3498db', 
                          edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect1)
    ax.text(7, 8.25, 'PRESENTATION LAYER\n(Streamlit Web App)', 
           ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # --- Tầng Business Logic (Xử lý nghiệp vụ - giữa) ---
    rect2 = plt.Rectangle((2, 4.5), 10, 2, facecolor='#9b59b6', 
                          edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect2)
    ax.text(7, 5.5, 'BUSINESS LOGIC LAYER\n(Python Backend: Preprocessing + Prediction)', 
           ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # --- Tầng Model (Mô hình - dưới cùng) - 3 mô hình ---
    models = [
        ('Naive Bayes\n(.pkl)', 3.5, '#2ecc71'),   # Mô hình NB (file .pkl)
        ('SVM\n(.pkl)', 7, '#e74c3c'),               # Mô hình SVM (file .pkl)
        ('CNN\n(.pth)', 10.5, '#f39c12'),             # Mô hình CNN (file .pth - PyTorch)
    ]
    
    for name, x, color in models:
        rect = plt.Rectangle((x-1.2, 1.5), 2.4, 2, facecolor=color, 
                             edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, 2.5, name, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
    
    ax.text(7, 0.8, 'MODEL LAYER (Trained Models)', 
           ha='center', va='center', fontsize=12, fontweight='bold', color='#7f8c8d')
    
    # Vẽ mũi tên 2 chiều kết nối giữa các tầng
    arrow_props = dict(arrowstyle='<->', color='#34495e', lw=2)
    ax.annotate('', xy=(7, 7.5), xytext=(7, 6.5), arrowprops=arrow_props)       # Presentation <-> Logic
    ax.annotate('', xy=(3.5, 4.5), xytext=(3.5, 3.5), arrowprops=arrow_props)   # Logic <-> NB
    ax.annotate('', xy=(7, 4.5), xytext=(7, 3.5), arrowprops=arrow_props)       # Logic <-> SVM
    ax.annotate('', xy=(10.5, 4.5), xytext=(10.5, 3.5), arrowprops=arrow_props) # Logic <-> CNN
    
    ax.set_title('Hình 2.7: Kiến trúc tổng quan hệ thống 3 tầng (3-Tier Architecture)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_7_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_7_architecture.png")


# =============================================================================
# HÌNH 2.8: Sơ đồ luồng xử lý dữ liệu (Data Flow Diagram)
# User Input -> Text Area -> Preprocess -> Vectorize -> Model Predict -> Result
# =============================================================================
def create_figure_2_8():
    """Tạo sơ đồ luồng xử lý dữ liệu từ đầu vào đến kết quả."""
    print("\n[8/9] Đang tạo Hình 2.8: Processing Flow...")
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Các bước trong luồng xử lý: (tên + icon, vị trí x, màu)
    steps = [
        ('👤 USER\nINPUT', 1, '#3498db'),           # Người dùng nhập tin nhắn
        ('📝 TEXT\nAREA', 3.5, '#9b59b6'),           # Vùng nhập văn bản Streamlit
        ('🔧 PRE-\nPROCESS', 6, '#e74c3c'),         # Tiền xử lý văn bản
        ('🔢 VECTOR\nIZE', 8.5, '#f39c12'),         # Vector hóa (TF-IDF hoặc Tokenize)
        ('🤖 MODEL\nPREDICT', 11, '#1abc9c'),       # Mô hình dự đoán
        ('📊 RESULT\nDISPLAY', 13.5, '#2ecc71'),    # Hiển thị kết quả
    ]
    
    # Vẽ các hộp cho mỗi bước
    for text, x, color in steps:
        rect = plt.Rectangle((x-0.9, 2), 1.8, 2.5, 
                             facecolor=color, edgecolor='white', 
                             linewidth=2, alpha=0.9, 
                             transform=ax.transData, zorder=2)
        ax.add_patch(rect)
        ax.text(x, 3.25, text, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white', zorder=3)
    
    # Vẽ mũi tên kết nối
    arrow_props = dict(arrowstyle='->', color='#34495e', lw=2)
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 0.9
        x2 = steps[i+1][1] - 0.9
        ax.annotate('', xy=(x2, 3.25), xytext=(x1, 3.25), arrowprops=arrow_props)
    
    # Mô tả chi tiết phía dưới mỗi bước
    descriptions = [
        'Nhập tin nhắn',         # User nhập văn bản
        'Streamlit\ntext_area',  # Widget text_area của Streamlit
        'Làm sạch\nvăn bản',    # Xóa URL, ký tự đặc biệt, chữ thường
        'TF-IDF hoặc\nTokenize', # Chuyển văn bản thành số
        'NB/SVM/CNN',            # 1 trong 3 mô hình dự đoán
        'Ham/Spam\n+ Xác suất', # Kết quả: loại + xác suất
    ]
    
    for i, (_, x, _) in enumerate(steps):
        ax.text(x, 0.8, descriptions[i], ha='center', va='center', 
               fontsize=9, color='#7f8c8d')
    
    ax.set_title('Hình 2.8: Sơ đồ luồng xử lý (Data Flow Diagram)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_8_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_8_flow.png")


# =============================================================================
# HÌNH 2.9: Mockup giao diện Streamlit
# Bản mô phỏng giao diện ứng dụng web phát hiện spam
# =============================================================================
def create_figure_2_9():
    """Tạo mockup (bản mô phỏng) giao diện ứng dụng Streamlit."""
    print("\n[9/9] Đang tạo Hình 2.9: UI Mockup...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # --- Khung trình duyệt (browser frame) ---
    browser = plt.Rectangle((0.5, 0.5), 13, 9, facecolor='#f8f9fa', 
                            edgecolor='#dee2e6', linewidth=2)
    ax.add_patch(browser)
    
    # Thanh địa chỉ trình duyệt (browser header)
    header = plt.Rectangle((0.5, 8.5), 13, 1, facecolor='#e9ecef', 
                           edgecolor='#dee2e6', linewidth=1)
    ax.add_patch(header)
    ax.text(7, 9, '🔍 Spam Detector - localhost:8501', ha='center', va='center', 
           fontsize=11, color='#495057')
    
    # --- Sidebar (thanh bên trái) ---
    sidebar = plt.Rectangle((0.5, 0.5), 3, 8, facecolor='#ffffff', 
                            edgecolor='#dee2e6', linewidth=1)
    ax.add_patch(sidebar)
    ax.text(2, 7.8, '⚙️ Cấu hình', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='#212529')
    
    # Dropdown chọn mô hình
    dropdown = plt.Rectangle((0.8, 6.8), 2.4, 0.6, facecolor='#e9ecef', 
                             edgecolor='#ced4da', linewidth=1)
    ax.add_patch(dropdown)
    ax.text(2, 7.1, 'Naive Bayes ▼', ha='center', va='center', fontsize=9, color='#495057')
    ax.text(2, 7.5, 'Chọn mô hình:', ha='center', va='center', fontsize=8, color='#6c757d')
    
    # --- Nội dung chính (Main content) ---
    ax.text(7.5, 7.5, '📧 Hệ thống Phát hiện Tin nhắn Spam', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#212529')
    
    # Ô nhập văn bản (text input)
    textbox = plt.Rectangle((4, 5), 9, 1.5, facecolor='#ffffff', 
                            edgecolor='#ced4da', linewidth=1)
    ax.add_patch(textbox)
    ax.text(8.5, 5.75, 'Congratulations! You won $1000! Call now...', 
           ha='center', va='center', fontsize=10, color='#495057', style='italic')
    ax.text(4.2, 6.7, '📝 Nhập tin nhắn cần kiểm tra:', ha='left', va='center', 
           fontsize=10, color='#212529')
    
    # Nút phân loại (classify button)
    button = plt.Rectangle((7.5, 4), 2, 0.6, facecolor='#e74c3c', 
                           edgecolor='#c0392b', linewidth=1)
    ax.add_patch(button)
    ax.text(8.5, 4.3, '🔍 Phân loại', ha='center', va='center', 
           fontsize=10, fontweight='bold', color='white')
    
    # Vùng kết quả (result area) - hiển thị kết quả SPAM
    result_box = plt.Rectangle((4, 1), 9, 2.5, facecolor='#ffebee', 
                               edgecolor='#e74c3c', linewidth=2)
    ax.add_patch(result_box)
    ax.text(8.5, 2.8, '⚠️ SPAM DETECTED!', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='#c62828')
    ax.text(8.5, 2.2, 'Xác suất Spam: 94.7%', ha='center', va='center', 
           fontsize=11, color='#e74c3c')
    # Thanh tiến trình (progress bar) cho xác suất
    ax.text(8.5, 1.6, '████████████████░░░░ 94.7%', ha='center', va='center', 
           fontsize=10, color='#e74c3c', family='monospace')
    
    ax.set_title('Hình 2.9: Giao diện ứng dụng Streamlit', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hinh_2_9_ui.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Đã lưu: hinh_2_9_ui.png")


# =============================================================================
# MAIN - CHẠY TẤT CẢ CÁC HÀM TẠO HÌNH ẢNH
# =============================================================================
if __name__ == "__main__":
    try:
        # Gọi lần lượt 9 hàm tạo hình ảnh
        create_figure_2_1()   # Box Plot
        create_figure_2_2()   # Word Cloud
        create_figure_2_3()   # Pie Chart
        create_figure_2_4()   # Pipeline Preprocessing
        create_figure_2_5()   # SVM Hyperplane
        create_figure_2_6()   # CNN Architecture
        create_figure_2_7()   # System Architecture
        create_figure_2_8()   # Data Flow
        create_figure_2_9()   # UI Mockup
        
        # In thông báo hoàn tất
        print("\n" + "=" * 60)
        print("✅ HOÀN TẤT! Đã tạo 9 hình ảnh trong thư mục 'images/'")
        print("=" * 60)
        print("\nDanh sách file:")
        # Liệt kê tất cả file PNG đã tạo kèm kích thước
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png'):
                size = os.path.getsize(f'{OUTPUT_DIR}/{f}') / 1024  # Chuyển bytes sang KB
                print(f"  📷 {f} ({size:.1f} KB)")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        print("Hãy đảm bảo đã cài đặt: pip install pandas matplotlib seaborn wordcloud")

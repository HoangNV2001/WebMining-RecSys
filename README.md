# Hệ thống Gợi ý Phim sử dụng các thuật toán Machine Learning

Dự án này xây dựng một hệ thống gợi ý phim (recommender system) sử dụng bộ dữ liệu MovieLens 1M. Thay vì các thuật toán gợi ý truyền thống (như Lọc cộng tác), dự án này tiếp cận bài toán dưới góc độ **phân loại học máy (Machine Learning Classification)**.

Mục tiêu là dự đoán xem một người dùng có "thích" một bộ phim hay không, và từ đó đưa ra các gợi ý phù hợp.

## 1. Bộ dữ liệu (Dataset)

Dự án sử dụng **MovieLens 1M**, một bộ dữ liệu phổ biến trong lĩnh vực hệ thống gợi ý, bao gồm:
- **1 triệu lượt đánh giá** từ 6,040 người dùng cho 3,952 bộ phim.
- **Thông tin nhân khẩu học** của người dùng (tuổi, giới tính, nghề nghiệp).
- **Thông tin về phim** (tiêu đề, thể loại).

Các tệp dữ liệu chính:
- `ratings.dat`: Chứa thông tin `userid`, `movieid`, `rating`, `timestamp`.
- `users.dat`: Chứa thông tin `userid`, `gender`, `age`, `occupation`.
- `movies.dat`: Chứa thông tin `movieid`, `title`, `genre`.

## 2. Chuẩn hóa và Chuẩn bị dữ liệu

Để phù hợp với bài toán phân loại, dữ liệu được xử lý qua các bước sau:

### a. Chuyển đổi bài toán thành Phân loại Nhị phân
- Cột `rating` (từ 1 đến 5) được chuyển thành một biến mục tiêu nhị phân là `liked`.
- Một lượt đánh giá được xem là **"thích" (`liked` = 1)** nếu `rating` >= 4.
- Ngược lại, được xem là **"không thích" (`liked` = 0)**.

### b. Phân chia dữ liệu theo Thời gian (Time-Based Split)
- Để mô phỏng kịch bản thực tế (dự đoán sở thích trong tương lai dựa trên dữ liệu quá khứ), dữ liệu được sắp xếp theo `timestamp`.
- **Tập huấn luyện (Train set)**: 80% dữ liệu có timestamp sớm nhất.
- **Tập kiểm thử (Test set)**: 20% dữ liệu có timestamp muộn nhất.
- Cách chia này giúp đánh giá mô hình một cách khách quan hơn so với chia ngẫu nhiên.

### c. Mã hóa Đặc trưng (Feature Encoding)
- Các đặc trưng dạng phân loại (categorical) như `userid` và `movieid` được mã hóa thành dạng số bằng `LabelEncoder`.
- **Quan trọng**: `LabelEncoder` chỉ được `fit` trên tập huấn luyện để tránh rò rỉ dữ liệu (data leakage) từ tập kiểm thử.

## 3. Kiến trúc Mô hình (Model Architecture)

Kiến trúc chung của hệ thống gợi ý dựa trên phân loại được mô tả như sau. Đây là một kiến trúc đơn giản nhưng hiệu quả để biến bài toán gợi ý thành bài toán học có giám sát.

```
┌──────────────────────────┐        ┌──────────────────────────┐
│      Đặc trưng đầu vào   │        │      Đầu ra Dự đoán      │
│  (Input Features)        │        │    (Prediction Output)   │
├──────────────────────────┤        ├──────────────────────────┤
│ - User ID (đã mã hóa)    │        │                          │
│ - Movie ID (đã mã hóa)   ├───────▶│      Mô hình Phân loại   ├───────▶  Xác suất "Thích"
│ - Tuổi (đã mã hóa)       │        │ (Naive Bayes, DT, RF)    │        │   (Probability)
│ - Giới tính (đã mã hóa)  │        │                          │        │
│ - Nghề nghiệp (đã mã hóa)│        └──────────────────────────┘        │   P(liked=1)
└──────────────────────────┘                                            │
                                                                        ▼
                                                              ┌──────────────────────────┐
                                                              │      Danh sách Gợi ý     │
                                                              │ (Top-N Recommendations)  │
                                                              └──────────────────────────┘
```

- **Đầu vào**: Một vector đặc trưng chứa thông tin đã được mã hóa của người dùng và phim.
- **Mô hình**: Một trong các thuật toán phân loại (Naive Bayes, Decision Tree, Random Forest) được huấn luyện để học mối quan hệ giữa các đặc trưng đầu vào và biến mục tiêu `liked`.
- **Đầu ra**: Mô hình dự đoán xác suất một cặp (user, movie) sẽ được "thích".
- **Gợi ý**: Các phim được xếp hạng dựa trên xác suất "thích" cao nhất và top-N phim đứng đầu sẽ được gợi ý cho người dùng.

## 4. Các Mô hình Machine Learning

Dự án triển khai 3 mô hình phân loại khác nhau:

### a. Naive Bayes (`naive_bayes_recommendation.ipynb`)
- **Cách hoạt động**: Mô hình này tính toán xác suất một người dùng sẽ "thích" một bộ phim dựa trên các đặc trưng đã cho (`userid`, `movieid`). Nó hoạt động dựa trên định lý Bayes và giả định rằng các đặc trưng là độc lập với nhau.
- **Ưu điểm**: Rất nhanh, hoạt động tốt với dữ liệu lớn và thưa thớt.
- **Kỹ thuật**: Sử dụng `CategoricalNB` với kỹ thuật làm mịn **Laplacian Smoothing** (tham số `alpha`) để xử lý các trường hợp chưa từng xuất hiện trong dữ liệu huấn luyện.

### b. Cây quyết định (Decision Tree) (`decision_tree_recommendation.ipynb`)
- **Cách hoạt động**: Mô hình xây dựng một cây gồm các quy tắc "if-then-else" dựa trên các đặc trưng (`userid`, `movieid`) để đưa ra quyết định phân loại (thích/không thích).
- **Ưu điểm**: Dễ diễn giải, có thể nắm bắt các mối quan hệ phi tuyến.
- **Kỹ thuật**: Độ sâu của cây (`max_depth`) được tinh chỉnh để cân bằng giữa độ chính xác và nguy cơ học vẹt (overfitting).

### c. Rừng ngẫu nhiên (Random Forest) (`random_forest_recommendation.ipynb`)
- **Cách hoạt động**: Đây là một mô hình tập hợp (ensemble) gồm nhiều Cây quyết định. Mỗi cây được huấn luyện trên một mẫu dữ liệu con. Kết quả cuối cùng được tổng hợp từ tất cả các cây (thường bằng cách lấy trung bình hoặc bỏ phiếu).
- **Ưu điểm**: Cho độ chính xác cao hơn và giảm thiểu overfitting so với một Cây quyết định đơn lẻ. Có khả năng đánh giá độ quan trọng của các đặc trưng.
- **Kỹ thuật**: Các siêu tham số như `n_estimators` (số lượng cây) và `max_depth` được tinh chỉnh để tối ưu hiệu suất.

## 5. Đánh giá Mô hình

Hiệu suất của các mô hình được đánh giá bằng các độ đo xếp hạng (ranking metrics) trên tập kiểm thử:

- **Precision@K**: Tỷ lệ các phim được gợi ý trong top-K mà người dùng thực sự thích.
  - `Precision@5`, `Precision@10`
- **NDCG@K (Normalized Discounted Cumulative Gain)**: Một độ đo phức tạp hơn, không chỉ quan tâm đến độ chính xác mà còn cả vị trí của các gợi ý đúng (gợi ý đúng ở vị trí càng cao thì điểm càng cao).
  - `NDCG@5`, `NDCG@10`

## 6. Cấu trúc Thư mục
```
.
├── README.md
├── data/
│   └── movielens-1m-dataset/
├── reqs.txt
└── src/
    ├── naive_bayes_recommendation.ipynb
    ├── decision_tree_recommendation.ipynb
    └── random_forest_recommendation.ipynb
```


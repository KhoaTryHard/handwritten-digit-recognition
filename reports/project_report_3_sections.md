# Báo cáo 3 mục cho repo nhận diện chữ số viết tay

## 1. Mô hình đề xuất

### 1.1. Kiến trúc được dùng ở model cuối

Artifact `models/stage_03_final.keras` hiện đang lưu một CNN nhỏ gọn cho bài toán phân lớp 10 chữ số. Nếu đếm theo chuỗi layer xử lý chính trong artifact, model cuối gồm 8 layer:

`Conv2D(32, kernel_size=3x3, activation=relu, padding=valid)` -> `MaxPooling2D(2x2)` -> `Conv2D(64, 3x3, relu, valid)` -> `MaxPooling2D(2x2)` -> `Conv2D(64, 3x3, relu, valid)` -> `Flatten` -> `Dense(64, relu)` -> `Dense(10, softmax)`

Bảng tóm tắt:

| Artifact | Kiến trúc | Regularization | Tổng tham số* |
| --- | --- | --- | ---: |
| `stage_03_final.keras` | 3 `Conv2D` + 2 `MaxPooling2D` + `Flatten` + 2 `Dense` | Không có `BatchNormalization`, không có `Dropout` | 93,322 |
| `stage_02_emnist_finetuned.keras` | Giống `stage_03_final.keras` | Không có `BatchNormalization`, không có `Dropout` | 93,322 |
| `stage_01_mnist_base.keras` | `InputLayer` + 5 `Conv2D` + 5 `BatchNormalization` + 5 `ReLU` + 2 `MaxPooling2D` + 3 `Dropout` + `GlobalAveragePooling2D` + 2 `Dense` | `Dropout(0.25, 0.30, 0.35)` và `BatchNormalization` | 134,954 |

\* Báo cáo này dùng `model.count_params()` để thống nhất cách đếm. `model.summary()` của file `.keras` còn hiển thị thêm optimizer state, nên tổng hiện thị trong summary lớn hơn con số bảng này.

### 1.2. Biến thể qua 3 stage

**Stage 1 - baseline MNIST (`stage_01_mnist_base.keras`)**

- Block 1: `Conv2D(32, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- Block 2: `Conv2D(32, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- `MaxPooling2D(2x2)` -> `Dropout(0.25)`
- Block 3: `Conv2D(64, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- Block 4: `Conv2D(64, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- `MaxPooling2D(2x2)` -> `Dropout(0.30)`
- Block 5: `Conv2D(96, 3x3, padding=same, use_bias=False)` -> `BatchNormalization` -> `ReLU`
- `GlobalAveragePooling2D` -> `Dense(128, relu)` -> `Dropout(0.35)` -> `Dense(10, softmax)`

**Stage 2 - fine-tune trên EMNIST (`stage_02_emnist_finetuned.keras`)**

- Kiến trúc artifact hiện có giống model cuối:
  `Conv2D(32, 3x3, relu)` -> `MaxPooling2D` -> `Conv2D(64, 3x3, relu)` -> `MaxPooling2D` -> `Conv2D(64, 3x3, relu)` -> `Flatten` -> `Dense(64, relu)` -> `Dense(10, softmax)`

**Stage 3 - model cuối (`stage_03_final.keras`)**

- Kiến trúc artifact giống stage 2, thông số 93,322.
- Đầu vào ảnh xám `28x28x1`, đầu ra `Dense(10, softmax)` cho 10 lớp `0-9`.

### 1.3. Lý do chọn kiến trúc và điểm cải tiến

- Baseline stage 1 được xây dựng theo hướng CNN có `BatchNormalization` và `Dropout` để học đặc trưng trên MNIST, giúp mô hình ổn định hơn và giảm overfitting trên bộ dữ liệu chuẩn.
- Stage 2 và stage 3 không cho thấy thay đổi kiến trúc trong artifact hiện lưu; cải tiến chính đến từ fine-tune theo miền dữ liệu, lần lượt trên `EMNIST digits` và bộ dữ liệu cá nhân `my_digits_28`.
- Nghĩa là, theo artifact thực tế trong repo, sự cải thiện ở model cuối không được chứng minh là do tăng độ phức tạp kiến trúc, mà chủ yếu do thích nghi trọng số với dữ liệu gần bài toán mục tiêu hơn.
- Cần lưu ý có độ lệch giữa artifact stage 2/3 và builder đang định nghĩa trong [`digit_pipeline/models/cnn.py`](../digit_pipeline/models/cnn.py): builder hiện tại mô tả mạng lớn hơn, gần với `stage_01_mnist_base.keras`, nhưng `stage_02_emnist_finetuned.keras` và `stage_03_final.keras` hiện đang là CNN 8 layer nhỏ hơn. Báo cáo này ưu tiên artifact thực tế đã lưu trong `models/`.

## 2. Cấu hình huấn luyện

### 2.1. Dataset và tiền xử lý

| Nguồn dữ liệu | Cách nạp | Tiền xử lý có trong repo |
| --- | --- | --- |
| MNIST | `tf.keras.datasets.mnist.load_data()` | Chuyển sang `float32`, chuẩn hóa về `[0, 1]`, nếu ảnh có rank 2 thì bổ sung channel để thành `28x28x1` |
| EMNIST digits | `tfds.load("emnist/digits", split=["train", "test"], as_supervised=True)` | Chuẩn hóa về `[0, 1]`, bổ sung channel nếu cần, augmentation mạnh hơn ở stage 2 |
| `my_digits_28/train`, `my_digits_28/val` | `tf.keras.utils.image_dataset_from_directory(..., color_mode="grayscale", image_size=(28, 28), label_mode="int")` | Ảnh xám `28x28`, chuẩn hóa về `[0, 1]`, giữ nhãn dạng số nguyên (`label_mode="int"`) |

Ghi chú về nhãn:

- Repo hiện tại **không one-hot hóa nhãn** trong pipeline huấn luyện.
- Nhãn được giữ dạng số nguyên và học với `SparseCategoricalCrossentropy`.

Ghi chú về dữ liệu cá nhân:

- Script tách dữ liệu gốc: `split_personal_data.py`
  - `train_dir = my_digits_new/train`
  - `val_dir = my_digits_new/val`
  - `val_ratio = 0.2`
  - `seed = 42`
- Script chuyển ảnh về dạng MNIST-like: `convert_personal_data_to_28.py`
  - `SOURCE_DIR = my_digits_new/val`
  - `DESTINATION_DIR = my_digits_28/val`
  - `THRESHOLD = 0.22`
- Trong `digit_pipeline/preprocessing/images.py`, ảnh viết tay được:
  - đưa về grayscale,
  - đảo cực nếu cần để đưa nét chữ số thành vùng sáng trên nền tối,
  - tạo mask bằng ngưỡng `threshold`,
  - dilate/erode,
  - giữ lại thành phần liên thông lớn,
  - cắt vùng chữ số,
  - resize vào khung `20x20`,
  - đặt vào canvas `28x28`,
  - căn giữa theo tâm khối lượng,
  - làm mượt nhẹ bằng `GaussianBlur(radius=0.4)`,
  - chuẩn hóa thành tensor `1x28x28x1`.

Quy mô dữ liệu cá nhân đang có trong repo:

| Tập | Số ảnh |
| --- | ---: |
| `my_digits_28/train` | 17,481 |
| `my_digits_28/val` | 4,369 |

### 2.2. Hyper-parameters theo từng stage

| Stage | Script | Đầu vào / Đầu ra | Dataset | Batch size | Epochs | Optimizer | Learning rate | Loss / Metric | Augmentation | Callback / Scheduler |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 1 | `train_mnist_base.py` | Tạo `models/stage_01_mnist_base.keras` | MNIST train / MNIST test | 64 | 5 | Adam | 1e-3 | `SparseCategoricalCrossentropy`, `accuracy` | Không có augmentation train riêng | Không cài đặt callback riêng |
| 2 | `finetune_emnist.py` | `stage_01_mnist_base.keras` -> `models/stage_02_emnist_finetuned.keras` | EMNIST digits train / test | 128 | 5 | Adam | 1e-4 | `SparseCategoricalCrossentropy`, `accuracy` | `RandomRotation(0.10)`, `RandomTranslation(0.15, 0.15)`, `RandomZoom(0.12)`, đảo màu ngẫu nhiên, speckle noise | `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` |
| 3 | `finetune_personal_data.py` | `stage_02_emnist_finetuned.keras` -> `models/stage_03_final.keras` | `my_digits_28/train`, `my_digits_28/val` | 64 | 40 | Adam | 1e-4 | `SparseCategoricalCrossentropy`, `accuracy` | `RandomRotation(0.08)`, `RandomTranslation(0.10, 0.10)`, `RandomZoom(0.08)` | `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` |

Chi tiết callback dùng chung cho stage 2 và stage 3:

| Callback | Cấu hình |
| --- | --- |
| `EarlyStopping` | monitor `val_loss`, `patience=5`, `restore_best_weights=True` |
| `ReduceLROnPlateau` | monitor `val_loss`, `factor=0.5`, `patience=2`, `min_lr=1e-6` |
| `ModelCheckpoint` | monitor `val_loss`, `save_best_only=True` |

### 2.3. Môi trường thư viện và GPU

| Thành phần | Giá trị trích từ repo |
| --- | --- |
| TensorFlow | `tensorflow==2.21.0` |
| TensorFlow Datasets | `tensorflow-datasets==4.9.9` |
| Keras | Không pin package `keras` riêng trong `requirements.txt`; repo dùng `tf.keras` |
| GPU / phần cứng | `(chưa có số liệu trong repo)` |
| Ghi chú runtime hiện tại | Log đánh giá read-only hiện tại báo TensorFlow native Windows không sử dụng GPU với TensorFlow >= 2.11 |

### 2.4. Lệnh huấn luyện và thư mục liên quan

```powershell
python train_mnist_base.py
python finetune_emnist.py
python finetune_personal_data.py
```

Thư mục / artifact tương ứng:

- `models/stage_01_mnist_base.keras`
- `models/stage_02_emnist_finetuned.keras`
- `models/stage_03_final.keras`
- `my_digits_28/train`
- `my_digits_28/val`

## 3. Kết quả mô hình cuối

### 3.1. Metric chính của `stage_03_final.keras`

Số liệu dưới đây được lấy từ lần đánh giá read-only hiện tại trên repo với artifact `models/stage_03_final.keras`.

| Tập đánh giá | Accuracy | Loss |
| --- | ---: | ---: |
| Train (`my_digits_28/train`) | 0.9603569508 | 0.1333358139 |
| Validation (`my_digits_28/val`) | 0.9567406774 | 0.1479387879 |
| Test | `(chưa có số liệu trong repo)` | `(chưa có số liệu trong repo)` |

### 3.2. Confusion matrix, per-class accuracy và artifact phân tích lỗi

Confusion matrix hiện tại trên `my_digits_28/val` (`rows=true`, `cols=pred`):

```text
[[440   0   1   3   1   3   0   0   3   0]
 [  1 433   1   1  11   1   0   4   2   0]
 [  0   0 443   8   0   0   0   1   3   0]
 [  1   0   1 433   0   3   0   1   6   1]
 [  2   1   2   0 424   0   3   2   5   3]
 [  1   0   1  11   0 411   0   0   6   0]
 [  4   0   3   2   4   5 409   0   4   0]
 [  1   1   1   1   1   1   0 420   2   1]
 [  2   0   5  12   2   9   0   0 388   4]
 [  4   0   1  10   3   4   0   4   4 379]]
```

Tổng số mẫu dự đoán sai: **189 / 4,369**.

Per-class accuracy:

| Lớp | Accuracy | Đúng / Tổng |
| --- | ---: | ---: |
| 0 | 0.9756 | 440 / 451 |
| 1 | 0.9537 | 433 / 454 |
| 2 | 0.9736 | 443 / 455 |
| 3 | 0.9709 | 433 / 446 |
| 4 | 0.9593 | 424 / 442 |
| 5 | 0.9558 | 411 / 430 |
| 6 | 0.9490 | 409 / 431 |
| 7 | 0.9790 | 420 / 429 |
| 8 | 0.9194 | 388 / 422 |
| 9 | 0.9267 | 379 / 409 |

Top confusion pairs:

| Hạng | Nhầm lẫn | Số lần |
| --- | --- | ---: |
| 1 | `8 -> 3` | 12 |
| 2 | `1 -> 4` | 11 |
| 3 | `5 -> 3` | 11 |
| 4 | `9 -> 3` | 10 |
| 5 | `8 -> 5` | 9 |
| 6 | `2 -> 3` | 8 |
| 7 | `3 -> 8` | 6 |
| 8 | `5 -> 8` | 6 |
| 9 | `4 -> 8` | 5 |
| 10 | `6 -> 5` | 5 |

Precision / recall theo lớp: `(chưa có số liệu trong repo)`. Script [`analyze_confusion_matrix.py`](../analyze_confusion_matrix.py) hiện chỉ in confusion matrix, per-class accuracy và top confusion.

Artifact bổ trợ để xem lỗi trực quan:

- CSV các mẫu sai: [misclassified_validation.csv](misclassified_validation.csv)
- Ảnh lỗi mẫu: [9_to_3_idx3914.png](confusion_pairs/9_to_3_idx3914.png), [9_to_3_idx4355.png](confusion_pairs/9_to_3_idx4355.png)

Lưu ý về độ lệch artifact:

- `reports/misclassified_validation.csv` hiện có 267 dòng sai, trong khi đánh giá hiện tại của `stage_03_final.keras` trên `my_digits_28/val` cho 189 mẫu sai.
- Thư mục `reports/confusion_pairs/` hiện đang có 36 ảnh `9 -> 3`, trong khi confusion matrix hiện tại cho thấy cặp `9 -> 3` xuất hiện 10 lần.
- Vì vậy, các tệp trong `reports/` ở trên chỉ nên dùng như artifact lịch sử / minh họa, **không** dùng làm số liệu chính cho model hiện tại.

### 3.3. Nhận xét

- Hai lớp đang yếu nhất trên tập validation hiện tại là `8` (0.9194) và `9` (0.9267).
- Cụm nhầm lẫn nổi bật xoay quanh `8/3/5` và `9/3`, cho thấy các mẫu có nét cong, vòng kín hoặc đuôi kéo dài có thể làm hình dạng giữa các lớp này trở nên gần nhau sau khi đưa về `28x28`.
- Cặp `1 -> 4` xuất hiện 11 lần, gợi ý một số mẫu `1` có nét nghiêng hoặc nét ngang/phụ trợ khi viết tay khiến mô hình nghiêng về `4`.
- Hướng cải thiện tiềm năng từ chính số liệu hiện có:
  - bổ sung thêm mẫu cá nhân cho các lớp `8`, `9`, `3`, `5`,
  - ưu tiên các biến thể viết tay khó phân biệt,
  - kiểm tra lại threshold / preprocessing cho nhóm mẫu bị mất nét sau khi chuyển về `28x28`,
  - có thể xuất thêm precision/recall theo lớp nếu muốn báo cáo chi tiết hơn ở phần đánh giá.

## Tài liệu tham khảo

- [`models/stage_01_mnist_base.keras`](../models/stage_01_mnist_base.keras)
- [`models/stage_02_emnist_finetuned.keras`](../models/stage_02_emnist_finetuned.keras)
- [`models/stage_03_final.keras`](../models/stage_03_final.keras)
- [`digit_pipeline/models/cnn.py`](../digit_pipeline/models/cnn.py)
- [`train_mnist_base.py`](../train_mnist_base.py)
- [`finetune_emnist.py`](../finetune_emnist.py)
- [`finetune_personal_data.py`](../finetune_personal_data.py)
- [`digit_pipeline/training/configs.py`](../digit_pipeline/training/configs.py)
- [`digit_pipeline/training/runners.py`](../digit_pipeline/training/runners.py)
- [`digit_pipeline/data_loading/datasets.py`](../digit_pipeline/data_loading/datasets.py)
- [`digit_pipeline/preprocessing/augmentations.py`](../digit_pipeline/preprocessing/augmentations.py)
- [`digit_pipeline/preprocessing/images.py`](../digit_pipeline/preprocessing/images.py)
- [`split_personal_data.py`](../split_personal_data.py)
- [`convert_personal_data_to_28.py`](../convert_personal_data_to_28.py)
- [`analyze_confusion_matrix.py`](../analyze_confusion_matrix.py)
- [`requirements.txt`](../requirements.txt)
- [`misclassified_validation.csv`](misclassified_validation.csv)
- [`confusion_pairs/9_to_3_idx3914.png`](confusion_pairs/9_to_3_idx3914.png)

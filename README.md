# Handwritten Digit Recognition

CNN-based handwritten digit recognition pipeline built with MNIST, EMNIST, and a personal 28x28 dataset.

## Overview

This project trains a digit classifier in three stages:

1. Train a base CNN on MNIST.
2. Fine-tune the base model on EMNIST digits.
3. Fine-tune again on a personal handwritten dataset converted to a unified 28x28 format.

The final model is used for prediction, evaluation, analysis scripts, and a local demo app in this repository.

## Training Pipeline

### Stage 1: Train on MNIST

Script:
- `python train_mnist_base.py`

Output:
- `models/stage_01_mnist_base.keras`

### Stage 2: Fine-tune on EMNIST

Script:
- `python finetune_emnist.py`

Output:
- `models/stage_02_emnist_finetuned.keras`

### Stage 3: Fine-tune on Personal Data

Script:
- `python finetune_personal_data.py`

Input dataset:
- `my_digits_28/train`
- `my_digits_28/val`

Output:
- `models/stage_03_final.keras`

## Repository Structure

```text
.
|- digit_pipeline/              Shared preprocessing, training, data, and evaluation helpers
|- models/                      Canonical saved model artifacts for the 3-stage workflow
|- my_digits_new/               Raw personal handwritten dataset
|- my_digits_28/                Converted 28x28 personal dataset
|- reports/                     Analysis exports and reports
|- train_mnist_base.py          Stage 1 training
|- finetune_emnist.py           Stage 2 training
|- finetune_personal_data.py    Stage 3 training
|- split_personal_data.py       Split raw personal data into train/val
|- convert_personal_data_to_28.py
|- predict_digit_app.py         Local demo app with image picker
|- predict_digit_image.py       Predict a single image
|- analyze_confusion_matrix.py
|- export_misclassified_csv.py
|- export_confusion_pair_images.py
|- preview_mnist_samples.py
|- requirements.txt
`- README.md
```

## File Guide / Mo Ta Tep

Quick reference for the main top-level files in this repository. Mo ta nhanh cong dung cua cac tep chinh o thu muc goc cua du an.

- `project_paths.py`: Provides small helpers for building paths relative to the project root. Tep nay cung cap cac ham helper de tao duong dan tinh tu thu muc goc cua project.
- `analyze_confusion_matrix.py`: Runs the final model on the validation set, prints the confusion matrix, reports per-class accuracy, and lists the most frequent confusion pairs. Tep nay chay model cuoi tren tap validation, in ma tran nham lan, do chinh xac theo tung lop, va cac cap du doan nham xuat hien nhieu nhat.
- `convert_personal_data_to_28.py`: Converts personal handwritten images into an MNIST-like 28x28 validation dataset using the configured preprocessing threshold. Tep nay chuyen anh chu so viet tay ca nhan thanh du lieu validation 28x28 giong MNIST voi nguong tien xu ly da cau hinh.
- `export_confusion_pair_images.py`: Exports image files for one chosen true-to-predicted confusion pair so the mistakes can be inspected visually. Tep nay xuat anh cua mot cap nham lan cu the de co the xem truc quan cac mau du doan sai.
- `export_misclassified_csv.py`: Exports all misclassified validation samples to a CSV file with file paths, labels, probabilities, and top-3 predictions. Tep nay xuat toan bo mau du doan sai trong validation ra CSV, kem duong dan file, nhan, xac suat, va top-3 du doan.
- `finetune_emnist.py`: Fine-tunes the stage 1 MNIST model on the EMNIST digits dataset with augmentation and saves `models/stage_02_emnist_finetuned.keras`. Tep nay fine-tune model stage 1 tren bo EMNIST digits co augmentation va luu thanh `models/stage_02_emnist_finetuned.keras`.
- `finetune_personal_data.py`: Fine-tunes the stage 2 model on `my_digits_28/train` and `my_digits_28/val` to produce the final personal-style model. Tep nay fine-tune tiep model stage 2 tren `my_digits_28/train` va `my_digits_28/val` de tao ra model cuoi phu hop voi du lieu ca nhan.
- `predict_digit_app.py`: Launches a local desktop demo app where you choose an image, view the processed preview, and see the prediction results visually. Tep nay mo mot app desktop local de ban chon anh, xem anh sau tien xu ly, va xem ket qua du doan mot cach truc quan.
- `predict_digit_image.py`: Runs the same inference pipeline from a script, either from the configured `IMAGE_PATH` variable or from a CLI path. Tep nay chay cung pipeline suy luan duoi dang script, ho tro ca bien `IMAGE_PATH` da cau hinh san va duong dan truyen qua CLI.
- `preview_mnist_samples.py`: Displays sample MNIST images for one selected digit to support quick visual comparison. Tep nay hien thi cac anh mau MNIST cua mot chu so duoc chon de phuc vu so sanh nhanh bang mat.
- `project_report_sections.md`: Stores draft report sections that describe the research method and the overall system-building process. Tep nay luu cac doan nhap cho bao cao, mo ta phuong phap nghien cuu va quy trinh xay dung he thong.
- `README.md`: Serves as the main project document with the overview, setup steps, workflow, and usage commands. Tep nay la tai lieu tong quan cua du an, gom gioi thieu, cai dat, quy trinh, va cach su dung.
- `requirements.txt`: Lists the Python dependencies needed for training, prediction, visualization, and analysis scripts. Tep nay liet ke cac thu vien Python can cho train model, du doan, truc quan hoa, va phan tich.
- `split_personal_data.py`: Moves part of the raw personal dataset from `my_digits_new/train` into `my_digits_new/val` to create a validation split. Tep nay chuyen mot phan du lieu ca nhan goc tu `my_digits_new/train` sang `my_digits_new/val` de tao tap validation.
- `train_mnist_base.py`: Trains the base CNN on MNIST from scratch and saves `models/stage_01_mnist_base.keras`. Tep nay huan luyen CNN nen tren MNIST tu dau va luu thanh `models/stage_01_mnist_base.keras`.

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Main dependencies:
- TensorFlow
- TensorFlow Datasets
- NumPy
- Pillow
- Pandas
- Matplotlib

## Data Preparation

### Split Raw Personal Data

Use this script to split raw handwritten images into training and validation sets:

```powershell
python split_personal_data.py
```

### Convert Personal Images to 28x28

Convert personal images into a MNIST-like 28x28 dataset:

```powershell
python convert_personal_data_to_28.py
```

## Usage

### Train the Full Pipeline

Run the scripts in this order:

```powershell
python train_mnist_base.py
python finetune_emnist.py
python split_personal_data.py
python convert_personal_data_to_28.py
python finetune_personal_data.py
```

### Local Demo App (Recommended)

```powershell
python predict_digit_app.py
```

Quick demo flow:

1. Activate the virtual environment and make sure `models/stage_03_final.keras` exists.
2. Run `python predict_digit_app.py`.
3. Click `Choose Image`, select an input file, and wait for the app to show the original image, processed 28x28 preview, predicted digit, confidence, and top probabilities.

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`.

### Predict a Single Image

If you prefer a script instead of the GUI app, run:

```powershell
python predict_digit_image.py
```

Set `IMAGE_PATH` inside `predict_digit_image.py` before running, or pass an image path through the CLI:

```powershell
python predict_digit_image.py path/to/image.png
```

### Analysis and Visualization

```powershell
python analyze_confusion_matrix.py
python export_misclassified_csv.py
python export_confusion_pair_images.py
python preview_mnist_samples.py
```

## Model Artifacts

- `models/stage_01_mnist_base.keras`: base model trained on MNIST
- `models/stage_02_emnist_finetuned.keras`: model adapted to EMNIST digits
- `models/stage_03_final.keras`: final model used for prediction and evaluation

The repository intentionally keeps only the three canonical stage outputs above. Extra experimental `.keras` files inside `models/` are ignored by `models/.gitignore` to avoid confusion with the main workflow.

## Important Configuration

Review these files before running experiments:

- `predict_digit_app.py`: `MODEL_PATH`, `PREPROCESS_THRESHOLD`, `TTA_SAMPLES`
- `predict_digit_image.py`: `MODEL_PATH`, `IMAGE_PATH`, `PREPROCESS_THRESHOLD`, `TTA_SAMPLES`
- `convert_personal_data_to_28.py`: `SOURCE_DIR`, `DESTINATION_DIR`, `THRESHOLD`
- `split_personal_data.py`: `TRAIN_DIR`, `VAL_DIR`, `VAL_RATIO`, `SEED`
- `finetune_personal_data.py`: dataset paths, batch size, epochs, learning rate

## Historical Script Mapping

| Old name | Current file |
| --- | --- |
| `Day2.py` | `train_mnist_base.py` |
| `finetune_emnist_digits.py` | `finetune_emnist.py` |
| `split_train_val.py` | `split_personal_data.py` |
| `convert_to_28.py` | `convert_personal_data_to_28.py` |
| `finetune_mystyle.py` | `finetune_personal_data.py` |
| `trainAIDigit.py` | `predict_digit_image.py` |
| `confusion_matrix.py` | `analyze_confusion_matrix.py` |
| `save_csv.py` | `export_misclassified_csv.py` |
| `xuatAnhCapSai.py` | `export_confusion_pair_images.py` |
| `xemAnhDaTrain.py` | `preview_mnist_samples.py` |

## Notes

- `my_digits_28` is the canonical personal 28x28 dataset used in the current workflow.
- `my_digits_28_new` has already been merged into `my_digits_28`.
- `models/.gitignore` ignores non-canonical model exports so only the three official stage artifacts stay tracked.
- Temporary folders such as `.venv/`, `.idea/`, and `__pycache__/` should not be committed.

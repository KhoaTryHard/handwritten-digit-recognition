import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from digit_pipeline.inference import load_digit_model, predict_digit_from_image
from project_paths import project_path


MODEL_PATH = project_path("models", "stage_03_final.keras")
PREPROCESS_THRESHOLD = 0.18
TTA_SAMPLES = 30
IMAGE_FILE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
    ("All files", "*.*"),
]
PREVIEW_SIZE = (320, 320)


class DigitPredictionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Demo")
        self.root.geometry("980x680")
        self.root.minsize(900, 600)

        self.model = None
        self.current_image_path = ""
        self.original_photo: ImageTk.PhotoImage | None = None
        self.processed_photo: ImageTk.PhotoImage | None = None

        self.path_var = tk.StringVar(value="No image selected.")
        self.prediction_var = tk.StringVar(value="-")
        self.confidence_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="Loading model...")

        self.top_labels: list[ttk.Label] = []
        self.choose_button: ttk.Button
        self.retry_button: ttk.Button
        self.original_panel: ttk.Label
        self.processed_panel: ttk.Label

        self._build_ui()
        self.root.after(100, self._load_model)

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Handwritten Digit Prediction Demo",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Choose an image, run the trained model, and compare the original input with the processed 28x28 preview.",
            wraplength=760,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        actions = ttk.Frame(container)
        actions.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        actions.columnconfigure(0, weight=1)
        actions.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(actions)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(2, weight=1)

        self.choose_button = ttk.Button(
            toolbar,
            text="Choose Image",
            command=self.choose_image,
        )
        self.choose_button.grid(row=0, column=0, sticky="w")

        self.retry_button = ttk.Button(
            toolbar,
            text="Predict Again",
            command=self.predict_current_image,
            state="disabled",
        )
        self.retry_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(
            toolbar,
            textvariable=self.path_var,
            wraplength=520,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        images_frame = ttk.Frame(actions)
        images_frame.grid(row=2, column=0, sticky="nsew")
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)

        original_group = ttk.LabelFrame(images_frame, text="Original Image", padding=12)
        original_group.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        original_group.columnconfigure(0, weight=1)
        original_group.rowconfigure(0, weight=1)

        self.original_panel = ttk.Label(
            original_group,
            text="Choose an image to preview it here.",
            anchor="center",
            justify="center",
        )
        self.original_panel.grid(row=0, column=0, sticky="nsew")

        processed_group = ttk.LabelFrame(
            images_frame,
            text="Processed 28x28 Preview",
            padding=12,
        )
        processed_group.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        processed_group.columnconfigure(0, weight=1)
        processed_group.rowconfigure(0, weight=1)

        self.processed_panel = ttk.Label(
            processed_group,
            text="The preprocessing result will appear here.",
            anchor="center",
            justify="center",
        )
        self.processed_panel.grid(row=0, column=0, sticky="nsew")

        result_card = ttk.LabelFrame(container, text="Prediction", padding=16)
        result_card.grid(row=1, column=1, sticky="nsew")
        result_card.columnconfigure(0, weight=1)

        ttk.Label(result_card, text="Predicted Digit", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            result_card,
            textvariable=self.prediction_var,
            font=("Segoe UI", 40, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        ttk.Label(result_card, text="Confidence", font=("Segoe UI", 11, "bold")).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(
            result_card,
            textvariable=self.confidence_var,
            font=("Segoe UI", 16),
        ).grid(row=3, column=0, sticky="w", pady=(4, 16))

        ttk.Label(result_card, text="Top Predictions", font=("Segoe UI", 11, "bold")).grid(
            row=4, column=0, sticky="w"
        )

        top_frame = ttk.Frame(result_card)
        top_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        top_frame.columnconfigure(0, weight=1)

        for index in range(5):
            label = ttk.Label(top_frame, text=f"{index + 1}. -", font=("Consolas", 11))
            label.grid(row=index, column=0, sticky="w", pady=2)
            self.top_labels.append(label)

        status_bar = ttk.Label(
            container,
            textvariable=self.status_var,
            anchor="w",
        )
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))

    def _load_model(self) -> None:
        try:
            self.model = load_digit_model(MODEL_PATH)
        except Exception as exc:
            self.status_var.set("Failed to load model.")
            self.choose_button.configure(state="disabled")
            messagebox.showerror(
                "Model Load Error",
                f"Could not load model from:\n{MODEL_PATH}\n\n{exc}",
            )
            return

        self.status_var.set("Model loaded. Choose an image to start the demo.")

    def choose_image(self) -> None:
        image_path = filedialog.askopenfilename(
            title="Choose an image for prediction",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not image_path:
            return

        self.current_image_path = image_path
        self.path_var.set(image_path)
        self.retry_button.configure(state="normal")
        self.predict_current_image()

    def predict_current_image(self) -> None:
        if not self.current_image_path:
            messagebox.showinfo("No Image", "Choose an image first.")
            return

        if self.model is None:
            messagebox.showinfo("Model Loading", "The model is still loading. Please wait.")
            return

        self.status_var.set("Running prediction...")
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        try:
            result = predict_digit_from_image(
                self.current_image_path,
                self.model,
                preprocess_threshold=PREPROCESS_THRESHOLD,
                tta_samples=TTA_SAMPLES,
            )
        except Exception as exc:
            self.status_var.set("Prediction failed.")
            messagebox.showerror(
                "Prediction Error",
                f"Could not run prediction for:\n{self.current_image_path}\n\n{exc}",
            )
            return
        finally:
            self.root.config(cursor="")

        self._set_original_preview(self.current_image_path)
        self._set_processed_preview(result.preview)
        self.prediction_var.set(str(result.prediction))
        self.confidence_var.set(f"{result.confidence * 100:.2f}%")

        for rank, digit in enumerate(result.top_indices, start=1):
            probability = float(result.probabilities[digit])
            self.top_labels[rank - 1].configure(
                text=f"{rank}. digit {int(digit)}  {probability * 100:6.2f}%"
            )

        self.status_var.set("Prediction complete.")

    def _set_original_preview(self, image_path: str) -> None:
        with Image.open(image_path) as image:
            display = image.convert("RGB")
        display.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(display)
        self.original_panel.configure(image=self.original_photo, text="")

    def _set_processed_preview(self, image: Image.Image) -> None:
        display = image.resize(PREVIEW_SIZE, Image.Resampling.NEAREST).convert("L")
        self.processed_photo = ImageTk.PhotoImage(display)
        self.processed_panel.configure(image=self.processed_photo, text="")


def main() -> None:
    root = tk.Tk()
    DigitPredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

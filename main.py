import argparse
from datetime import datetime, timezone
import json
import requests
import os
import sys
import subprocess
import gzip
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Train the MNIST convolutional network.")
parser.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "artifacts", "mnist_best.keras"))
parser.add_argument("--epochs", type=int, default=40, help="Additional epochs for this run.")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("--translation", type=float, default=0.05)
parser.add_argument("--rotation", type=float, default=0.04)
parser.add_argument("--zoom", type=float, default=0.05)
parser.add_argument("--conv-dropout-1", type=float, default=0.15)
parser.add_argument("--conv-dropout-2", type=float, default=0.2)
parser.add_argument("--dense-dropout", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=884736743)
parser.add_argument(
    "--initial-weights",
    help="Initialize from a compatible .weights.h5 file or saved .keras model (takes precedence over resume).",
)
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Start from random initialization instead of resuming the saved best model.",
)
parser.add_argument(
    "--evaluate",
    "--evaluate-only",
    action="store_true",
    help="Evaluate the saved checkpoint on all MNIST test images without training.",
)
parser.add_argument(
    "--eval-dir",
    default=os.path.join(SCRIPT_DIR, "eval"),
    help="Base directory containing previous, current, and comparison subfolders.",
)
args = parser.parse_args()

if args.evaluate:
    evaluator_path = os.path.join(SCRIPT_DIR, "evaluate.py")
    subprocess.run(
        [
            sys.executable,
            evaluator_path,
            "--checkpoint",
            args.checkpoint,
            "--output-dir",
            args.eval_dir,
        ],
        check=True,
    )
    raise SystemExit(0)

if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
    parser.error("--epochs and --batch-size must be positive, and --learning-rate must be greater than zero")
for name in ("translation", "rotation", "zoom", "conv_dropout_1", "conv_dropout_2", "dense_dropout"):
    if not 0 <= getattr(args, name) < 1:
        parser.error(f"--{name.replace('_', '-')} must be in the range [0, 1)")

metadata_path = os.path.splitext(args.checkpoint)[0] + ".json"
os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}

data_dir = os.path.join(SCRIPT_DIR, "_data")
os.makedirs(data_dir, exist_ok=True)

base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True)
        resp.raise_for_status()  # Ensure download was succesful
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

mnist_dataset = {}

for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)

for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)

seed = args.seed
rng = np.random.default_rng(seed)
tf.keras.utils.set_random_seed(seed)

permutation = rng.permutation(len(x_train))
validation_size = 5000
training_indices = permutation[validation_size:]
validation_indices = permutation[:validation_size]

training_images = (
    x_train[training_indices].astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
)
training_labels = y_train[training_indices].astype(np.int64)
validation_images = (
    x_train[validation_indices].astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
)
validation_labels = y_train[validation_indices].astype(np.int64)
test_images = x_test.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
test_labels = y_test.astype(np.int64)

LR_REDUCTION_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 6

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomTranslation(
            args.translation, args.translation, fill_mode="constant"
        ),
        tf.keras.layers.RandomRotation(args.rotation, fill_mode="constant"),
        tf.keras.layers.RandomZoom(args.zoom, fill_mode="constant"),
    ],
    name="training_augmentation",
)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28, 1)),
        augmentation,
        tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(args.conv_dropout_1),
        tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(args.conv_dropout_2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(args.dense_dropout),
        tf.keras.layers.Dense(10, activation="softmax"),
    ],
    name="mnist_cnn",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

best_accuracy_before_run = 0.0
best_loss_before_run = float("inf")
previous_best_epoch = 0
last_completed_epoch = 0
previous_validation_accuracy = None
has_previous_metadata = False
if os.path.isfile(args.checkpoint) and os.path.exists(metadata_path):
    with open(metadata_path, encoding="utf-8") as metadata_file:
        previous_metadata = json.load(metadata_file)
    has_previous_metadata = True
    best_accuracy_before_run = float(previous_metadata.get("best_validation_accuracy", 0.0))
    best_loss_before_run = float(previous_metadata.get("best_validation_loss", float("inf")))
    previous_best_epoch = int(previous_metadata.get("best_epoch", 0))
    last_completed_epoch = int(
        previous_metadata.get("last_epoch", previous_best_epoch)
    )
    if previous_metadata.get("last_validation_accuracy") is not None:
        previous_validation_accuracy = float(previous_metadata["last_validation_accuracy"])

if args.initial_weights:
    if not os.path.isfile(args.initial_weights):
        parser.error(f"Initial weights file does not exist: {args.initial_weights}")
    if args.initial_weights.endswith(".keras"):
        initial_model = tf.keras.models.load_model(args.initial_weights, compile=False)
        if [tuple(weight.shape) for weight in initial_model.get_weights()] != [
            tuple(weight.shape) for weight in model.get_weights()
        ]:
            parser.error("Initial .keras model architecture does not match this training model.")
        model.set_weights(initial_model.get_weights())
    else:
        model.load_weights(args.initial_weights)
    print(f"Initialized model weights from {args.initial_weights}")
elif not args.fresh and os.path.isfile(args.checkpoint):
    model = tf.keras.models.load_model(args.checkpoint, compile=True)
    print(f"Resuming from saved best checkpoint: {args.checkpoint}")
elif not args.fresh:
    print("No saved checkpoint found; starting with new model weights.")

initial_epoch = (
    last_completed_epoch
    if os.path.exists(args.checkpoint) and not args.fresh and not args.initial_weights
    else 0
)
if initial_epoch:
    print(f"Continuing epoch numbering from epoch {initial_epoch + 1}.")

model.summary()

class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        self.loss = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        loss, accuracy = self.model.evaluate(
            self.images, self.labels, batch_size=512, verbose=0
        )
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        print(f"  Training (inference mode): loss={loss:.4f}, accuracy={accuracy:.4f}")

training_metrics = TrainingMetricsCallback(training_images, training_labels)


class ValidationAccuracyLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, previous_accuracy, best_accuracy):
        super().__init__()
        self.previous_accuracy = previous_accuracy
        self.best_accuracy = best_accuracy
        self.verbose = 2

    def set_params(self, params):
        super().set_params(params)
        self.verbose = 2

    @staticmethod
    def format_change(current_accuracy, reference_accuracy):
        change = (current_accuracy - reference_accuracy) * 100
        if change > 0:
            color, arrow = "\033[92m", "🠝"
        elif change < 0:
            color, arrow = "\033[91m", "🠟"
        else:
            return "→ 0.00%"
        return f"{color}{arrow} {change:+.2f}%\033[0m"

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        accuracy = float((logs or {})["val_accuracy"])
        last_change = (
            self.format_change(accuracy, self.previous_accuracy)
            if self.previous_accuracy is not None
            else "—"
        )
        best_change = (
            self.format_change(accuracy, self.best_accuracy)
            if self.best_accuracy is not None
            else "—"
        )
        print(
            f"{last_change} (Validation Accuracy) (Compared to last Epoch) | "
            f"{best_change} (Validation Accuracy) (Compared to all-time best)",
            flush=True,
        )
        self.previous_accuracy = accuracy
        if self.best_accuracy is None or accuracy > self.best_accuracy:
            self.best_accuracy = accuracy


class BestCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_path, initial_accuracy, initial_loss):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.best_accuracy = initial_accuracy
        self.best_loss = initial_loss
        self.best_metadata = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = float(logs["val_accuracy"])
        loss = float(logs["val_loss"])
        if accuracy < self.best_accuracy or (
            accuracy == self.best_accuracy and loss >= self.best_loss
        ):
            return

        self.model.save(self.checkpoint_path)
        self.best_accuracy = accuracy
        self.best_loss = loss
        self.best_metadata = {
            "best_epoch": epoch + 1,
            "best_validation_accuracy": accuracy,
            "best_validation_loss": loss,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "tensorflow_version": tf.__version__,
            "training_examples": int(len(training_images)),
            "validation_examples": int(len(validation_images)),
            "seed": seed,
            "validation_split_seed": seed,
            "architecture": {
                "type": "convolutional_neural_network",
                "convolution_filters": [32, 32, 64, 64],
                "dense_units": 128,
            },
            "learning_rate_at_best_epoch": float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            ),
            "run_parameters": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs_requested": args.epochs,
                "lr_reduction_patience": LR_REDUCTION_PATIENCE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "translation": args.translation,
                "rotation": args.rotation,
                "zoom": args.zoom,
                "conv_dropout_1": args.conv_dropout_1,
                "conv_dropout_2": args.conv_dropout_2,
                "dense_dropout": args.dense_dropout,
            },
            "checkpoint": os.path.basename(args.checkpoint),
        }
        print(f"Candidate checkpoint improved to {accuracy:.4%} validation accuracy.")

candidate_checkpoint = f"{args.checkpoint}.{os.getpid()}.candidate.keras"
checkpoint_callback = BestCheckpointCallback(
    candidate_checkpoint,
    best_accuracy_before_run,
    best_loss_before_run,
)
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=LR_REDUCTION_PATIENCE,
        min_lr=1e-5,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    training_images,
    training_labels,
    validation_data=(validation_images, validation_labels),
    batch_size=args.batch_size,
    initial_epoch=initial_epoch,
    epochs=initial_epoch + args.epochs,
    shuffle=True,
    callbacks=[
        training_metrics,
        checkpoint_callback,
        *callbacks,
        ValidationAccuracyLogger(
            previous_validation_accuracy,
            best_accuracy_before_run if has_previous_metadata else None,
        ),
    ],
    verbose=0,
)

candidate_is_better = checkpoint_callback.best_metadata is not None
if candidate_is_better:
    os.replace(candidate_checkpoint, args.checkpoint)
    final_metadata = checkpoint_callback.best_metadata
    print(f"Run completed; promoted the improved checkpoint to {args.checkpoint}.")
elif os.path.isfile(candidate_checkpoint):
    os.remove(candidate_checkpoint)
    print("Run completed; existing best checkpoint was kept.")
elif os.path.isfile(args.checkpoint):
    print("Run completed; existing best checkpoint was kept.")

if not os.path.isfile(args.checkpoint):
    raise RuntimeError(
        f"Training finished without producing a checkpoint at {args.checkpoint}. "
        "Check validation metrics and write permissions."
    )

model = tf.keras.models.load_model(args.checkpoint, compile=True)
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4%}")
if candidate_is_better:
    pass
elif os.path.isfile(metadata_path):
    with open(metadata_path, encoding="utf-8") as metadata_file:
        final_metadata = json.load(metadata_file)
else:
    final_metadata = {
        "checkpoint": os.path.basename(args.checkpoint),
        "best_validation_accuracy": best_accuracy_before_run,
        "best_validation_loss": best_loss_before_run,
        "best_epoch": previous_best_epoch,
    }
final_metadata["checkpoint"] = os.path.basename(args.checkpoint)
final_metadata["test_loss"] = float(test_loss)
final_metadata["test_accuracy"] = float(test_accuracy)
final_metadata["last_validation_accuracy"] = float(
    history.history["val_accuracy"][-1]
)
final_metadata["last_epoch"] = initial_epoch + len(history.history["val_accuracy"])
temporary_metadata = metadata_path + ".tmp"
with open(temporary_metadata, "w", encoding="utf-8") as metadata_file:
    json.dump(final_metadata, metadata_file, indent=2)
os.replace(temporary_metadata, metadata_path)

epoch_range = np.arange(initial_epoch + 1, initial_epoch + len(history.history["loss"]) + 1)
training_error = 1 - np.asarray(training_metrics.accuracy)
validation_error = 1 - np.asarray(history.history["val_accuracy"])
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
plots = (
    (
        axes[0],
        training_metrics.accuracy,
        history.history["val_accuracy"],
        "Accuracy",
    ),
    (axes[1], training_error, validation_error, "Classification error rate"),
    (
        axes[2],
        training_metrics.loss,
        history.history["val_loss"],
        "Cross-entropy loss",
    ),
)
for ax, training_values, validation_values, title in plots:
    ax.plot(epoch_range, training_values, label="Training")
    ax.plot(epoch_range, validation_values, label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()

fig.tight_layout()
plt.show()
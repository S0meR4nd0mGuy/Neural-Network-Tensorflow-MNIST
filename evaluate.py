"""Evaluate a saved MNIST model and write detailed, comparable reports."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "_data")
TEST_FILES = {
    "images": "t10k-images-idx3-ubyte.gz",
    "labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DIGITS = np.arange(10)
CALIBRATION_BINS = 15


def load_test_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename in TEST_FILES.values():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            print(f"Downloading MNIST test data: {filename}", flush=True)
            response = requests.get(BASE_URL + filename, stream=True, timeout=60)
            response.raise_for_status()
            with open(path, "wb") as data_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        data_file.write(chunk)

    with gzip.open(os.path.join(DATA_DIR, TEST_FILES["images"]), "rb") as images_file:
        images = np.frombuffer(images_file.read(), dtype=np.uint8, offset=16)
    with gzip.open(os.path.join(DATA_DIR, TEST_FILES["labels"]), "rb") as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=8)
    if images.size != 10_000 * 28 * 28 or labels.size != 10_000:
        raise ValueError("MNIST test files must contain exactly 10,000 28x28 images and labels.")
    images = images.reshape(10_000, 28, 28, 1).astype(np.float32) / 255.0
    return images, labels.astype(np.int64)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_metrics(labels, probabilities):
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correct = predictions == labels
    count = len(labels)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    true_probabilities = probabilities[np.arange(count), labels]
    cross_entropy = float(-np.log(np.clip(true_probabilities, 1e-7, 1.0)).mean())
    one_hot = np.eye(10, dtype=np.float64)[labels]
    brier_score = float(np.square(probabilities - one_hot).sum(axis=1).mean())

    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positives = np.diag(confusion)
    precision = np.divide(
        true_positives,
        predicted_support,
        out=np.zeros(10, dtype=np.float64),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positives,
        true_support,
        out=np.zeros(10, dtype=np.float64),
        where=true_support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10, dtype=np.float64),
        where=(precision + recall) != 0,
    )

    bin_edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    calibration = []
    expected_calibration_error = 0.0
    for bin_index in range(CALIBRATION_BINS):
        lower, upper = float(bin_edges[bin_index]), float(bin_edges[bin_index + 1])
        in_bin = (confidences >= lower) & (
            confidences <= upper if bin_index == CALIBRATION_BINS - 1 else confidences < upper
        )
        bin_count = int(in_bin.sum())
        mean_confidence = float(confidences[in_bin].mean()) if bin_count else None
        accuracy = float(correct[in_bin].mean()) if bin_count else None
        if bin_count:
            expected_calibration_error += (
                bin_count / count * abs(accuracy - mean_confidence)
            )
        calibration.append(
            {
                "lower": lower,
                "upper": upper,
                "count": bin_count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    top_k = min(3, probabilities.shape[1])
    top_k_predictions = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    return {
        "count": count,
        "accuracy": float(correct.mean()),
        "error_count": int((~correct).sum()),
        "cross_entropy": cross_entropy,
        "top_3_accuracy": float(np.any(top_k_predictions == labels[:, None], axis=1).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=true_support)),
        "brier_score": brier_score,
        "expected_calibration_error": float(expected_calibration_error),
        "mean_confidence": float(confidences.mean()),
        "median_confidence": float(np.median(confidences)),
        "mean_correct_confidence": float(confidences[correct].mean()) if correct.any() else None,
        "mean_incorrect_confidence": float(confidences[~correct].mean()) if (~correct).any() else None,
        "confusion_matrix": confusion.tolist(),
        "classes": [
            {
                "digit": int(digit),
                "precision": float(precision[digit]),
                "recall": float(recall[digit]),
                "f1": float(f1[digit]),
                "support": int(true_support[digit]),
                "predicted": int(predicted_support[digit]),
            }
            for digit in DIGITS
        ],
        "calibration": calibration,
        "predictions": predictions.astype(int).tolist(),
        "confidence": confidences.astype(float).tolist(),
        "labels": labels.astype(int).tolist(),
        "correct": correct.astype(bool).tolist(),
        "probabilities": probabilities.astype(float).tolist(),
    }


def save_prediction_csv(path, result):
    header = ["image_index", "actual", "predicted", "confidence", "correct"] + [
        f"probability_{digit}" for digit in DIGITS
    ]
    rows = np.column_stack(
        [
            np.arange(result["count"]),
            result["labels"],
            result["predictions"],
            result["confidence"],
            np.asarray(result["correct"], dtype=np.int64),
            np.asarray(result["probabilities"]),
        ]
    )
    np.savetxt(
        path,
        rows,
        delimiter=",",
        fmt=["%d", "%d", "%d", "%.9f", "%d"] + ["%.9f"] * 10,
        header=",".join(header),
        comments="",
    )


def save_evaluation_plots(plot_path, error_plot_path, images, result, run_id):
    confusion = np.asarray(result["confusion_matrix"])
    classes = result["classes"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    image = axes[0, 0].imshow(confusion, cmap="Blues")
    axes[0, 0].set(
        title="Confusion matrix (actual rows, predicted columns)",
        xlabel="Predicted digit",
        ylabel="Actual digit",
        xticks=DIGITS,
        yticks=DIGITS,
    )
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    x = np.arange(10)
    width = 0.25
    axes[0, 1].bar(x - width, [item["precision"] for item in classes], width, label="Precision")
    axes[0, 1].bar(x, [item["recall"] for item in classes], width, label="Recall")
    axes[0, 1].bar(x + width, [item["f1"] for item in classes], width, label="F1")
    axes[0, 1].set(title="Per-digit scores", xlabel="Digit", ylabel="Score", xticks=DIGITS)
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].legend()

    confidence = np.asarray(result["confidence"])
    correct = np.asarray(result["correct"], dtype=bool)
    axes[1, 0].hist(confidence[correct], bins=30, range=(0, 1), alpha=0.7, label="Correct")
    if (~correct).any():
        axes[1, 0].hist(
            confidence[~correct], bins=30, range=(0, 1), alpha=0.7, label="Incorrect"
        )
    axes[1, 0].set(title="Prediction confidence", xlabel="Maximum class probability", ylabel="Images")
    axes[1, 0].legend()

    nonempty_bins = [item for item in result["calibration"] if item["count"]]
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1, 1].plot(
        [item["mean_confidence"] for item in nonempty_bins],
        [item["accuracy"] for item in nonempty_bins],
        "o-",
        label="Model",
    )
    axes[1, 1].set(
        title=f"Confidence calibration (ECE {result['expected_calibration_error']:.4%})",
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1, 1].legend()
    fig.suptitle(f"MNIST test evaluation — {run_id}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    incorrect_indices = np.flatnonzero(~correct)
    if not incorrect_indices.size:
        return False
    shown_indices = incorrect_indices[:40]
    columns = 5
    rows = (len(shown_indices) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2.3), squeeze=False)
    for axis, index in zip(axes.flat, shown_indices):
        axis.imshow(images[index, ..., 0], cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"#{index}: {result['labels'][index]} → {result['predictions'][index]}\n"
            f"confidence {result['confidence'][index]:.1%}"
        )
        axis.axis("off")
    for axis in axes.flat[len(shown_indices):]:
        axis.axis("off")
    fig.suptitle(f"First {len(shown_indices)} misclassified images")
    fig.tight_layout()
    fig.savefig(error_plot_path, dpi=150)
    plt.close(fig)
    return True


def format_percentage(value):
    return "N/A" if value is None else f"{value:.4%}"


def write_evaluation_report(path, plot_name, errors_plot_name, csv_name, model_path, digest, result, run_id):
    metrics = result
    lines = [
        f"# MNIST test evaluation — {run_id}",
        "",
        f"- **Checkpoint:** `{os.path.abspath(model_path)}`",
        f"- **Checkpoint SHA-256:** `{digest}`",
        f"- **Evaluated at (UTC):** `{run_id}`",
        f"- **Dataset:** MNIST official test split; all {metrics['count']:,} images",
        f"- **TensorFlow:** `{tf.__version__}`",
        f"- **Model parameters:** {metrics['parameter_count']:,}",
        f"- **Model input/output:** `{metrics['input_shape']}` → `{metrics['output_shape']}`",
        "",
        "## Overall metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Accuracy | {format_percentage(metrics['accuracy'])} ({metrics['count'] - metrics['error_count']:,}/{metrics['count']:,}) |",
        f"| Cross-entropy loss | {metrics['cross_entropy']:.8f} |",
        f"| Top-3 accuracy | {format_percentage(metrics['top_3_accuracy'])} |",
        f"| Macro precision | {format_percentage(metrics['macro_precision'])} |",
        f"| Macro recall | {format_percentage(metrics['macro_recall'])} |",
        f"| Macro F1 | {format_percentage(metrics['macro_f1'])} |",
        f"| Weighted F1 | {format_percentage(metrics['weighted_f1'])} |",
        f"| Multiclass Brier score (lower is better) | {metrics['brier_score']:.8f} |",
        f"| Expected calibration error (15 bins; lower is better) | {format_percentage(metrics['expected_calibration_error'])} |",
        f"| Mean / median top-class confidence | {metrics['mean_confidence']:.4%} / {metrics['median_confidence']:.4%} |",
        f"| Mean confidence on correct / incorrect predictions | {format_percentage(metrics['mean_correct_confidence'])} / {format_percentage(metrics['mean_incorrect_confidence'])} |",
        f"| Misclassified images | {metrics['error_count']:,} |",
        "",
        f"![Evaluation charts]({plot_name})",
        "",
    ]
    if errors_plot_name:
        lines.extend([f"![Misclassified examples]({errors_plot_name})", ""])
    lines.extend(
        [
            "## Per-digit metrics",
            "",
            "| Digit | Precision | Recall / accuracy | F1 | Actual examples | Predicted as digit |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["classes"]:
        lines.append(
            f"| {item['digit']} | {item['precision']:.4%} | {item['recall']:.4%} | "
            f"{item['f1']:.4%} | {item['support']:,} | {item['predicted']:,} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are actual digits; columns are predicted digits.",
            "",
            "| Actual \\ Predicted | " + " | ".join(map(str, DIGITS)) + " |",
            "|---:|" + "---:|" * 10,
        ]
    )
    for digit, row in enumerate(metrics["confusion_matrix"]):
        lines.append(f"| {digit} | " + " | ".join(f"{value:,}" for value in row) + " |")
    lines.extend(
        [
            "",
            "## Confidence calibration",
            "",
            "| Confidence interval | Images | Mean confidence | Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in metrics["calibration"]:
        lines.append(
            f"| [{item['lower']:.2f}, {item['upper']:.2f}{']' if item['upper'] == 1 else ')'} "
            f"| {item['count']:,} | {format_percentage(item['mean_confidence'])} | "
            f"{format_percentage(item['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Misclassified images",
            "",
            f"{metrics['error_count']:,} errors, listed below with zero-based test-set index, "
            "actual/predicted digit, top confidence, and the three most likely classes.",
            "",
            "| Index | Actual | Predicted | Confidence | Top 3 class probabilities |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    probabilities = np.asarray(metrics["probabilities"])
    for index, is_correct in enumerate(metrics["correct"]):
        if is_correct:
            continue
        best_classes = np.argsort(probabilities[index])[-3:][::-1]
        alternatives = ", ".join(
            f"{digit}: {probabilities[index, digit]:.4%}" for digit in best_classes
        )
        lines.append(
            f"| {index} | {metrics['labels'][index]} | {metrics['predictions'][index]} | "
            f"{metrics['confidence'][index]:.4%} | {alternatives} |"
        )
    lines.extend(
        [
            "",
            "## Per-image predictions",
            "",
            f"The complete output for every test image, including all ten class probabilities, "
            f"is in [{csv_name}]({csv_name}) (10,000 rows). This preserves exact per-image output "
            "without expanding the report into a 100,000-probability Markdown table.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_comparison_plot(path, current, previous, current_id, previous_id):
    metric_names = ("accuracy", "macro_f1", "top_3_accuracy")
    labels = ("Accuracy", "Macro F1", "Top-3 accuracy")
    positions = np.arange(len(metric_names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        positions - width / 2,
        [previous["metrics"][key] for key in metric_names],
        width,
        label=previous_id,
    )
    axis.bar(
        positions + width / 2,
        [current[key] for key in metric_names],
        width,
        label=current_id,
    )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Score")
    axis.set_title("MNIST test-run comparison")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_comparison_report(
    path, plot_name, current, previous, current_id, previous_id, current_model_sha256
):
    old = previous["metrics"]
    metric_rows = [
        ("Accuracy", "accuracy", True),
        ("Cross-entropy loss", "cross_entropy", False),
        ("Top-3 accuracy", "top_3_accuracy", True),
        ("Macro precision", "macro_precision", True),
        ("Macro recall", "macro_recall", True),
        ("Macro F1", "macro_f1", True),
        ("Weighted F1", "weighted_f1", True),
        ("Brier score", "brier_score", False),
        ("Expected calibration error", "expected_calibration_error", False),
        ("Misclassified images", "error_count", False),
    ]
    lines = [
        f"# MNIST test-run comparison — {current_id} vs {previous_id}",
        "",
        f"- **Current run:** [{current_id}](../current/evaluation_{current_id}.md)",
        f"- **Previous run:** [{previous_id}](../previous/evaluation_{previous_id}.md)",
        f"- **Model changed:** "
        f"{'Yes' if current_model_sha256 != previous['model_sha256'] else 'No'}",
        "",
        "Positive deltas are improvements except for loss, Brier score, calibration error, "
        "and error count, where a negative delta is better.",
        "",
        "| Metric | Previous | Current | Change |",
        "|---|---:|---:|---:|",
    ]
    for label, key, percentage in metric_rows:
        before, after = old[key], current[key]
        if percentage:
            before_text, after_text = format_percentage(before), format_percentage(after)
            delta_text = f"{(after - before) * 100:+.4f} percentage points"
        elif key == "error_count":
            before_text, after_text = f"{before:,}", f"{after:,}"
            delta_text = f"{after - before:+,}"
        else:
            before_text, after_text = f"{before:.8f}", f"{after:.8f}"
            delta_text = f"{after - before:+.8f}"
        lines.append(f"| {label} | {before_text} | {after_text} | {delta_text} |")
    lines.extend(
        [
            "",
            f"![Run metric comparison]({plot_name})",
            "",
            "## Per-digit accuracy change",
            "",
            "| Digit | Previous accuracy | Current accuracy | Change |",
            "|---:|---:|---:|---:|",
        ]
    )
    for before, after in zip(old["classes"], current["classes"]):
        lines.append(
            f"| {before['digit']} | {before['recall']:.4%} | {after['recall']:.4%} | "
            f"{(after['recall'] - before['recall']) * 100:+.4f} percentage points |"
        )
    previous_errors = {
        index
        for index, is_correct in enumerate(old["correct"])
        if not is_correct
    }
    current_errors = {
        index
        for index, is_correct in enumerate(current["correct"])
        if not is_correct
    }
    corrected = sorted(previous_errors - current_errors)
    newly_missed = sorted(current_errors - previous_errors)
    lines.extend(
        [
            "",
            "## Changed image outcomes",
            "",
            f"- **Previously wrong, now correct:** {len(corrected)}"
            + (f" (indices: {', '.join(map(str, corrected))})" if corrected else ""),
            f"- **Previously correct, now wrong:** {len(newly_missed)}"
            + (f" (indices: {', '.join(map(str, newly_missed))})" if newly_missed else ""),
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def clear_run_artifacts(directory):
    for pattern in ("evaluation_*", "misclassified_*"):
        for artifact_path in Path(directory).glob(pattern):
            if artifact_path.is_file():
                artifact_path.unlink()


def clear_comparison_artifacts(directory):
    for artifact_path in Path(directory).glob("comparison_*"):
        if artifact_path.is_file():
            artifact_path.unlink()


def move_run_artifacts(source_dir, destination_dir, run_id):
    for prefix in ("evaluation_", "misclassified_"):
        for artifact_path in Path(source_dir).glob(f"{prefix}{run_id}.*"):
            shutil.move(str(artifact_path), os.path.join(destination_dir, artifact_path.name))


def latest_metadata(directory):
    paths = sorted(
        Path(directory).glob("evaluation_*.json"),
        key=lambda path: path.name,
        reverse=True,
    )
    return paths[0] if paths else None


def prepare_evaluation_folders(output_dir):
    previous_dir = os.path.join(output_dir, "previous")
    current_dir = os.path.join(output_dir, "current")
    comparison_dir = os.path.join(output_dir, "comparison")
    for directory in (previous_dir, current_dir, comparison_dir):
        os.makedirs(directory, exist_ok=True)

    current_metadata_path = latest_metadata(current_dir)
    if current_metadata_path is not None:
        with open(current_metadata_path, encoding="utf-8") as metadata_file:
            previous = json.load(metadata_file)
        previous_id = previous["run_id"]
        clear_run_artifacts(previous_dir)
        move_run_artifacts(current_dir, previous_dir, previous_id)
        clear_run_artifacts(current_dir)
    else:
        previous_metadata_path = latest_metadata(previous_dir)
        if previous_metadata_path is None:
            previous_metadata_path = latest_metadata(output_dir)
            if previous_metadata_path is not None:
                with open(previous_metadata_path, encoding="utf-8") as metadata_file:
                    previous = json.load(metadata_file)
                previous_id = previous["run_id"]
                clear_run_artifacts(previous_dir)
                move_run_artifacts(output_dir, previous_dir, previous_id)
                clear_run_artifacts(output_dir)
                clear_comparison_artifacts(output_dir)
            else:
                previous = None
                previous_id = None
        else:
            with open(previous_metadata_path, encoding="utf-8") as metadata_file:
                previous = json.load(metadata_file)
            previous_id = previous["run_id"]

    clear_run_artifacts(current_dir)
    clear_comparison_artifacts(comparison_dir)
    return current_dir, comparison_dir, previous, previous_id


def run_evaluation(model_path, output_dir):
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    images, labels = load_test_data()
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = np.asarray(model.predict(images, batch_size=512, verbose=0))
    if probabilities.shape != (10_000, 10):
        raise ValueError(
            f"Expected model output shape (10000, 10), got {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Model produced non-finite or negative class probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Model outputs must be ten-class probabilities that sum to one.")

    result = calculate_metrics(labels, probabilities)
    result["parameter_count"] = int(model.count_params())
    result["input_shape"] = str(model.input_shape)
    result["output_shape"] = str(model.output_shape)
    digest = file_sha256(model_path)
    (
        current_dir,
        comparison_dir,
        previous,
        previous_id,
    ) = prepare_evaluation_folders(output_dir)
    plot_name = f"evaluation_{run_id}.png"
    errors_plot_name = f"misclassified_{run_id}.png"
    csv_name = f"evaluation_{run_id}.csv"
    report_name = f"evaluation_{run_id}.md"
    plot_path = os.path.join(current_dir, plot_name)
    errors_plot_path = os.path.join(current_dir, errors_plot_name)
    csv_path = os.path.join(current_dir, csv_name)
    report_path = os.path.join(current_dir, report_name)
    has_error_plot = save_evaluation_plots(
        plot_path, errors_plot_path, images, result, run_id
    )
    save_prediction_csv(csv_path, result)
    write_evaluation_report(
        report_path,
        plot_name,
        errors_plot_name if has_error_plot else None,
        csv_name,
        model_path,
        digest,
        result,
        run_id,
    )

    comparable_metrics = {
        key: value
        for key, value in result.items()
        if key not in {"predictions", "confidence", "labels", "probabilities"}
    }
    metadata = {
        "run_id": run_id,
        "model_path": model_path,
        "model_sha256": digest,
        "tensorflow_version": tf.__version__,
        "metrics": comparable_metrics,
    }
    metadata_path = os.path.join(current_dir, f"evaluation_{run_id}.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    if previous is not None:
        comparison_id = f"{run_id}_vs_{previous_id}"
        comparison_plot_name = f"comparison_{comparison_id}.png"
        comparison_report_name = f"comparison_{comparison_id}.md"
        save_comparison_plot(
            os.path.join(comparison_dir, comparison_plot_name),
            result,
            previous,
            run_id,
            previous_id,
        )
        write_comparison_report(
            os.path.join(comparison_dir, comparison_report_name),
            comparison_plot_name,
            result,
            previous,
            run_id,
            previous_id,
            digest,
        )

    print(
        f"Evaluation complete: {result['accuracy']:.4%} accuracy, "
        f"{result['error_count']} errors on {result['count']:,} test images."
    )
    print(f"Report: {report_path}")
    print(f"Predictions: {csv_path}")
    if previous_id is not None:
        comparison_path = os.path.join(
            comparison_dir, f"comparison_{run_id}_vs_{previous_id}.md"
        )
        print(f"Comparison: {comparison_path}")
    else:
        print("A comparison report will be created after a second evaluation run.")


def main():
    command_parser = argparse.ArgumentParser(description=__doc__)
    command_parser.add_argument(
        "--checkpoint",
        default=os.path.join(SCRIPT_DIR, "artifacts", "mnist_best.keras"),
    )
    command_parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "eval"),
        help="Base directory containing previous, current, and comparison subfolders.",
    )
    args = command_parser.parse_args()
    run_evaluation(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()

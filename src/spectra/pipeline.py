"""Command line utilities for training and analysing SDSS spectra."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .data import Spectrum, load_spectrum, train_validation_split
from .dataset import SpectraDataset
from .features import FeatureSummary, find_emission_absorption_features, nearest_reference_lines
from .labels import LabelCatalog
from .metadata import (
    PreprocessingLogger,
    build_spectra_manifest,
    collect_label_rows,
    collect_metadata_rows,
    write_label_table,
    write_manifest_table,
    write_metadata_table,
    write_preprocessing_log,
)
from .model import ModelConfig, SpectralClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spectral classifier and inspect features.")
    parser.add_argument("data_dir", type=Path, help="Directory containing FITS files organised by class labels")
    parser.add_argument("--input-length", type=int, default=2048, help="Length to which spectra are padded or cropped")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--plot-example",
        type=Path,
        help="Path to a FITS file to visualise spectral features",
    )
    parser.add_argument(
        "--history-plot",
        type=Path,
        help="Optional path to save a PNG showing training/validation accuracy and loss",
    )
    parser.add_argument(
        "--no-show-history",
        action="store_true",
        help="Disable interactive display of the training history graph",
    )
    parser.add_argument(
        "--label-catalog",
        type=Path,
        help="Optional CSV exported from MaStar, APOGEE DR17, or SDSS-V metadata containing ground-truth labels",
    )
    parser.add_argument(
        "--catalog-object-field",
        type=str,
        help="Column name in the CSV that identifies each spectrum (required when --label-catalog is set)",
    )
    parser.add_argument(
        "--catalog-spectral-field",
        type=str,
        help="Column containing spectral class labels (e.g., OBAFGKM)",
    )
    parser.add_argument(
        "--catalog-luminosity-field",
        type=str,
        help="Column containing luminosity class labels",
    )
    parser.add_argument(
        "--catalog-teff-field",
        type=str,
        help="Column with effective temperatures (T_eff)",
    )
    parser.add_argument(
        "--catalog-logg-field",
        type=str,
        help="Column with log g surface gravities",
    )
    parser.add_argument(
        "--catalog-feh-field",
        type=str,
        help="Column with [Fe/H] metallicities",
    )
    parser.add_argument(
        "--catalog-source-label",
        type=str,
        help="Optional label describing the catalogue source for reporting",
    )
    parser.add_argument(
        "--catalog-quality-field",
        type=str,
        help="Column containing label quality flags or confidence codes",
    )
    parser.add_argument(
        "--catalog-strip-prefix",
        type=str,
        help="Prefix to strip from FITS filenames before catalogue lookup",
    )
    parser.add_argument(
        "--catalog-strip-suffix",
        type=str,
        help="Suffix to strip from FITS filenames before catalogue lookup",
    )
    parser.add_argument(
        "--metadata-table",
        type=Path,
        help="Path to store per-spectrum metadata (CSV or Parquet)",
    )
    parser.add_argument(
        "--metadata-format",
        choices=("csv", "parquet"),
        help="Explicit metadata serialisation format when extension is ambiguous",
    )
    parser.add_argument(
        "--data-release",
        type=str,
        help="Data release identifier to record in the metadata table",
    )
    parser.add_argument(
        "--retrieval-date",
        type=str,
        help="Retrieval date (ISO format) to log alongside metadata",
    )
    parser.add_argument(
        "--survey-wavelength-calibrated",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record whether the survey pipeline delivered wavelength-calibrated spectra",
    )
    parser.add_argument(
        "--survey-flux-calibrated",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record whether the survey pipeline delivered flux-calibrated spectra",
    )
    parser.add_argument(
        "--post-processing-steps",
        type=str,
        help="Comma-separated summary of any continuum normalisation or other processing applied",
    )
    parser.add_argument(
        "--calibration-reference",
        action="append",
        help="Optional reference (URL or citation) describing survey calibration accuracy",
    )
    parser.add_argument(
        "--spectra-manifest",
        type=Path,
        help="Path to store the spectra manifest (CSV or Parquet)",
    )
    parser.add_argument(
        "--manifest-format",
        choices=("csv", "parquet"),
        help="Explicit manifest serialisation format when extension is ambiguous",
    )
    parser.add_argument(
        "--survey",
        type=str,
        help="Survey identifier to record in the manifest (e.g., SDSS-IV)",
    )
    parser.add_argument(
        "--manifest-notes",
        type=str,
        help="Free-form notes stored alongside each manifest row",
    )
    parser.add_argument(
        "--manifest-created-at",
        type=str,
        help="ISO 8601 timestamp to record as the manifest creation time",
    )
    parser.add_argument(
        "--label-table",
        type=Path,
        help="Path to store the ground-truth label table (CSV or Parquet)",
    )
    parser.add_argument(
        "--label-table-format",
        choices=("csv", "parquet"),
        help="Explicit label table serialisation format when extension is ambiguous",
    )
    parser.add_argument(
        "--preproc-log",
        type=Path,
        help="Path to store the preprocessing log (CSV or Parquet)",
    )
    parser.add_argument(
        "--preproc-log-format",
        choices=("csv", "parquet"),
        help="Explicit preprocessing log format when extension is ambiguous",
    )
    return parser.parse_args()


def _parse_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("--retrieval-date must be in ISO format (YYYY-MM-DD)") from exc


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("--manifest-created-at must be in ISO 8601 format") from exc


def train_classifier(args: argparse.Namespace) -> tf.keras.callbacks.History:
    label_catalog = None
    id_from_path = None
    if args.label_catalog is not None:
        if args.catalog_object_field is None:
            raise ValueError("--catalog-object-field must be set when providing --label-catalog")
        label_catalog = LabelCatalog.from_csv(
            args.label_catalog,
            object_id_field=args.catalog_object_field,
            spectral_class_field=args.catalog_spectral_field,
            luminosity_class_field=args.catalog_luminosity_field,
            effective_temperature_field=args.catalog_teff_field,
            surface_gravity_field=args.catalog_logg_field,
            metallicity_field=args.catalog_feh_field,
            source_label=args.catalog_source_label,
            quality_field=args.catalog_quality_field,
        )

        def build_object_id(path: Path) -> str:
            name = path.stem
            if args.catalog_strip_prefix and name.startswith(args.catalog_strip_prefix):
                name = name[len(args.catalog_strip_prefix) :]
            if args.catalog_strip_suffix and name.endswith(args.catalog_strip_suffix):
                name = name[: -len(args.catalog_strip_suffix)]
            return name

        id_from_path = build_object_id

    preproc_logger = PreprocessingLogger() if args.preproc_log is not None else None

    dataset = SpectraDataset(
        args.data_dir,
        label_catalog=label_catalog,
        id_from_path=id_from_path,
        preprocessing_logger=preproc_logger,
    )
    files = dataset.list_files()
    if not files:
        raise ValueError(f"No FITS files found under {args.data_dir}")

    distribution = dataset.class_distribution()
    total_examples = sum(distribution.values())
    print(f"Discovered {total_examples} spectra across {len(distribution)} classes.")
    for label, count in distribution.items():
        print(f"  - {label}: {count}")
    if total_examples < 5000:
        print(
            "Warning: Fewer than 5,000 spectra detected. Accuracy may suffer without a larger sample."
        )
    if label_catalog is not None:
        misses = dataset.catalog_misses()
        if misses:
            print(
                "Warning: Some spectra lacked catalogue labels and fell back to directory names:"
            )
            for miss in misses[:10]:
                print(f"  - {miss}")
            if len(misses) > 10:
                print(f"  ... ({len(misses) - 10} additional IDs omitted)")

    if args.metadata_table:
        metadata_rows = collect_metadata_rows(
            files,
            object_id_fn=dataset.object_id_from_path,
            data_release=args.data_release,
            retrieval_date=_parse_date(args.retrieval_date),
            label_catalog=label_catalog,
            survey_wavelength_calibrated=args.survey_wavelength_calibrated,
            survey_flux_calibrated=args.survey_flux_calibrated,
            post_processing_steps=args.post_processing_steps,
            calibration_references=args.calibration_reference,
        )
        output_path = write_metadata_table(
            metadata_rows,
            args.metadata_table,
            output_format=args.metadata_format,
        )
        print(f"Wrote metadata for {len(metadata_rows)} spectra to {output_path}")

    if args.spectra_manifest:
        manifest_rows = build_spectra_manifest(
            files,
            object_id_fn=dataset.object_id_from_path,
            survey=args.survey,
            data_release=args.data_release,
            created_at=_parse_timestamp(args.manifest_created_at),
            notes=args.manifest_notes,
        )
        manifest_path = write_manifest_table(
            manifest_rows,
            args.spectra_manifest,
            output_format=args.manifest_format,
        )
        print(f"Wrote manifest with {len(manifest_rows)} entries to {manifest_path}")

    if args.label_table:
        if label_catalog is None:
            raise ValueError("--label-table requires --label-catalog so labels can be resolved")
        label_rows = collect_label_rows(
            files,
            object_id_fn=dataset.object_id_from_path,
            label_catalog=label_catalog,
        )
        label_path = write_label_table(
            label_rows,
            args.label_table,
            output_format=args.label_table_format,
        )
        print(f"Wrote label table with {len(label_rows)} rows to {label_path}")

    train_files, val_files = train_validation_split(files)

    train_examples = list(dataset.examples_from_paths(train_files))
    val_examples = list(dataset.examples_from_paths(val_files))
    if not train_examples:
        raise ValueError("Dataset split produced no training examples. Ensure enough FITS files are available.")
    if not val_examples:
        raise ValueError("Dataset split produced no validation examples. Adjust validation_fraction or add more data.")
    labels = sorted({example.label for example in train_examples + val_examples})
    label_indices = {label: idx for idx, label in enumerate(labels)}

    def to_dataset(examples):
        def generator():
            for example in examples:
                flux = example.flux
                original_len = int(flux.shape[0])
                flux = _pad_or_crop(flux, args.input_length)
                flux32 = flux.astype(np.float32)
                if preproc_logger is not None:
                    action = "unchanged"
                    if original_len > args.input_length:
                        action = "cropped"
                    elif original_len < args.input_length:
                        action = "padded"
                    preproc_logger.log_step(
                        example.spec_id,
                        "pad_or_crop",
                        params={
                            "target_length": args.input_length,
                            "original_length": original_len,
                            "action": action,
                        },
                        result=flux32,
                    )
                yield flux32, np.array(label_indices[example.label], dtype=np.int32)

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(args.input_length,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        ds = ds.shuffle(buffer_size=len(examples)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = to_dataset(train_examples)
    val_ds = to_dataset(val_examples)

    model = SpectralClassifier(ModelConfig(num_classes=len(labels), input_length=args.input_length))
    model.compile_for_training(args.learning_rate)

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    if preproc_logger is not None and args.preproc_log is not None:
        rows = preproc_logger.rows()
        if rows:
            log_path = write_preprocessing_log(
                rows,
                args.preproc_log,
                output_format=args.preproc_log_format,
            )
            print(f"Wrote preprocessing log with {len(rows)} steps to {log_path}")
        else:
            print("Warning: No preprocessing steps were recorded; skipping log export.")
    return history


def plot_training_history(
    history: tf.keras.callbacks.History,
    *,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """Plot training history charts for accuracy and loss.

    Parameters
    ----------
    history
        Keras training history returned by :func:`model.fit`.
    show
        Whether to display the matplotlib figure interactively.
    save_path
        Optional path to save the figure as a PNG.
    """

    metrics = history.history
    epochs = np.arange(1, len(next(iter(metrics.values()), [])) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    for name, ax, title, ylabel in [
        ("accuracy", ax_acc, "Accuracy", "Accuracy"),
        ("loss", ax_loss, "Loss", "Loss"),
    ]:
        train_values = metrics.get(name)
        val_values = metrics.get(f"val_{name}")
        if train_values is None:
            continue
        ax.plot(epochs, train_values, label="Train", marker="o")
        if val_values is not None:
            ax.plot(epochs, val_values, label="Validation", marker="s")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    fig.suptitle("Training progress")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


def _latest_metric(history: tf.keras.callbacks.History, name: str) -> float | None:
    values = history.history.get(name)
    if values:
        return values[-1]
    return None


def plot_spectrum_with_features(fits_path: Path) -> None:
    spectrum = load_spectrum(fits_path)
    summary = find_emission_absorption_features(spectrum.flux, spectrum.wavelength)
    _plot_summary(spectrum, summary)


def _plot_summary(spectrum: Spectrum, summary: FeatureSummary) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(spectrum.wavelength, spectrum.flux, color="black", linewidth=1)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalized Flux")
    calibration = spectrum.calibration
    calibration_label = (
        f"{calibration.method} (log10 grid)"
        if calibration.is_logarithmic
        else calibration.method
    )
    ax.set_title(
        f"Spectrum: {spectrum.meta.get('filename', 'Unknown')}\n"
        f"Calibration: {calibration_label}"
    )

    for feature in summary.emission:
        ax.axvline(feature.wavelength, color="red", linestyle="--", alpha=0.5)
        ax.text(feature.wavelength, np.max(spectrum.flux), "Emission", rotation=90, color="red")
    for feature in summary.absorption:
        ax.axvline(feature.wavelength, color="blue", linestyle=":", alpha=0.5)
        ax.text(feature.wavelength, np.min(spectrum.flux), "Absorption", rotation=90, color="blue")

    associations = nearest_reference_lines(summary.emission + summary.absorption)
    for element, features in associations.items():
        if not features:
            continue
        ax.text(0.02, 0.95 - 0.05 * len(features), f"{element}: {len(features)} lines", transform=ax.transAxes)

    plt.tight_layout()
    plt.show()


def _pad_or_crop(array: np.ndarray, length: int) -> np.ndarray:
    if array.shape[0] == length:
        return array
    if array.shape[0] > length:
        return array[:length]
    padded = np.zeros(length, dtype=array.dtype)
    padded[: array.shape[0]] = array
    return padded


def main() -> None:
    args = parse_args()
    history = train_classifier(args)
    train_acc = _latest_metric(history, "accuracy")
    val_acc = _latest_metric(history, "val_accuracy")
    if train_acc is not None:
        print("Training accuracy:", train_acc)
    if val_acc is not None:
        print("Validation accuracy:", val_acc)

    plot_training_history(
        history,
        show=not args.no_show_history,
        save_path=args.history_plot,
    )

    if args.plot_example:
        plot_spectrum_with_features(args.plot_example)


if __name__ == "__main__":
    main()

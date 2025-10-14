"""High-level dataset utilities for SDSS spectral classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import tensorflow as tf

from .data import load_spectrum
from .labels import LabelCatalog
from .metadata import PreprocessingLogger


@dataclass
class SpectrumExample:
    """Represents a spectrum with an associated categorical label."""

    spec_id: str
    wavelength: np.ndarray
    flux: np.ndarray
    label: str


class SpectraDataset:
    """Create TensorFlow datasets from directories of SDSS FITS spectra."""

    def __init__(
        self,
        root: Path | str,
        label_map: Dict[str, str] | None = None,
        normalise_flux: bool = True,
        label_catalog: LabelCatalog | None = None,
        id_from_path: Callable[[Path], str] | None = None,
        preprocessing_logger: PreprocessingLogger | None = None,
    ) -> None:
        self.root = Path(root)
        self.label_map = label_map or {}
        self.normalise_flux = normalise_flux
        self.label_catalog = label_catalog
        self._id_from_path = id_from_path or (lambda path: path.stem)
        self._catalog_misses: set[str] = set()
        self._preprocessing_logger = preprocessing_logger

    def list_files(self) -> List[Path]:
        """Return all FITS files discovered under the dataset root."""

        return sorted(self.root.rglob("*.fits"))

    # Backwards compatibility for earlier internal usage.
    _discover_files = list_files

    def class_distribution(self) -> Dict[str, int]:
        """Count spectra per class label for balance reporting."""

        counts: Dict[str, int] = {}
        for path in self.list_files():
            label = self._label_from_path(path)
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def _label_from_path(self, path: Path) -> str:
        if self.label_catalog is not None:
            object_id = self._id_from_path(path)
            catalog_label = self.label_catalog.class_label(object_id)
            if catalog_label is not None:
                return catalog_label
            self._catalog_misses.add(object_id)
        if path.parent.name in self.label_map:
            return self.label_map[path.parent.name]
        return path.parent.name

    def catalog_misses(self) -> List[str]:
        """Return object identifiers that lacked catalogue labels."""

        return sorted(self._catalog_misses)

    def object_id_from_path(self, path: Path | str) -> str:
        """Return the catalogue lookup identifier for *path*."""

        return self._id_from_path(Path(path))

    def _prepare_flux(self, spec_id: str, flux: np.ndarray) -> np.ndarray:
        if not self.normalise_flux:
            return flux
        flux_min = float(np.nanmin(flux))
        flux_max = float(np.nanmax(flux))
        if flux_max - flux_min == 0:
            if self._preprocessing_logger is not None:
                self._preprocessing_logger.log_step(
                    spec_id,
                    "continuum_normalise",
                    params={"method": "min-max", "status": "skipped"},
                    result=flux,
                    warnings="Zero dynamic range; normalisation skipped",
                )
            return flux
        normalised = (flux - flux_min) / (flux_max - flux_min)
        if self._preprocessing_logger is not None:
            self._preprocessing_logger.log_step(
                spec_id,
                "continuum_normalise",
                params={"method": "min-max", "min": flux_min, "max": flux_max},
                result=normalised,
            )
        return normalised

    def _make_example(self, path: Path) -> SpectrumExample:
        spectrum = load_spectrum(path)
        spec_id = self._id_from_path(path)
        if self._preprocessing_logger is not None:
            self._preprocessing_logger.log_step(
                spec_id,
                "load_spectrum",
                params={"source_path": str(path)},
                result=spectrum.flux,
            )
        return SpectrumExample(
            spec_id=spec_id,
            wavelength=spectrum.wavelength,
            flux=self._prepare_flux(spec_id, spectrum.flux),
            label=self._label_from_path(path),
        )

    def examples(self) -> Iterator[SpectrumExample]:
        for path in self.list_files():
            yield self._make_example(path)

    def examples_from_paths(self, paths: Iterable[Path | str]) -> Iterator[SpectrumExample]:
        for path in paths:
            yield self._make_example(Path(path))

    def to_tensorflow_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        output_length: int | None = None,
        pad_value: float = 0.0,
    ) -> tf.data.Dataset:
        """Convert discovered spectra to a `tf.data.Dataset`.

        Parameters
        ----------
        batch_size:
            Number of samples per batch.
        shuffle:
            Whether to shuffle the dataset before batching.
        output_length:
            Optionally crop or pad spectra to a fixed length for model input.
        pad_value:
            Value to use when padding shorter spectra.
        """

        examples = list(self.examples())
        labels = sorted({example.label for example in examples})
        label_indices = {label: idx for idx, label in enumerate(labels)}

        def generator() -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            for example in examples:
                flux = example.flux
                if output_length is not None:
                    flux = _pad_or_crop(flux, output_length, pad_value)
                yield flux.astype(np.float32), example.wavelength.astype(np.float32), np.array(
                    label_indices[example.label], dtype=np.int32
                )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(output_length or None,), dtype=tf.float32),
                tf.TensorSpec(shape=(output_length or None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(examples))
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)


def _pad_or_crop(array: np.ndarray, length: int, pad_value: float) -> np.ndarray:
    if array.shape[0] == length:
        return array
    if array.shape[0] > length:
        return array[:length]
    padded = np.full(length, pad_value, dtype=array.dtype)
    padded[: array.shape[0]] = array
    return padded

"""Utilities for working with astronomical spectra from the SDSS SkyServer."""

from .data import WavelengthCalibration, Spectrum, load_spectrum, build_wavelength_axis
from .features import find_emission_absorption_features
from .dataset import SpectraDataset
from .labels import GroundTruthLabel, LabelCatalog
from .metadata import (
    SpectrumMetadataRow,
    SpectraManifestRow,
    LabelTableRow,
    PreprocessingLogRow,
    build_spectra_manifest,
    collect_label_rows,
    collect_metadata_rows,
    write_label_table,
    write_manifest_table,
    write_metadata_table,
    PreprocessingLogger,
    write_preprocessing_log,
)
from .model import SpectralClassifier

__all__ = [
    "WavelengthCalibration",
    "Spectrum",
    "load_spectrum",
    "build_wavelength_axis",
    "find_emission_absorption_features",
    "SpectraDataset",
    "GroundTruthLabel",
    "LabelCatalog",
    "SpectrumMetadataRow",
    "SpectraManifestRow",
    "LabelTableRow",
    "PreprocessingLogRow",
    "collect_metadata_rows",
    "collect_label_rows",
    "build_spectra_manifest",
    "write_metadata_table",
    "write_manifest_table",
    "write_label_table",
    "PreprocessingLogger",
    "write_preprocessing_log",
    "SpectralClassifier",
]

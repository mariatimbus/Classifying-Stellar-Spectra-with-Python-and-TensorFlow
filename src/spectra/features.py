"""Spectral feature detection for emission and absorption lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.signal import find_peaks

# Wavelengths in Angstroms for prominent lines of selected elements.
REFERENCE_LINES = {
    "Hydrogen": [6562.79, 4861.35, 4340.47, 4101.74],  # H-alpha, H-beta, H-gamma, H-delta
    "Helium": [5875.62, 4471.50, 4026.19],
    "Calcium": [3933.66, 3968.47],  # Ca II K, Ca II H
    "Iron": [5270.40, 5169.03, 5018.44],
    "Magnesium": [5175.4, 4481.13],
    "Sodium": [5891.58, 5897.56],
}


@dataclass
class SpectralFeature:
    wavelength: float
    prominence: float
    kind: str  # "emission" or "absorption"


@dataclass
class FeatureSummary:
    emission: List[SpectralFeature]
    absorption: List[SpectralFeature]


def find_emission_absorption_features(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    prominence: float = 0.1,
    distance: int | None = None,
) -> FeatureSummary:
    """Identify emission and absorption features using local extrema.

    Parameters
    ----------
    spectrum:
        Flux values, ideally normalised.
    wavelength:
        Wavelength axis in Angstroms or nanometres.
    prominence:
        Minimum prominence for peak detection; adjust based on normalisation.
    distance:
        Minimum number of samples between peaks.
    """

    emission_indices, emission_properties = find_peaks(spectrum, prominence=prominence, distance=distance)
    absorption_indices, absorption_properties = find_peaks(-spectrum, prominence=prominence, distance=distance)

    emission = [
        SpectralFeature(wavelength=float(wavelength[idx]), prominence=float(emission_properties["prominences"][i]), kind="emission")
        for i, idx in enumerate(emission_indices)
    ]
    absorption = [
        SpectralFeature(wavelength=float(wavelength[idx]), prominence=float(absorption_properties["prominences"][i]), kind="absorption")
        for i, idx in enumerate(absorption_indices)
    ]
    return FeatureSummary(emission=emission, absorption=absorption)


def nearest_reference_lines(features: Iterable[SpectralFeature], tolerance: float = 5.0) -> Dict[str, List[SpectralFeature]]:
    """Map detected features to nearest known reference lines within tolerance."""

    associations: Dict[str, List[SpectralFeature]] = {element: [] for element in REFERENCE_LINES}
    for feature in features:
        for element, lines in REFERENCE_LINES.items():
            for line in lines:
                if abs(feature.wavelength - line) <= tolerance:
                    associations[element].append(feature)
                    break
    return associations

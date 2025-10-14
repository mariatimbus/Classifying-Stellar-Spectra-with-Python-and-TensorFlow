"""Data loading helpers for SDSS SkyServer spectral FITS files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from astropy.io import fits


@dataclass
class WavelengthCalibration:
    """Describe the wavelength solution reconstructed from FITS metadata."""

    method: str
    is_logarithmic: bool
    keywords: dict[str, float]


@dataclass
class Spectrum:
    """Represents a single spectrum with wavelength and flux arrays."""

    wavelength: np.ndarray
    flux: np.ndarray
    calibration: WavelengthCalibration
    meta: dict


def build_wavelength_axis(
    header: fits.Header,
    flux: np.ndarray,
    wavelength: Optional[np.ndarray] = None,
    wavelength_source: Optional[str] = None,
) -> Tuple[np.ndarray, WavelengthCalibration]:
    """Build a wavelength axis from FITS header metadata.

    Parameters
    ----------
    header:
        FITS header that may contain wavelength calibration keywords.
    flux:
        Flux array; used to infer the length of the wavelength axis.
    wavelength:
        Optional wavelength column already extracted from the FITS file.
    wavelength_source:
        Name of the FITS column used for ``wavelength`` (e.g. ``loglam``) when
        provided externally.
    Returns
    -------
    Tuple[np.ndarray, WavelengthCalibration]
        The wavelength axis and a description of the calibration approach.
    """

    if wavelength is not None:
        provided = np.asarray(wavelength, dtype=np.float32)
        is_log = wavelength_source == "loglam"
        method = "provided-log10" if is_log else "provided"
        return provided, WavelengthCalibration(method=method, is_logarithmic=is_log, keywords={})

    # Typical SDSS spectra use logarithmic wavelength calibration with LOG10
    # dispersion. When not present we fall back to linear calibration.
    naxis1 = header.get("NAXIS1", flux.shape[-1])
    crval1 = header.get("CRVAL1")
    cdelt1 = header.get("CDELT1")
    coeff0 = header.get("COEFF0")
    coeff1 = header.get("COEFF1")

    if coeff0 is not None and coeff1 is not None:
        pixel_index = np.arange(naxis1, dtype=np.float32)
        loglam = coeff0 + coeff1 * pixel_index
        axis = np.power(10.0, loglam, dtype=np.float32)
        calibration = WavelengthCalibration(
            method="log10-polynomial",
            is_logarithmic=True,
            keywords={"COEFF0": float(coeff0), "COEFF1": float(coeff1)},
        )
        return axis, calibration

    if crval1 is not None and cdelt1 is not None:
        pixel_index = np.arange(naxis1, dtype=np.float32)
        axis = crval1 + cdelt1 * pixel_index
        calibration = WavelengthCalibration(
            method="linear",
            is_logarithmic=False,
            keywords={"CRVAL1": float(crval1), "CDELT1": float(cdelt1)},
        )
        return axis.astype(np.float32), calibration

    # As a last resort, normalise pixel numbers to unity.
    pixel_index = np.arange(naxis1, dtype=np.float32)
    axis = pixel_index / np.max(pixel_index)
    calibration = WavelengthCalibration(
        method="pixel-normalised",
        is_logarithmic=False,
        keywords={},
    )
    return axis, calibration


def load_spectrum(path: Path | str) -> Spectrum:
    """Load a spectrum from a FITS file.

    The function handles several SDSS table layouts, including extensions
    where flux values live in the first binary table and one-dimensional
    primary HDU images.
    """

    fits_path = Path(path)
    with fits.open(fits_path, memmap=False) as hdul:
        wavelength_source = None
        if len(hdul) > 1 and hasattr(hdul[1], "columns"):
            table = hdul[1].data
            columns = {name.lower(): table[name] for name in table.names}
            flux = np.asarray(columns.get("flux"))
            if flux is None:
                raise KeyError("FITS table does not contain a 'flux' column")
            if flux.ndim > 1:
                flux = flux.squeeze()
            wavelength = None
            wavelength_source = None
            for candidate in ("wavelength", "lambda", "loglam"):
                if candidate in columns:
                    wavelength = columns[candidate]
                    if candidate == "loglam":
                        wavelength = np.power(10.0, wavelength, dtype=np.float32)
                    wavelength_source = candidate
                    break
            header = hdul[1].header
        else:
            image = hdul[0].data
            if image is None:
                raise ValueError("FITS file does not contain image data")
            flux = np.asarray(image, dtype=np.float32).squeeze()
            header = hdul[0].header
            wavelength = None
            wavelength_source = None

        wavelength_axis, calibration = build_wavelength_axis(
            header, flux, wavelength, wavelength_source
        )
        meta = {
            "filename": fits_path.name,
            "header": dict(header),
            "calibration": {
                "method": calibration.method,
                "is_logarithmic": calibration.is_logarithmic,
                "keywords": calibration.keywords,
                "wavelength_source": wavelength_source,
            },
        }

    return Spectrum(
        wavelength=wavelength_axis,
        flux=np.asarray(flux, dtype=np.float32),
        calibration=calibration,
        meta=meta,
    )


def iter_spectra(paths: Iterable[Path | str]) -> Iterable[Spectrum]:
    """Yield `Spectrum` objects for the given FITS files."""

    for path in paths:
        yield load_spectrum(path)


def train_validation_split(
    paths: Iterable[Path | str],
    validation_fraction: float = 0.2,
) -> Tuple[list[Path], list[Path]]:
    """Split file paths into train and validation subsets."""

    paths = [Path(p) for p in paths]
    rng = np.random.default_rng(seed=42)
    rng.shuffle(paths)
    split_index = int(len(paths) * (1 - validation_fraction))
    return paths[:split_index], paths[split_index:]

"""Utilities for extracting and storing per-spectrum provenance metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, MutableMapping, Sequence, Set

import numpy as np


SPEED_OF_LIGHT_KMS = 299_792.458

from .data import load_spectrum
from .labels import LabelCatalog


@dataclass
class SpectrumMetadataRow:
    """Structured per-spectrum observing and instrument metadata."""

    spec_id: str
    obj_id: str | None
    file_path: str
    plate: int | None = None
    mjd: int | None = None
    fiber: int | None = None
    instrument: str | None = None
    resolution_r: float | None = None
    resolution_source: str | None = None
    exptime_s: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    z_or_rv_kms: float | None = None
    heliocentric_corr_applied: bool | None = None
    sn_median: float | None = None
    sn_per_pixel: float | None = None
    ebv_sfd: float | None = None
    mask_frac_bad: float | None = None
    quality_flags: str | None = None
    wavelength_frame: str | None = None
    flux_unit: str | None = None
    data_release: str | None = None
    retrieval_date: str | None = None
    survey_wavelength_calibrated: bool | None = None
    survey_flux_calibrated: bool | None = None
    post_processing_steps: str | None = None
    calibration_references: str | None = None

    def to_dict(self) -> MutableMapping[str, object | None]:
        return asdict(self)


@dataclass
class SpectraManifestRow:
    """Normalised manifest entry describing a spectrum's provenance."""

    spec_id: str
    survey: str | None
    data_release: str | None
    source_path: str
    checksum: str
    wavelength_frame: str | None
    wavelength_grid: str | None
    crval1: float | None
    cd1_1: float | None
    coeff0: float | None
    coeff1: float | None
    flux_unit: str | None
    created_at: str
    notes: str | None = None

    def to_dict(self) -> MutableMapping[str, object | None]:
        return asdict(self)


@dataclass
class LabelTableRow:
    """Normalised ground-truth targets referenced by ``spec_id``."""

    spec_id: str
    class_mk: str | None
    teff_k: float | None
    logg_cgs: float | None
    fe_h_dex: float | None
    label_source: str | None
    label_quality: str | None

    def to_dict(self) -> MutableMapping[str, object | None]:
        return asdict(self)


@dataclass
class PreprocessingLogRow:
    """Normalised record describing a preprocessing step for a spectrum."""

    spec_id: str
    step_order: int
    step_name: str
    params_json: str
    result_hash: str
    timestamp: str
    warnings: str | None = None

    def to_dict(self) -> MutableMapping[str, object | None]:
        return asdict(self)


def collect_metadata_rows(
    paths: Iterable[Path | str],
    *,
    object_id_fn: Callable[[Path], str],
    data_release: str | None = None,
    retrieval_date: datetime | None = None,
    label_catalog: LabelCatalog | None = None,
    survey_wavelength_calibrated: bool | None = None,
    survey_flux_calibrated: bool | None = None,
    post_processing_steps: str | None = None,
    calibration_references: Sequence[str] | None = None,
) -> List[SpectrumMetadataRow]:
    """Extract :class:`SpectrumMetadataRow` entries for the given FITS files.

    The optional survey calibration flags and post-processing description are
    copied verbatim to each row so downstream documentation can assert whether
    SDSS pipelines already handled wavelength/flux calibration and what
    additional normalisation or masking steps were applied locally.
    """

    retrieval_iso = _normalise_retrieval_date(retrieval_date)
    rows: List[SpectrumMetadataRow] = []
    for path_like in paths:
        path = Path(path_like)
        spectrum = load_spectrum(path)
        header: Mapping[str, object] = spectrum.meta.get("header", {})
        spec_id = object_id_fn(path)
        label = label_catalog.get(spec_id) if label_catalog is not None else None

        quality_dict = _quality_flags(header)
        sn_median, sn_per_pixel = _sn_metrics(header)
        radial_velocity = _first_numeric(
            header,
            (
                "VHELIO",
                "VRAD",
                "RVEL",
                "VELOCITY",
                "Z_V",
                "HELIOVEL",
            ),
        )
        redshift = _first_numeric(header, ("Z", "REDH", "ZHELIO"))

        rows.append(
            SpectrumMetadataRow(
                spec_id=spec_id,
                obj_id=_object_identifier(header, label),
                file_path=str(path),
                plate=_first_int(header, ("PLATEID", "PLATE")),
                mjd=_first_int(header, ("MJD", "MJD5")),
                fiber=_first_int(header, ("FIBERID", "FIBER", "FIBERIDR")),
                instrument=_clean_str(_first_value(header, ("INSTRUME", "TELESCOP"))),
                resolution_r=_resolution_value(header),
                resolution_source=_resolution_source(header),
                exptime_s=_first_numeric(header, ("EXPTIME", "TEXPTIME", "EXPOSURE")),
                ra_deg=_first_numeric(
                    header,
                    ("RA", "RAOBJ", "PLUG_RA", "OBJRA", "ALPHA"),
                ),
                dec_deg=_first_numeric(
                    header,
                    ("DEC", "DECOBJ", "PLUG_DEC", "OBJDEC", "DELTA"),
                ),
                z_or_rv_kms=_velocity_from_metrics(radial_velocity, redshift),
                heliocentric_corr_applied=_heliocentric_applied(header),
                sn_median=sn_median,
                sn_per_pixel=sn_per_pixel,
                ebv_sfd=_first_numeric(header, ("EBV", "SFD_EBV", "EBMV", "E_BV", "A_V")),
                mask_frac_bad=_mask_fraction(header, quality_dict),
                quality_flags=json.dumps(quality_dict) if quality_dict else None,
                wavelength_frame=_wavelength_medium(header),
                flux_unit=_clean_str(_first_value(header, ("BUNIT", "FLUXUNIT"))),
                data_release=data_release or _clean_str(_first_value(header, ("RUN2D", "DR"))),
                retrieval_date=retrieval_iso,
                survey_wavelength_calibrated=survey_wavelength_calibrated,
                survey_flux_calibrated=survey_flux_calibrated,
                post_processing_steps=_clean_str(post_processing_steps),
                calibration_references="; ".join(calibration_references)
                if calibration_references
                else None,
            )
        )
    return rows


def collect_label_rows(
    paths: Iterable[Path | str],
    *,
    object_id_fn: Callable[[Path], str],
    label_catalog: LabelCatalog,
    include_unlabelled: bool = True,
) -> List[LabelTableRow]:
    """Build normalised label rows keyed by ``spec_id``."""

    rows: List[LabelTableRow] = []
    seen: Set[str] = set()
    for path_like in paths:
        path = Path(path_like)
        spec_id = object_id_fn(path)
        if spec_id in seen:
            continue
        seen.add(spec_id)
        label = label_catalog.get(spec_id)
        if label is None and not include_unlabelled:
            continue
        rows.append(
            LabelTableRow(
                spec_id=spec_id,
                class_mk=label.mk_class() if label is not None else None,
                teff_k=label.effective_temperature if label is not None else None,
                logg_cgs=label.surface_gravity_log_g if label is not None else None,
                fe_h_dex=label.metallicity_feh if label is not None else None,
                label_source=label.source if label is not None else None,
                label_quality=label.quality if label is not None else None,
            )
        )
    return rows


def build_spectra_manifest(
    paths: Iterable[Path | str],
    *,
    object_id_fn: Callable[[Path], str],
    survey: str | None = None,
    data_release: str | None = None,
    created_at: datetime | None = None,
    notes: str | Callable[[Path], str] | None = None,
) -> List[SpectraManifestRow]:
    """Create manifest rows describing spectral provenance for each FITS file."""

    created_iso = _normalise_timestamp(created_at)
    rows: List[SpectraManifestRow] = []
    for path_like in paths:
        path = Path(path_like)
        spectrum = load_spectrum(path)
        header: Mapping[str, object] = spectrum.meta.get("header", {})
        calibration = spectrum.calibration
        calibration_keywords: Mapping[str, object] = spectrum.meta.get("calibration", {}).get("keywords", {})

        crval1 = _numeric_or_keyword(header, calibration_keywords, "CRVAL1")
        cd1_1 = _numeric_or_keyword(header, calibration_keywords, "CD1_1")
        if cd1_1 is None:
            cd1_1 = _numeric_or_keyword(header, calibration_keywords, "CDELT1")
        coeff0 = _numeric_or_keyword(header, calibration_keywords, "COEFF0")
        coeff1 = _numeric_or_keyword(header, calibration_keywords, "COEFF1")

        grid = "log10" if calibration.is_logarithmic else "linear"
        if calibration.method not in {"log10-polynomial", "linear"}:
            grid = calibration.method

        note_value = notes(path) if callable(notes) else notes

        rows.append(
            SpectraManifestRow(
                spec_id=object_id_fn(path),
                survey=_clean_str(survey) or _clean_str(_first_value(header, ("SURVEY",))),
                data_release=data_release or _clean_str(_first_value(header, ("RUN2D", "DR"))),
                source_path=str(path),
                checksum=_sha256(path),
                wavelength_frame=_wavelength_medium(header),
                wavelength_grid=grid,
                crval1=crval1,
                cd1_1=cd1_1,
                coeff0=coeff0,
                coeff1=coeff1,
                flux_unit=_clean_str(_first_value(header, ("BUNIT", "FLUXUNIT"))),
                created_at=created_iso,
                notes=_clean_str(note_value),
            )
        )
    return rows


def write_metadata_table(
    rows: Sequence[SpectrumMetadataRow],
    output_path: Path | str,
    *,
    output_format: str | None = None,
) -> Path:
    """Persist *rows* to ``output_path`` in CSV or Parquet format."""

    return _write_rows([row.to_dict() for row in rows], output_path, output_format)


def write_manifest_table(
    rows: Sequence[SpectraManifestRow],
    output_path: Path | str,
    *,
    output_format: str | None = None,
) -> Path:
    """Persist manifest rows to disk in CSV or Parquet format."""

    return _write_rows([row.to_dict() for row in rows], output_path, output_format)


def write_label_table(
    rows: Sequence[LabelTableRow],
    output_path: Path | str,
    *,
    output_format: str | None = None,
) -> Path:
    """Persist label rows to disk in CSV or Parquet format."""

    return _write_rows([row.to_dict() for row in rows], output_path, output_format)


class PreprocessingLogger:
    """Collect preprocessing steps to support reproducibility exports."""

    def __init__(self) -> None:
        self._rows: MutableMapping[str, List[PreprocessingLogRow]] = {}

    def log_step(
        self,
        spec_id: str,
        step_name: str,
        *,
        params: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
        result: object | None = None,
        warnings: str | None = None,
        timestamp: datetime | None = None,
    ) -> PreprocessingLogRow:
        """Record a preprocessing step for *spec_id*."""

        params_json = _serialise_params(params)
        result_hash = _hash_result(result)
        timestamp_iso = _normalise_timestamp(timestamp)

        rows = self._rows.setdefault(spec_id, [])
        row = PreprocessingLogRow(
            spec_id=spec_id,
            step_order=len(rows) + 1,
            step_name=step_name,
            params_json=params_json,
            result_hash=result_hash,
            timestamp=timestamp_iso,
            warnings=_clean_str(warnings),
        )
        rows.append(row)
        return row

    def rows(self) -> List[PreprocessingLogRow]:
        """Return all recorded preprocessing steps sorted deterministically."""

        collected: List[PreprocessingLogRow] = []
        for spec_rows in self._rows.values():
            collected.extend(spec_rows)
        return sorted(collected, key=lambda row: (row.spec_id, row.step_order))


def write_preprocessing_log(
    rows: Sequence[PreprocessingLogRow],
    output_path: Path | str,
    *,
    output_format: str | None = None,
) -> Path:
    """Persist preprocessing log rows to disk in CSV or Parquet format."""

    return _write_rows([row.to_dict() for row in rows], output_path, output_format)


def _serialise_params(
    params: Mapping[str, object] | Sequence[tuple[str, object]] | None,
) -> str:
    if params is None:
        serialisable: Mapping[str, object] | Sequence[tuple[str, object]] = {}
    else:
        serialisable = params
    if isinstance(serialisable, Mapping):
        items = serialisable.items()
    else:
        items = serialisable
    normalised = {str(key): _normalise_param_value(value) for key, value in items}
    return json.dumps(normalised, sort_keys=True)


def _normalise_param_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


def _hash_result(result: object | None) -> str:
    digest = hashlib.sha256()
    if result is None:
        digest.update(b"")
    elif isinstance(result, np.ndarray):
        digest.update(result.tobytes())
    elif isinstance(result, (bytes, bytearray, memoryview)):
        digest.update(bytes(result))
    else:
        payload = json.dumps(_normalise_param_value(result), sort_keys=True, default=str)
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _normalise_retrieval_date(retrieval_date: datetime | None) -> str:
    if retrieval_date is None:
        retrieval_date = datetime.now(timezone.utc)
    if retrieval_date.tzinfo is None:
        retrieval_date = retrieval_date.replace(tzinfo=timezone.utc)
    return retrieval_date.date().isoformat()


def _normalise_timestamp(timestamp: datetime | None) -> str:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.isoformat()


def _first_value(header: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in header:
            return header[key]
    return None


def _first_numeric(header: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = header.get(key)
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    return None


def _first_int(header: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key not in header:
            continue
        value = header.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _resolution(header: Mapping[str, object]) -> tuple[float | None, str | None]:
    for key in ("SPECRES", "RESOLUTN", "R", "SPEC_R"):
        numeric = _to_float(header.get(key))
        if numeric is not None:
            return numeric, key
    return None, None


def _resolution_value(header: Mapping[str, object]) -> float | None:
    value, _ = _resolution(header)
    return value


def _resolution_source(header: Mapping[str, object]) -> str | None:
    _, source = _resolution(header)
    return source


def _clean_str(value: object | None) -> str | None:
    if value is None:
        return None
    string = str(value).strip()
    return string or None


def _wavelength_medium(header: Mapping[str, object]) -> str | None:
    value = header.get("VACUUM")
    if value is not None:
        return "vacuum" if _truthy(value) else "air"
    value = header.get("AIRORVAC")
    if isinstance(value, str):
        lower = value.strip().lower()
        if "vac" in lower:
            return "vacuum"
        if "air" in lower:
            return "air"
    value = header.get("WAVEFORM")
    if isinstance(value, str):
        lower = value.strip().lower()
        if "vac" in lower:
            return "vacuum"
        if "air" in lower:
            return "air"
    return None


def _quality_flags(header: Mapping[str, object]) -> Mapping[str, object]:
    quality: dict[str, object] = {}
    for key in (
        "ANDMASK",
        "ORMASK",
        "BADMASK",
        "QUALFLAG",
        "QUALITY",
        "DQ",
        "DQMASK",
        "FLAGS",
        "MASK",
    ):
        if key in header:
            quality[key] = header[key]
    return quality


def _mask_fraction(header: Mapping[str, object], quality: Mapping[str, object]) -> float | None:
    value = _first_numeric(
        header,
        (
            "MASKFRAC",
            "FRAC_BAD",
            "FRACBAD",
            "FRACMASK",
            "BADFRAC",
            "FRAC_BADPIX",
        ),
    )
    if value is not None:
        return value
    for key in ("MASKFRAC", "FRAC_BAD", "FRACBAD"):
        numeric = _to_float(quality.get(key))
        if numeric is not None:
            return numeric
    return None


def _sn_metrics(header: Mapping[str, object]) -> tuple[float | None, float | None]:
    sn_median = _first_numeric(
        header,
        (
            "SN_MEDIAN",
            "SN_MEDIAN_ALL",
            "SNR_MEDIAN",
            "SN_MEDIAN_TOT",
            "SNR",
        ),
    )
    sn_per_pixel = _first_numeric(
        header,
        (
            "SN_MEDIAN_ALL",
            "SNR_PER_PIXEL",
            "SN_PERPIX",
            "SNRPX",
            "SN_PERPIXEL",
        ),
    )
    return sn_median, sn_per_pixel


def _velocity_from_metrics(radial_velocity: float | None, redshift: float | None) -> float | None:
    if radial_velocity is not None:
        return radial_velocity
    if redshift is not None:
        return redshift * SPEED_OF_LIGHT_KMS
    return None


def _heliocentric_applied(header: Mapping[str, object]) -> bool | None:
    for key in ("HELIO", "HELIOCOR", "HC", "LHELIO", "HELIO_AP"):
        if key not in header:
            continue
        value = header[key]
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"t", "true", "y", "yes", "1", "applied"}:
                return True
            if lower in {"f", "false", "n", "no", "0", "not applied"}:
                return False
    return None


def _object_identifier(header: Mapping[str, object], label: object | None) -> str | None:
    candidate = _clean_str(
        _first_value(
            header,
            (
                "OBJID",
                "APOGEE_ID",
                "APOGEEID",
                "MANGAID",
                "TARGETID",
                "BESTOBJID",
                "DESIGNID",
                "OBJNAME",
                "OBJECT",
            ),
        )
    )
    if candidate:
        return candidate
    if label is not None:
        object_id = getattr(label, "object_id", None)
        if object_id:
            return object_id
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_or_keyword(
    header: Mapping[str, object], keywords: Mapping[str, object], key: str
) -> float | None:
    value = _to_float(keywords.get(key))
    if value is not None:
        return value
    if key in header:
        return _to_float(header.get(key))
    return None


def _infer_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit.lower()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(
        "Unable to infer metadata format from extension. Provide --metadata-format explicitly."
    )


def _write_csv(rows: Sequence[MutableMapping[str, object | None]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_parquet(rows: Sequence[MutableMapping[str, object | None]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Parquet output requires pandas to be installed."
        ) from exc
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)


def _write_rows(
    serialisable_rows: Sequence[MutableMapping[str, object | None]],
    output_path: Path | str,
    output_format: str | None,
) -> Path:
    if not serialisable_rows:
        raise ValueError("No rows provided for output")

    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    format_name = _infer_format(path, output_format)
    if format_name == "csv":
        _write_csv(serialisable_rows, path)
    elif format_name == "parquet":
        _write_parquet(serialisable_rows, path)
    else:  # pragma: no cover - safeguarded by _infer_format
        raise ValueError(f"Unsupported metadata format: {format_name}")
    return path


def _to_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "t", "true", "y", "yes"}
    return bool(value)


__all__ = [
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
]


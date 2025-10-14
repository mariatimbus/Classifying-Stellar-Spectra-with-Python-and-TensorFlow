"""Ground-truth label utilities sourced from authoritative SDSS catalogues."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class GroundTruthLabel:
    """Represents stellar ground-truth metadata for a single object."""

    object_id: str
    spectral_class: str | None = None
    luminosity_class: str | None = None
    effective_temperature: float | None = None
    surface_gravity_log_g: float | None = None
    metallicity_feh: float | None = None
    source: str | None = None
    quality: str | None = None

    def mk_class(self) -> str | None:
        """Return the Morgan–Keenan class string (e.g., ``G2V``) when possible."""

        if self.spectral_class and self.luminosity_class:
            spectral = self.spectral_class.strip()
            luminosity = self.luminosity_class.strip()
            return f"{spectral}{luminosity}".replace(" ", "") or None
        if self.spectral_class:
            return self.spectral_class.strip() or None
        return None

    def categorical_label(self) -> str | None:
        """Return a human-readable class label when available."""

        label = self.mk_class()
        if label is not None:
            return label
        return None


class LabelCatalog:
    """Container mapping object identifiers to ground-truth metadata."""

    def __init__(self, labels: Dict[str, GroundTruthLabel]):
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, object_id: str) -> bool:  # pragma: no cover - passthrough
        return object_id in self._labels

    def get(self, object_id: str) -> GroundTruthLabel | None:
        """Fetch the :class:`GroundTruthLabel` for *object_id* if present."""

        return self._labels.get(object_id)

    def class_label(self, object_id: str) -> str | None:
        """Return the stellar class label to use for *object_id* if available."""

        label = self.get(object_id)
        if label is None:
            return None
        return label.categorical_label()

    def labels(self) -> Iterable[GroundTruthLabel]:  # pragma: no cover - simple iterator
        return self._labels.values()

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        *,
        object_id_field: str,
        spectral_class_field: str | None = None,
        luminosity_class_field: str | None = None,
        effective_temperature_field: str | None = None,
        surface_gravity_field: str | None = None,
        metallicity_field: str | None = None,
        source_label: str | None = None,
        quality_field: str | None = None,
    ) -> "LabelCatalog":
        """Load a label catalogue from a CSV exported by SDSS catalogues."""

        csv_path = Path(path)
        labels: Dict[str, GroundTruthLabel] = {}
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {csv_path} does not contain headers")
            for row in reader:
                object_id = row.get(object_id_field)
                if not object_id:
                    continue
                object_id = object_id.strip()
                labels[object_id] = GroundTruthLabel(
                    object_id=object_id,
                    spectral_class=_clean_str(row.get(spectral_class_field)),
                    luminosity_class=_clean_str(row.get(luminosity_class_field)),
                    effective_temperature=_clean_float(row.get(effective_temperature_field)),
                    surface_gravity_log_g=_clean_float(row.get(surface_gravity_field)),
                    metallicity_feh=_clean_float(row.get(metallicity_field)),
                    source=source_label or csv_path.name,
                    quality=_clean_str(row.get(quality_field)),
                )
        return cls(labels)


def _clean_str(value: Optional[str]) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _clean_float(value: Optional[str]) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

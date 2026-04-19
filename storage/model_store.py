"""
storage/model_store.py  —  Subsea VFM Pro
==========================================
Owns all model persistence (save / load / existence checks).
Both pretrain.py and app.py use this — no more scattered pickle calls.

Public interface
----------------
ModelStore.save(ctrl, trained, calibrated, well_configs, meta)
ModelStore.load()  → payload dict | None
ModelStore.exists() → bool
ModelStore.meta()   → dict
ModelStore.delete()
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("pretrained_models")


class ModelStore:
    """
    Handles serialization of the FieldController and associated metadata
    to a local directory.

    File layout
    -----------
    <store_dir>/
        models.pkl          ← pickle of full payload
        training_meta.json  ← human-readable JSON summary
    """

    PKL_NAME  = "models.pkl"
    META_NAME = "training_meta.json"

    def __init__(self, store_dir: str | Path = _DEFAULT_DIR):
        self._dir  = Path(store_dir)
        self._pkl  = self._dir / self.PKL_NAME
        self._meta = self._dir / self.META_NAME

    # ── Existence ────────────────────────────────────────────────

    def exists(self) -> bool:
        return self._pkl.exists()

    # ── Save ─────────────────────────────────────────────────────

    def save(
        self,
        ctrl,
        trained:      dict,
        calibrated:   dict,
        well_configs: dict,
        data_source:  str  = "unknown",
        extra_meta:   dict | None = None,
    ) -> Path:
        """
        Serialize the FieldController and state to disk.

        Parameters
        ----------
        ctrl         : FieldController
        trained      : {well: bool}
        calibrated   : {well: bool}
        well_configs : WELL_CONFIGS dict used to build the controller
        data_source  : "volve" | "client" | label for metadata
        extra_meta   : any additional info to write to training_meta.json

        Returns
        -------
        Path to the saved .pkl file
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "ctrl":               ctrl,
            "trained":            trained,
            "calibrated":         calibrated,
            "active_well_configs": well_configs,
            "data_source":        data_source,
            "saved_at":           time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                               time.gmtime()),
        }

        # Pickle
        t0 = time.perf_counter()
        with open(self._pkl, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = time.perf_counter() - t0

        size_mb = self._pkl.stat().st_size / 1024 / 1024
        logger.info(f"ModelStore: saved {size_mb:.1f} MB in {elapsed:.1f}s → {self._pkl}")

        # Human-readable metadata
        meta = {
            "saved_at":    payload["saved_at"],
            "data_source": data_source,
            "pkl_size_mb": round(size_mb, 2),
            "wells": {
                w: {
                    "trained":    trained.get(w, False),
                    "calibrated": calibrated.get(w, False),
                }
                for w in ctrl.wells
            },
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(self._meta, "w") as f:
            json.dump(meta, f, indent=2)

        return self._pkl

    # ── Load ─────────────────────────────────────────────────────

    def load(self) -> dict | None:
        """
        Load and return the saved payload dict, or None if not found.

        Returns
        -------
        {ctrl, trained, calibrated, active_well_configs, data_source, saved_at}
        or None if file missing / corrupt.
        """
        if not self._pkl.exists():
            logger.info("ModelStore: no pretrained file found")
            return None

        try:
            with open(self._pkl, "rb") as f:
                payload = pickle.load(f)
            logger.info(f"ModelStore: loaded from {self._pkl}")
            return payload
        except Exception as e:
            logger.warning(f"ModelStore: load failed ({e}) — falling back to untrained")
            return None

    # ── Metadata ─────────────────────────────────────────────────

    def meta(self) -> dict:
        """Return the JSON metadata dict, or {} if not found."""
        if not self._meta.exists():
            return {}
        try:
            return json.loads(self._meta.read_text())
        except Exception:
            return {}

    # ── Delete ───────────────────────────────────────────────────

    def delete(self) -> bool:
        """Delete both pkl and meta files. Returns True if pkl was deleted."""
        deleted = False
        if self._pkl.exists():
            self._pkl.unlink()
            deleted = True
        if self._meta.exists():
            self._meta.unlink()
        logger.info("ModelStore: deleted pretrained models")
        return deleted

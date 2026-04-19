"""
services/vfm_service.py  —  Subsea VFM Pro
============================================
Single point of contact for all VFM predictions.
app.py, API routes, and future streaming clients all call this —
never touching FieldController or WellAgent directly.

Public interface
----------------
VFMService.predict(well, inputs)          → single-well result dict
VFMService.predict_field(telemetry)       → multi-well results + allocation
VFMService.well_names()                   → list of available wells
VFMService.is_trained(well)              → bool
VFMService.is_calibrated(well)           → bool
VFMService.status()                       → full status dict for UI
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VFMInputs:
    """
    Validated, typed inputs for a single VFM prediction.
    All pressure in psi, temperature in K.
    """
    p_res:         float           # Reservoir pressure [psi]
    whp_psi:       float           # Wellhead pressure [psi]
    wht_k:         float           # Wellhead temperature [K]
    wc:            float           # Water cut [fraction 0–1]
    gor:           float           # GOR [SCF/STB]
    bhp_psi:       float | None = None   # Downhole pressure (optional)
    p_choke_dn:    float | None = None   # Downstream choke pressure (optional)
    choke_size_64: float | None = None   # Choke size [1/64 in] (optional)

    def __post_init__(self):
        # Clamp to physically sensible ranges
        self.wc  = float(max(0.0, min(0.999, self.wc)))
        self.gor = float(max(50.0, min(5000.0, self.gor)))
        if self.p_res <= 0:
            raise ValueError(f"p_res must be positive, got {self.p_res}")
        if self.whp_psi <= 0:
            raise ValueError(f"whp_psi must be positive, got {self.whp_psi}")
        if self.wht_k < 200:
            raise ValueError(f"wht_k looks wrong ({self.wht_k} K) — did you pass °C?")


class VFMService:
    """
    Stateless service wrapper around FieldController.
    Holds a reference to the controller — does not own it.
    The controller is owned by ModelStore / session state.
    """

    def __init__(self, ctrl, trained: dict, calibrated: dict):
        """
        Parameters
        ----------
        ctrl       : FieldController instance (pre-loaded or freshly built)
        trained    : dict {well_name: bool}
        calibrated : dict {well_name: bool}
        """
        self._ctrl       = ctrl
        self._trained    = trained
        self._calibrated = calibrated

    # ── Query methods ────────────────────────────────────────────

    def well_names(self) -> list[str]:
        """Return sorted list of registered well names."""
        return sorted(self._ctrl.wells.keys())

    def is_trained(self, well: str) -> bool:
        return self._trained.get(well, False)

    def is_calibrated(self, well: str) -> bool:
        return self._calibrated.get(well, False)

    def any_trained(self) -> bool:
        return any(self._trained.values())

    def status(self) -> dict:
        """Full status dict for sidebar / monitoring UI."""
        wells = {}
        for name in self._ctrl.wells:
            wells[name] = {
                "trained":    self._trained.get(name, False),
                "calibrated": self._calibrated.get(name, False),
                "ml_source":  self._ctrl.wells[name].ml.meta.get("source", "none"),
                "n_records":  self._ctrl.wells[name].ml.meta.get("n", 0),
            }
        return {
            "n_wells":        len(self._ctrl.wells),
            "n_trained":      sum(1 for v in wells.values() if v["trained"]),
            "n_calibrated":   sum(1 for v in wells.values() if v["calibrated"]),
            "wells":          wells,
        }

    # ── Core prediction ──────────────────────────────────────────

    def predict(self, well: str, inputs: VFMInputs | dict) -> dict:
        """
        Run VFM for a single well.

        Parameters
        ----------
        well   : well name (must be in well_names())
        inputs : VFMInputs dataclass or plain dict with same keys

        Returns
        -------
        Result dict from WellAgent.run_vfm(), augmented with:
          _compute_ms : elapsed milliseconds
          _well       : well name (copy, convenient for downstream)
        """
        if well not in self._ctrl.wells:
            raise KeyError(f"Unknown well '{well}'. Available: {self.well_names()}")

        # Accept either dataclass or dict
        if isinstance(inputs, dict):
            inputs = VFMInputs(**{k: v for k, v in inputs.items()
                                  if k in VFMInputs.__dataclass_fields__})

        t0 = time.perf_counter()
        result = self._ctrl.wells[well].run_vfm(
            p_res         = inputs.p_res,
            whp_psi       = inputs.whp_psi,
            wht_k         = inputs.wht_k,
            wc            = inputs.wc,
            gor           = inputs.gor,
            bhp_psi       = inputs.bhp_psi,
            p_choke_dn    = inputs.p_choke_dn,
            choke_size_64 = inputs.choke_size_64,
        )
        result["_compute_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["_well"]       = well
        return result

    def predict_field(self, telemetry: dict[str, dict],
                      fiscal_oil_stbd: float = None) -> dict:
        """
        Run VFM for multiple wells simultaneously.

        Parameters
        ----------
        telemetry       : {well_name: {input dict}} — same keys as VFMInputs
        fiscal_oil_stbd : optional fiscal meter reading for reconciliation

        Returns
        -------
        {
          "results":  [per-well result dicts],
          "summary":  allocation/totals dict from FieldController.allocate()
          "errors":   {well: error_message} for any failed wells
        }
        """
        results = []
        errors  = {}

        for well, raw_inputs in telemetry.items():
            try:
                inp = VFMInputs(**{k: v for k, v in raw_inputs.items()
                                   if k in VFMInputs.__dataclass_fields__})
                results.append(self.predict(well, inp))
            except Exception as e:
                logger.error(f"[{well}] predict failed: {e}")
                errors[well] = str(e)
                results.append({
                    "well": well, "error": str(e),
                    "q_total_stbd": 0, "q_oil_stbd": 0,
                    "q_water_stbd": 0, "q_gas_mscfd": 0,
                })

        summary = self._ctrl.allocate(
            results,
            fiscal_oil_stbd=fiscal_oil_stbd
        )
        return {"results": results, "summary": summary, "errors": errors}

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        """
        Run VFM on a list of row dicts (from a CSV upload or streaming input).
        Each row must have a 'well' key plus VFMInputs fields.

        Returns list of result dicts with original row columns preserved.
        Rows with unknown wells or bad inputs get an 'error' key instead.
        """
        out = []
        for row in rows:
            well = row.get("well")
            if not well:
                out.append({**row, "error": "Missing 'well' column"})
                continue
            if well not in self._ctrl.wells:
                out.append({**row, "error": f"Unknown well '{well}'"})
                continue
            try:
                result = self.predict(well, row)
                out.append({**row, **result})
            except Exception as e:
                out.append({**row, "error": str(e)})
        return out

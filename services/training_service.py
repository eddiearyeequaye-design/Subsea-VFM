"""
services/training_service.py  —  Subsea VFM Pro
=================================================
Owns the training and calibration workflow.
app.py calls this instead of touching WellAgent / HybridML directly.

Public interface
----------------
TrainingService.train_all(ml_data)                → training report dict
TrainingService.train_well(well, df)              → per-well report dict
TrainingService.calibrate_all(well_tests)         → calibration report dict
TrainingService.calibrate_well(well, wt_df)       → per-well report dict
TrainingService.trained_wells()                   → list[str]
TrainingService.calibrated_wells()               → list[str]
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Wraps ML training and physics calibration for all wells.

    Parameters
    ----------
    ctrl       : FieldController — holds WellAgent instances
    trained    : mutable dict {well: bool} — updated in place
    calibrated : mutable dict {well: bool} — updated in place
    n_boot     : bootstrap iterations for HybridML
    n_est      : GBR estimators per bootstrap
    on_progress: optional callback(well, step, fraction) for UI progress
    """

    def __init__(
        self,
        ctrl,
        trained:    dict,
        calibrated: dict,
        n_boot: int = 20,
        n_est:  int = 80,
        on_progress: Callable | None = None,
    ):
        self._ctrl       = ctrl
        self._trained    = trained
        self._calibrated = calibrated
        self.n_boot      = n_boot
        self.n_est       = n_est
        self._progress   = on_progress or (lambda well, step, frac: None)

    # ── Status ───────────────────────────────────────────────────

    def trained_wells(self) -> list[str]:
        return [w for w, v in self._trained.items() if v]

    def calibrated_wells(self) -> list[str]:
        return [w for w, v in self._calibrated.items() if v]

    # ── Training ─────────────────────────────────────────────────

    def train_well(self, well: str, df: pd.DataFrame,
                   source: str = "field_data") -> dict:
        """
        Train ML for a single well on real field data.

        Parameters
        ----------
        well   : well name — must exist in FieldController
        df     : DataFrame with columns matching HybridML.FEATS + q_true
        source : label stored in model metadata

        Returns
        -------
        {status, well, n_records, elapsed_s, error?}
        """
        if well not in self._ctrl.wells:
            return {"status": "failed", "well": well,
                    "error": f"Well '{well}' not registered in FieldController"}

        t0 = time.perf_counter()
        try:
            w = self._ctrl.wells[well]
            w.ml.N_BOOT = self.n_boot
            w.ml.N_EST  = self.n_est
            w.train_ml_on_field_data(df, source=source)
            self._trained[well] = True
            elapsed = round((time.perf_counter() - t0), 1)
            logger.info(f"[{well}] Trained | {len(df)} records | {elapsed}s")
            return {
                "status":    "trained",
                "well":      well,
                "n_records": len(df),
                "n_boot":    self.n_boot,
                "n_est":     self.n_est,
                "elapsed_s": elapsed,
            }
        except Exception as e:
            logger.error(f"[{well}] Training failed: {e}")
            return {"status": "failed", "well": well, "error": str(e)}

    def train_all(self, ml_data: dict[str, pd.DataFrame],
                  source: str = "field_data") -> dict:
        """
        Train ML for all wells in ml_data.

        Parameters
        ----------
        ml_data : {well_name: DataFrame} — from data_loader.prepare_ml_dataset()
        source  : label stored in model metadata

        Returns
        -------
        {
          "results": {well: per-well report},
          "n_trained": int,
          "n_failed":  int,
          "total_elapsed_s": float,
        }
        """
        results   = {}
        n_trained = 0
        n_failed  = 0
        t_total   = time.perf_counter()
        n_wells   = len(ml_data)

        for i, (well, df) in enumerate(ml_data.items()):
            self._progress(well, "training", i / max(n_wells, 1))
            report = self.train_well(well, df, source=source)
            results[well] = report
            if report["status"] == "trained":
                n_trained += 1
            else:
                n_failed += 1

        self._progress("all", "complete", 1.0)
        return {
            "results":         results,
            "n_trained":       n_trained,
            "n_failed":        n_failed,
            "total_elapsed_s": round(time.perf_counter() - t_total, 1),
        }

    # ── Calibration ──────────────────────────────────────────────

    def calibrate_well(self, well: str, wt_df: pd.DataFrame,
                       max_tests: int = 12) -> dict:
        """
        Calibrate physics (friction_mult + pi_mult) for a single well.
        ML must be trained first.

        Parameters
        ----------
        well      : well name
        wt_df     : well-test DataFrame from data_loader.prepare_well_tests()
        max_tests : cap on number of well tests used (Nelder-Mead is fast but
                    more tests = slower convergence for marginal gain)

        Returns
        -------
        {status, well, mape_pct, friction_mult, pi_mult, elapsed_s, error?}
        """
        if well not in self._ctrl.wells:
            return {"status": "failed", "well": well,
                    "error": f"Well '{well}' not registered"}

        if not self._trained.get(well):
            return {"status": "skipped", "well": well,
                    "error": "ML not trained — train first then calibrate"}

        t0 = time.perf_counter()
        try:
            cal = self._ctrl.wells[well].calibrate(wt_df.head(max_tests))
            self._calibrated[well] = True
            elapsed = round(time.perf_counter() - t0, 1)
            logger.info(
                f"[{well}] Calibrated | MAPE={cal['mape_pct']}% "
                f"fm={cal['friction_mult']:.3f} pm={cal['pi_mult']:.3f}"
            )
            return {
                "status":        "calibrated",
                "well":          well,
                "mape_pct":      cal["mape_pct"],
                "friction_mult": cal["friction_mult"],
                "pi_mult":       cal["pi_mult"],
                "n_tests":       cal.get("n_tests", len(wt_df)),
                "elapsed_s":     elapsed,
            }
        except Exception as e:
            logger.error(f"[{well}] Calibration failed: {e}")
            return {"status": "failed", "well": well, "error": str(e)}

    def calibrate_all(self, well_tests: dict[str, pd.DataFrame],
                      max_tests: int = 12) -> dict:
        """
        Calibrate physics for all wells in well_tests.

        Parameters
        ----------
        well_tests : {well_name: DataFrame} — from data_loader.prepare_well_tests()
        max_tests  : per-well cap on number of well test rows

        Returns
        -------
        {
          "results": {well: per-well report},
          "n_calibrated": int,
          "n_failed":     int,
          "n_skipped":    int,
          "total_elapsed_s": float,
        }
        """
        results      = {}
        n_calibrated = 0
        n_failed     = 0
        n_skipped    = 0
        t_total      = time.perf_counter()
        n_wells      = len(well_tests)

        for i, (well, wt_df) in enumerate(well_tests.items()):
            self._progress(well, "calibrating", i / max(n_wells, 1))
            report = self.calibrate_well(well, wt_df, max_tests=max_tests)
            results[well] = report
            if report["status"] == "calibrated":
                n_calibrated += 1
            elif report["status"] == "skipped":
                n_skipped += 1
            else:
                n_failed += 1

        self._progress("all", "complete", 1.0)
        return {
            "results":         results,
            "n_calibrated":    n_calibrated,
            "n_failed":        n_failed,
            "n_skipped":       n_skipped,
            "total_elapsed_s": round(time.perf_counter() - t_total, 1),
        }

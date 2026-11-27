"""Reconstruction methods.

Built-in methods are addressable by short name; any other method can be
referenced as ``"package.module:ClassName"`` (see :func:`load_method`).
"""

from __future__ import annotations

import importlib

from .base import Reconstructor, finalize
from .baselines import DowerDipole, PopulationGaussian

REGISTRY: dict[str, str] = {
    "dower_dipole": "ecgrecon.methods.baselines:DowerDipole",
    "population_gaussian": "ecgrecon.methods.baselines:PopulationGaussian",
    "patient_regression": "ecgrecon.methods.patient_regression:PatientRegression",
    "beat_template": "ecgrecon.methods.beat_template:BeatTemplate",
    "spatiotemporal_gauss": "ecgrecon.methods.spatiotemporal_gauss:SpatioTemporalGaussian",
    "patient_fa": "ecgrecon.methods.patient_fa:PatientFactorAnalysis",
    "physio": "ecgrecon.methods.physio:PhysioReconstructor",
}


def load_method(spec: str, **kwargs) -> Reconstructor:
    """Instantiate a method from a short name or ``module:Class`` string."""
    spec = REGISTRY.get(spec, spec)
    if ":" not in spec:
        raise ValueError(f"Unknown method {spec!r}; known: {sorted(REGISTRY)}")
    mod, cls = spec.split(":", 1)
    return getattr(importlib.import_module(mod), cls)(**kwargs)


__all__ = ["Reconstructor", "finalize", "DowerDipole", "PopulationGaussian", "load_method", "REGISTRY"]

"""TemporalCorr-MetaNet Model Components.

This package contains all model components:
- LightweightBackbone: Shared CNN backbone (~160K params)
- PTCF: Parametric Temporal Correlation Fusion (~280K params)
- ATSM: Adaptive Temporal Similarity Matrices (~150K params)
- MCB: Meta-Learned Change Boundaries (~80K params)
- SimpleDecoder: Upsampling decoder (~80K params)
- TemporalCorrMetaNet: Complete model assembly (~750K params)
- MetaLearner: MAML-style meta-learning wrapper
"""

from .backbone import LightweightBackbone
from .ptcf import PTCF
from .atsm import ATSM
from .mcb import MCB
from .decoder import SimpleDecoder
from .temporalcorr_metanet import TemporalCorrMetaNet
from .meta_learner import MetaLearner

__all__ = [
    "LightweightBackbone",
    "PTCF",
    "ATSM",
    "MCB",
    "SimpleDecoder",
    "TemporalCorrMetaNet",
    "MetaLearner",
]

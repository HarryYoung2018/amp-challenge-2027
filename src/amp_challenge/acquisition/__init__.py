"""Mixed acquisition for internal labeling and policy replay."""

from amp_challenge.acquisition.counterfactual import (
    AdvantageGateResult,
    ContrastMoments,
    HistoricalInfluenceCredit,
    conservative_paired_advantage,
    gate_paired_advantages,
    gp_difference_kernel,
    historical_posterior_influence_credit,
    linear_paired_contrast,
    observed_contrast_noise_variance,
    sampled_paired_contrast,
)
from amp_challenge.acquisition.mixer import (
    CandidateBatch,
    MixedAcquisitionSelector,
    SelectionConfig,
    SelectionResult,
    StartSelectionEvidence,
)
from amp_challenge.acquisition.recommendation import (
    PosteriorMeanRecommendation,
    recommend_posterior_mean,
)
from amp_challenge.acquisition.soft_kg import (
    BeamJointSoftKGResult,
    BeamSearchDepthTrace,
    EvaluationBatch,
    GaussianSoftKG,
    JointSoftKGResult,
    PreferenceMeasure,
    SoftKGProblem,
    SoftKGResult,
    UpperChanceConstraint,
    maximum_exhaustive_pool_size,
    stable_soft_value,
)

__all__ = [
    "AdvantageGateResult",
    "BeamJointSoftKGResult",
    "BeamSearchDepthTrace",
    "CandidateBatch",
    "ContrastMoments",
    "EvaluationBatch",
    "GaussianSoftKG",
    "HistoricalInfluenceCredit",
    "JointSoftKGResult",
    "MixedAcquisitionSelector",
    "PosteriorMeanRecommendation",
    "PreferenceMeasure",
    "SelectionConfig",
    "SelectionResult",
    "SoftKGProblem",
    "SoftKGResult",
    "StartSelectionEvidence",
    "UpperChanceConstraint",
    "conservative_paired_advantage",
    "gate_paired_advantages",
    "gp_difference_kernel",
    "historical_posterior_influence_credit",
    "linear_paired_contrast",
    "maximum_exhaustive_pool_size",
    "observed_contrast_noise_variance",
    "recommend_posterior_mean",
    "sampled_paired_contrast",
    "stable_soft_value",
]

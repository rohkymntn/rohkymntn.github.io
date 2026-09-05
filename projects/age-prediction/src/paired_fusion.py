"""Guarded utility for FUTURE paired data, not evidence of validated fusion.

Training/calibration ages may be used by the caller after joining by exact,
study-qualified participant ID. This module never pairs observations by age.
"""
import pandas as pd

def align_paired_modalities(methylation: pd.DataFrame, microbiome: pd.DataFrame):
    """Return only verified same-person rows; reject missing, duplicate or disjoint IDs.

    Both indices must contain actual identifiers from a documented shared cohort.
    Matching text cannot itself verify biological identity; the caller must supply
    that provenance. Excludes outcome/age columns from the allowable feature frames.
    """
    for frame in (methylation,microbiome):
        if frame.index.hasnans or not frame.index.is_unique:
            raise ValueError('One row per verified participant is required.')
        if any(str(c).lower() in {'age','chronological_age','target','label'} for c in frame):
            raise ValueError('Outcome columns must be kept separate from features.')
    paired=methylation.index.intersection(microbiome.index,sort=False)
    if len(paired)==0:
        raise ValueError('No shared participants: unpaired modalities cannot be fused.')
    return methylation.loc[paired].copy(),microbiome.loc[paired].copy()

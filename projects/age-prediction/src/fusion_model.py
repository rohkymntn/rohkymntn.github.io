"""Late-fusion age prediction with independently fitted modality-specific regressors.

Equal weighting is a prespecified inference rule, not a weight learned from
unpaired subjects. Joint predictive accuracy requires a verified paired test set.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from paired_fusion import align_paired_modalities

@dataclass
class LateFusionAgeModel:
    blood_model: object
    skin_model: object
    blood_features: tuple
    skin_features: tuple
    blood_weight: float = 0.5

    def __post_init__(self):
        if not np.isfinite(self.blood_weight) or not 0 <= self.blood_weight <= 1:
            raise ValueError('Blood weight must be finite and between zero and one.')

    @staticmethod
    def _schema(frame, features):
        if not frame.columns.is_unique or set(frame.columns) != set(features):
            raise ValueError('Input feature identifiers must match the fitted schema exactly.')
        return frame.loc[:, list(features)]

    def predict_blood(self, frame):
        return self.blood_model.predict(self._schema(frame,self.blood_features))

    def predict_skin(self, frame):
        return self.skin_model.predict(self._schema(frame,self.skin_features))

    def predict_paired(self, blood, skin, *, pairing_provenance):
        """Indices must be documented same-person/same-visit specimen identifiers.

        Provenance is retained in the returned table. This declaration cannot
        replace source verification; identity must never be manufactured by age.
        """
        if not isinstance(pairing_provenance,str) or not pairing_provenance.strip():
            raise ValueError('Document the shared cohort and specimen linkage.')
        if set(blood.index) != set(skin.index):
            raise ValueError('Both modalities must contain exactly the same verified participants.')
        b,s=align_paired_modalities(blood,skin)
        pb,ps=self.predict_blood(b),self.predict_skin(s)
        return pd.DataFrame({'blood_age':pb,'skin_age':ps,
            'fused_age':self.blood_weight*pb+(1-self.blood_weight)*ps,
            'blood_weight':self.blood_weight,'pairing_provenance':pairing_provenance},index=b.index)

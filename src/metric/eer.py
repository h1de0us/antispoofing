from src.base.base_metric import BaseMetric
from src.metric.utils import compute_eer

class EER(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, bonafide_scores, other_scores, **batch):
        return compute_eer(bonafide_scores.detach().numpy(), other_scores.detach().numpy())[0]
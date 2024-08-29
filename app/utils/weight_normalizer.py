from typing import Dict


class WeightNormalizer:
    @staticmethod
    def normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else weights

    @staticmethod
    def to_k_values(weights: Dict[str, float], base_k: float = 60.0) -> Dict[str, float]:
        normalized = WeightNormalizer.normalize(weights)
        return {k: base_k / v if v > 0 else 0 for k, v in normalized.items()}

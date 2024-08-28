class WeightNormalizer:
    @staticmethod
    def normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}

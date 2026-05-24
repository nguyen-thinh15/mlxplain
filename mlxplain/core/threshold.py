"""Binary classification threshold logic."""


def classify(
    probability: float,
    threshold: float = 0.5,
    positive_label: str = "Positive",
    negative_label: str = "Negative",
) -> str:
    """Classify a probability into a label based on a threshold."""
    return positive_label if probability >= threshold else negative_label

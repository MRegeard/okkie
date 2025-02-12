__all__ = ["sum_models"]


def sum_models(models, model_type):
    """Sum all models."""
    summ = getattr(models[0], f"{model_type}_model")
    for m in models[1:]:
        summ += getattr(m, f"{model_type}_model")
    return summ

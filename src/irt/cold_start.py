

from irt.interfaces import ModelProfile


def simple_cold_start_theta(model: ModelProfile) -> float:
    """Map model metadata to a prior ability theta.

    Heuristic: small sizes → lower theta, large → higher. Family can shift slightly.
    """
    theta = 0.0
    size = (model.model_size_params or "").upper()
    if "70B" in size or "80B" in size or "100B" in size:
        theta += 1.0
    elif "13B" in size or "20B" in size:
        theta += 0.5
    elif "7B" in size or "8B" in size:
        theta += 0.2
    fam = (model.model_family or "").lower()
    if fam in {"familyx", "llama", "mistral"}:
        theta += 0.1
    return theta



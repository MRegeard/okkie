from gammapy.modeling.models import MODEL_REGISTRY

from okkie import link
from okkie.modeling.models import SourceModel


def test_link():
    link("gammapy")
    assert MODEL_REGISTRY.get_cls("SourceModel") == SourceModel
    assert len(MODEL_REGISTRY) == 4
    link("gammapy")
    assert len(MODEL_REGISTRY) == 4

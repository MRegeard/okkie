import numpy as np
import pytest
from numpy.testing import assert_allclose

from okkie.modeling.models import (
    AsymmetricGaussianNormPhaseModel,
    AsymmetricGaussianPhaseModel,
    AsymmetricLorentzianNormPhaseModel,
    AsymmetricLorentzianPhaseModel,
    ConstantPhaseModel,
    GaussianNormPhaseModel,
    GaussianPhaseModel,
    LorentzianNormPhaseModel,
    LorentzianPhaseModel,
    ScalePhaseModel,
    TemplatePhaseModel,
)

TEST_MODELS = [
    dict(
        name="constant",
        model=ConstantPhaseModel(const=5),
        val_at_05=5,
        integral_0_1=5,
    ),
    dict(
        name="gaussian",
        model=GaussianPhaseModel(
            amplitude=10,
            mean=0.3,
            sigma=0.1,
        ),
        val_at_05=1.3533528,
        integral_0_1=2.5066282746,
    ),
    dict(
        name="lorentzian",
        model=LorentzianPhaseModel(
            amplitude=12,
            sigma=0.1,
            mean=0.7,
        ),
        val_at_05=2.7022563,
        integral_0_1=3.7262218890,
    ),
    dict(
        name="asymetric-gaussian",
        model=AsymmetricGaussianPhaseModel(
            amplitude=25,
            mean=0.45,
            sigma_1=0.05,
            sigma_2=0.085,
        ),
        val_at_05=21.028222,
        integral_0_1=4.2299352134,
    ),
    dict(
        name="asymetric-lorentzian",
        model=AsymmetricLorentzianPhaseModel(
            amplitude=7,
            mean=0.1,
            sigma_1=0.01,
            sigma_2=0.1,
        ),
        val_at_05=0.46704696,
        integral_0_1=1.1975126460,
    ),
]


TEST_MODELS.append(
    dict(
        name="comp1",
        model=TEST_MODELS[3]["model"] + TEST_MODELS[4]["model"],
        val_at_05=21.495269,
        integral_0_1=5.4101274349,
    )
)


TEST_MODELS.append(
    dict(
        name="scale",
        model=ScalePhaseModel(model=TEST_MODELS[1]["model"], scale=2),
        val_at_05=2 * 1.3533528,
        integral_0_1=2 * 2.5066282746310007,
    )
)


@pytest.mark.parametrize("phase", TEST_MODELS)
def test_models(phase):
    model = phase["model"]

    for p in model.parameters:
        assert p.type == "phase"

    phi = 0.5
    value = model(phi)
    assert_allclose(value, phase["val_at_05"])

    phi_min = 0
    phi_max = 1
    assert_allclose(
        model.integral(phi_min, phi_max),
        phase["integral_0_1"],
    )

    assert_allclose(model(0), model(1))


TEST_NORM_MODELS = [
    dict(
        name="gaussian-norm",
        model=GaussianNormPhaseModel(mean=0.3, sigma=0.1),
        val_at_05=0.5399096651,
    ),
    dict(
        name="lorentzian-norm",
        model=LorentzianNormPhaseModel(mean=0.7, sigma=0.1),
        val_at_05=0.7462730461,
    ),
    dict(
        name="asymetric-gaussian-norm",
        model=AsymmetricGaussianNormPhaseModel(
            mean=0.45,
            sigma_1=0.05,
            sigma_2=0.085,
        ),
        val_at_05=4.9712870346,
    ),
    dict(
        name="asymetric-lorentzian-norm",
        model=AsymmetricLorentzianNormPhaseModel(
            mean=0.1,
            sigma_1=0.01,
            sigma_2=0.1,
        ),
        val_at_05=0.3957378508,
    ),
]


@pytest.mark.parametrize("phase", TEST_NORM_MODELS)
def test_norm_models(phase):
    model = phase["model"]

    for p in model.parameters:
        assert p.type == "phase"

    phi = 0.5
    value = model(phi)
    assert_allclose(value, phase["val_at_05"])

    phi_min = 0
    phi_max = 1
    assert_allclose(model.integral(phi_min, phi_max), 1.0)

    assert_allclose(model(0), model(1))


NORM_MODELS = [
    (
        GaussianPhaseModel(amplitude=10, mean=0.3, sigma=0.1),
        GaussianNormPhaseModel,
    ),
    (
        LorentzianPhaseModel(amplitude=12, sigma=0.1, mean=0.7),
        LorentzianNormPhaseModel,
    ),
    (
        AsymmetricGaussianPhaseModel(
            amplitude=25, mean=0.45, sigma_1=0.05, sigma_2=0.085
        ),
        AsymmetricGaussianNormPhaseModel,
    ),
    (
        AsymmetricLorentzianPhaseModel(
            amplitude=7, mean=0.1, sigma_1=0.01, sigma_2=0.1
        ),
        AsymmetricLorentzianNormPhaseModel,
    ),
]


@pytest.mark.parametrize("model, norm_cls", NORM_MODELS)
def test_to_norm_returns_norm(model, norm_cls):
    norm_model = model.to_norm()
    assert isinstance(norm_model, norm_cls)
    assert_allclose(norm_model.integral(0, 1), 1.0)


def test_template():
    phases = np.linspace(0, 1, 100)
    model = GaussianPhaseModel(mean=0.5, sigma=0.1, amplitude=10)
    values = model(phases)
    template = TemplatePhaseModel(phases, values)

    assert_allclose(template.phase_shift.value, 0.0)
    assert template.phase_shift.frozen is False

    assert_allclose(template(0.6), 6.0641672)

    template.phase_shift.value = 0.1
    assert_allclose(template(0.5), 6.0641672)

    norm = template.to_norm()
    assert_allclose(norm.integral(0, 1), 1.0)

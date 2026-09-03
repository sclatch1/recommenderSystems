from src.algorithm import PPAC_BPRMF
from src.pipeline import ppac


def test_ppac_bprmf_accepts_ppac_default_params():
    """Regression test: ppac()'s default params dict used to include
    lambda_h/lambda_w, which collided with the lambda_h=0.0/lambda_w=0.0
    PPAC_BPRMF.__init__ hardcodes internally (it always forces both to 0 -
    see algorithm.py), raising 'got multiple values for keyword argument
    lambda_h' when PipelineBuilder instantiated it."""
    params, _ = ppac(include_hyperparams=False)

    model = PPAC_BPRMF(**params)

    assert model.lambda_h == 0.0
    assert model.lambda_w == 0.0

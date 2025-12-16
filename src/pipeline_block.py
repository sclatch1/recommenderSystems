from recpack.pipelines import PipelineBuilder
from src.my_bprmf import MyBPRMF, PPAC_BPRMF
from recpack.pipelines import ALGORITHM_REGISTRY
from recpack.scenarios.splitters import FractionInteractionSplitter


class MyPipeLine():
    def __init__(self):
        self.pipeline_builder = PipelineBuilder()

    
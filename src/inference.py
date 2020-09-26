import pandas as pd

from src.feature_generator import FeatureGenerator


class Inference:

    def __init__(self):
        pass

    def pipeline(self):
        f = FeatureGenerator()
        features = f.generator()
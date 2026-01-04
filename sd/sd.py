import pysubdisc
import pandas as pd

class SubgroupDiscovery:
    def __init__(self, min_coverage=10, max_coverage=100, depth=2):
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.depth = depth

    def run(self, df: pd.DataFrame, target: str):
        sd = pysubdisc.singleNumericTarget(df, target)

        sd.searchDepth = self.depth
        sd.minimumCoverage = self.min_coverage
        sd.maximumCoverage = self.max_coverage
        sd.numericStrategy = 'NUMERIC_BEST'
        sd.qualityMeasureMinimum = 0.0

        sd.run(verbose=False)
        return sd
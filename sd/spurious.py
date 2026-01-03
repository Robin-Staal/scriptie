import pysubdisc

class SwapRandomisationSignificance:
    def __init__(self, n_runs=100, alpha=0.05):
        self.n_runs = n_runs
        self.alpha = alpha

    def run(self, df, target):
        sd = pysubdisc.singleNumericTarget(df, target)

        sd.randomisationType = 'SWAP'
        sd.numberOfRandomisations = self.n_runs
        sd.significanceLevel = self.alpha

        sd.run(verbose=False)

        return sd.qualityMeasureMinimum
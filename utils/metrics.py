import numpy as np

def consensus_degree(opinions):
        """Compute consensus degree from an array of opinions.

        Returns a score in [0, 1], where 1 indicates perfect consensus.
        """
        mean_opinion = np.mean(opinions)
        absolute_deviations = np.abs(opinions - mean_opinion)
        total_deviation = np.sum(absolute_deviations)
        return 1 - (total_deviation / len(opinions))
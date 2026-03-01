from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
import numpy as np


def compute_similarity(v1, v2, metric='cosine'):
        v1 = np.array(v1).reshape(1, -1)
        v2 = np.array(v2).reshape(1, -1)

        if metric == 'cosine':
            return cosine_similarity(v1, v2)[0][0]
        
        elif metric == 'euclidean':
            dist = euclidean(v1.flatten(), v2.flatten())
            return 1 / (1 + dist)

        elif metric == 'manhattan':
            dist = cityblock(v1.flatten(), v2.flatten())
            return 1 / (1 + dist)
        
        elif metric == 'pearson':
            corr, _ = pearsonr(v1.flatten(), v2.flatten())
            return corr

        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
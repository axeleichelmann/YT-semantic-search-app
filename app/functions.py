import requests
import json
from sentence_transformers import SentenceTransformer, util

import pandas as pd
import numpy as np




def returnSearchResults(query : str, 
                        df_embedding : pd.DataFrame,
                        model : SentenceTransformer,
                        dist : util,
                        threshold : float=0.5, top_k : int=5) -> np.ndarray:
    
    # Embed Query
    query_embedding = model.encode(query).reshape(1,-1)

    # Compute distance between query and titles
    dist_arr = dist(df_embedding.values, query_embedding)

    # Identify videos that are close to query based on threshold
    idx_above_threshold = np.argwhere(dist_arr.flatten()>threshold).flatten()
    idx_sorted = np.argsort((-dist_arr[idx_above_threshold]), axis=0).flatten()

    # return indexes of top k search results
    return idx_above_threshold[idx_sorted][:top_k]



import requests
import json
from sentence_transformers import SentenceTransformer, util

import pandas as pd
import numpy as np




def getVideoRecords(response: requests.models.Response) -> list:
    """
        Function to extract YouTube video data from GET request response
    """

    video_record_list = []
    
    for raw_item in json.loads(response.text)['items']:
    
        # only execute for youtube videos
        if raw_item['id']['kind'] != "youtube#video":
            continue
        
        video_record = {}
        video_record['video_id'] = raw_item['id']['videoId']
        video_record['datetime'] = raw_item['snippet']['publishedAt']
        video_record['title'] = raw_item['snippet']['title']
        
        video_record_list.append(video_record)

    return video_record_list






def extract_text(transcript: list) -> str:
    """
        Function to extract text from transcript dictionary
    """
    
    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)






def returnVideoID_index(df: pd.DataFrame, df_eval: pd.DataFrame, query_n: int) -> int:
    """
        Function to return the index of a dataframe corresponding to the nth row in evaluation dataframe
    """

    return [i for i in range(len(df)) if df['video_id'][i]==df_eval['video_id'][query_n]][0]





def evalTrueRankings(dist_arr_sorted: np.ndarray, df: pd.DataFrame, df_eval: pd.DataFrame) -> np.ndarray:
    """
        Function to return "true" video ID rankings for each evaluation query
    """
    
    # intialize array to store rankings of "correct" search result
    true_rank_arr = np.empty((1, dist_arr_sorted.shape[1]))
    
    # evaluate ranking of correct result for each query
    for query_n in range(dist_arr_sorted.shape[1]):
    
        # return "true" video ID's in df
        video_id_idx = returnVideoID_index(df, df_eval, query_n)
        
        # evaluate the ranking of the "true" video ID
        true_rank = np.argwhere(dist_arr_sorted[:,query_n]==video_id_idx)[0][0]
        
        # store the "true" video ID's ranking in array
        true_rank_arr[0,query_n] = true_rank

    return true_rank_arr




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











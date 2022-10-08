from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def get_similarity (data, kind='user'):
  if kind == 'user':
    similarity = cosine_similarity(data.T, data.T)
    similarity = pd.DataFrame(similarity, columns = data.columns, index = data.columns)
  elif kind == 'item':
    similarity = cosine_similarity(data, data)
    similarity = pd.DataFrame(similarity, columns = data.index, index = data.index)
  return similarity

def calculate_from_input (input_dict, data, kind = 'user'):

  # Explain about the parameters:
  # 'input_dict' is used for the input dictionary
  # 'data' is for the user-item matrix that is used for calculating (train, test sets)
  # 'kind' is for the kind of CF you want to calculate, item-based or user-based
  
  # Convert input dictionary into dataframe
  blank_df = pd.DataFrame(index = data.index)
  input_df = blank_df.join(pd.DataFrame.from_dict(input_dict, orient='index'))
  input_df = input_df.fillna(0)

  # Item-based filtering
  if (kind == 'item'):
    similarity = get_similarity(data, kind='item')
    output_df = similarity.dot(input_df)/np.array([np.abs(similarity).sum(axis=1)]).T

  # User-based filtering
  elif (kind == 'user'):
    similarity = pd.DataFrame(cosine_similarity(input_df.T, data.T))
    output_df = data.to_numpy().dot(similarity.to_numpy().T)/np.array([np.abs(similarity).sum(axis=1)])
    output_df = pd.DataFrame(output_df, index = data.index)

  # Not change any given grade
  for i in input_df.index:
    if input_df.loc[i][0] != 0:
      output_df.drop(i,inplace=True)

  output_dict = output_df.to_dict()
  output_dict = output_dict[0]
  output_dict = sorted(output_dict.items(), key=lambda item: item[1],reverse=True)  
  return output_dict
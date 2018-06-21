
import numpy as np
import pandas as pd


SUBJECT_COLUMN_NAME = 'SUBJECT'
OBJECT_COLUMN_NAME = 'OBJECT'
RATING_COLUMN_NAME = 'RATING'

def ratings_table_to_matrix(ratings_df, resort=False):
    if RATING_COLUMN_NAME in ratings_df.columns:
        rating = lambda row: float(row[RATING_COLUMN_NAME])
    else:
        rating = lambda row: 1.0
    
    subjects = ratings_df[SUBJECT_COLUMN_NAME].unique()
    objects  = ratings_df[OBJECT_COLUMN_NAME].unique()

    if resort:
        subjects = np.sort(subjects)
        objects  = np.sort(objects)

    matrix_df = pd.DataFrame(0.0, index=subjects, columns=objects)
    for _, entry in ratings_df.iterrows():
        row = entry[SUBJECT_COLUMN_NAME]
        col = entry[OBJECT_COLUMN_NAME]
        matrix_df[col][row] = rating(entry)  # The row,col ordering in the DF API is unconventional

    return matrix_df

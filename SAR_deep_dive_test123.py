# SAR = Collaborative Filtering

import sys
import logging
import scipy
import numpy as np
import pandas as pd

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.sar import SAR
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

#%%
# Top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"
#%%
# set log level to INFO
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["UserId", "MovieId", "Rating", "Timestamp"],
    title_col="Title",
)

# Convert the float precision to 32-bit in order to reduce memory consumption
data["Rating"] = data["Rating"].astype(np.float32)

print(data.head())

# Search for user with UserId=54
user_selected_df = data[data['UserId'] == 943]

print("selected user default: ")
# Display the result
print(user_selected_df.head(10))

header = {
    "col_user": "UserId",
    "col_item": "MovieId",
    "col_rating": "Rating",
    "col_timestamp": "Timestamp",
    "col_prediction": "Prediction",
}
#%%
train, test = python_stratified_split(
    data, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"], seed=42
)

model = SAR(
    similarity_type="jaccard",
    time_decay_coefficient=30,
    time_now=None,
    timedecay_formula=True,
    **header
)
#%%
model.fit(train)

top_k = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

top_k_with_titles = top_k.join(
    data[["MovieId", "Title"]].drop_duplicates().set_index("MovieId"),
    on="MovieId",
    how="inner",
).sort_values(by=["UserId", "Prediction"], ascending=False)

print(top_k_with_titles.head(10))

# Now let's look at the results for a specific user
user_id = 42

# Search for user with UserId=54
user_selected_df = top_k_with_titles[top_k_with_titles['UserId'] == user_id]

print("selected user prediction: ")
# Display the result
print(user_selected_df.head(10))

# ground_truth = top_k_with_titles[top_k_with_titles["userID"] == user_id].sort_values(
#     by="rating", ascending=False
# )[:TOP_K]
# prediction = model.recommend_k_items(
#     pd.DataFrame(dict(userID=[user_id])), remove_seen=True
# )
# df = pd.merge(ground_truth, prediction, on=["userID", "MovieId"], how="left")
# print(df.head(10))


# all ranking metrics have the same arguments
args = [test, top_k]
kwargs = dict(
    col_user="UserId",
    col_item="MovieId",
    col_rating="Rating",
    col_prediction="Prediction",
    relevancy_method="top_k",
    k=TOP_K,
)

eval_map = map_at_k(*args, **kwargs)
eval_ndcg = ndcg_at_k(*args, **kwargs)
eval_precision = precision_at_k(*args, **kwargs)
eval_recall = recall_at_k(*args, **kwargs)
#%%
print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')


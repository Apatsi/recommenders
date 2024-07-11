import sys
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.utils.python_utils import binarize
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.models.sar import SAR
from recommenders.evaluation.python_evaluation import (
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE
)

print(data)

# Convert the float precision to 32-bit in order to reduce memory consumption
data["rating"] = data["rating"].astype(np.float32)

print(data.head())

train, test = python_stratified_split(data, ratio=0.75, col_user="userID", col_item="itemID", seed=42)

print("""
Train:
Total Ratings: {train_total}
Unique Users: {train_users}
Unique Items: {train_items}

Test:
Total Ratings: {test_total}
Unique Users: {test_users}
Unique Items: {test_items}
""".format(
    train_total=len(train),
    train_users=len(train['userID'].unique()),
    train_items=len(train['itemID'].unique()),
    test_total=len(test),
    test_users=len(test['userID'].unique()),
    test_items=len(test['itemID'].unique()),
))

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard",
    time_decay_coefficient=30,
    timedecay_formula=True,
    normalize=True
)

with Timer() as train_time:
    model.fit(train)

print("Took {} seconds for training.".format(train_time.interval))

with Timer() as test_time:
    top_k = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

print("Took {} seconds for prediction.".format(test_time.interval))

print(top_k.head())

# Ranking metrics
eval_map = map(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)
eval_ndcg = ndcg_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)
eval_precision = precision_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)
eval_recall = recall_at_k(test, top_k, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)

# Rating metrics
eval_rmse = rmse(test, top_k, col_user="userID", col_item="itemID", col_rating="rating")
eval_mae = mae(test, top_k, col_user="userID", col_item="itemID", col_rating="rating")
eval_rsquared = rsquared(test, top_k, col_user="userID", col_item="itemID", col_rating="rating")
eval_exp_var = exp_var(test, top_k, col_user="userID", col_item="itemID", col_rating="rating")

positivity_threshold = 2
test_bin = test.copy()
test_bin["rating"] = binarize(test_bin["rating"], positivity_threshold)

top_k_prob = top_k.copy()
top_k_prob["prediction"] = minmax_scale(top_k_prob["prediction"].astype(float))

eval_logloss = logloss(
    test_bin, top_k_prob, col_user="userID", col_item="itemID", col_rating="rating"
)

print("Model:\t",
      "Top K:\t%d" % TOP_K,
      "MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall,
      "RMSE:\t%f" % eval_rmse,
      "MAE:\t%f" % eval_mae,
      "R2:\t%f" % eval_rsquared,
      "Exp var:\t%f" % eval_exp_var,
      "Logloss:\t%f" % eval_logloss,
      sep='\n')

# Now let's look at the results for a specific user
user_id = 54

ground_truth = test[test["userID"] == user_id].sort_values(
    by="rating", ascending=False
)[:TOP_K]
prediction = model.recommend_k_items(
    pd.DataFrame(dict(userID=[user_id])), remove_seen=True
)
df = pd.merge(ground_truth, prediction, on=["userID", "itemID"], how="left")
print(df.head(10))

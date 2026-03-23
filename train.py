import os
from src.config import *
from src.data_loader.preprocessor import load_user_data, load_item_data, split_train_val
from src.recall.item_cf import ItemCF
from src.recall.item2vec import Item2VecRecall
from src.features.build_features import build_ranking_dataset
from src.ranking.lgbm_ranker import LGBMRanker
import joblib

def main():
    print("Loading Data")
    df_A = load_user_data(os.path.join(RAW_DATA_DIR, 'Antai_hackathon_train.csv'))
    item_df = load_item_data(os.path.join(RAW_DATA_DIR, 'Antai_hackathon_attr.csv'))

    print("Splitting Train/Val")
    df_train_hist, df_val_label = split_train_val(df_A)

    user_hist_dict = df_train_hist.groupby('buyer_admin_id')['item_id'].apply(list).to_dict()
    user_label_dict = df_val_label.set_index('buyer_admin_id')['item_id'].to_dict()

    item_cf_path = os.path.join(MODEL_DIR, 'item_cf.pkl')
    item2vec_path = os.path.join(MODEL_DIR, 'item2vec.pkl')
    lgbm_ranker_path = os.path.join(MODEL_DIR, 'lgbm_ranker.pkl')

    print("Training Recall Models")
    if os.path.exists(item_cf_path):
        item_cf = joblib.load(item_cf_path)
    else:
        item_cf = ItemCF()
        item_cf.fit(df_train_hist)
        joblib.dump(item_cf, item_cf_path)

    if os.path.exists(item2vec_path):
        item2vec = joblib.load(item2vec_path)
    else:
        item2vec = Item2VecRecall(vector_size=I2V_VECTOR_SIZE, window=I2V_WINDOW)
        item2vec.fit(df_train_hist)
        joblib.dump(item2vec, item2vec_path)

    print("Building Ranking Dataset")
    df_rank_train = build_ranking_dataset(user_hist_dict, item_cf, item2vec, item_df, labels=user_label_dict)

    print("Training Ranking Model")
    if os.path.exists(lgbm_ranker_path):
        ranker = joblib.load(lgbm_ranker_path)
    else:
        ranker = LGBMRanker(LGBM_PARAMS)
        ranker.fit(df_rank_train)
        ranker.save(lgbm_ranker_path)


if __name__ == "__main__":
    main()
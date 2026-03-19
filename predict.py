import os
import pandas as pd
from src.config import *
from src.data_loader.preprocessor import load_user_data, load_item_data
from src.features.build_features import build_ranking_dataset
from src.ranking.lgbm_ranker import LGBMRanker
import joblib

def main():
    print("Loading dataset Item Data Models")
    df_B = load_user_data(os.path.join(RAW_DATA_DIR, 'dianshang_test.csv'))
    item_df = load_item_data(os.path.join(RAW_DATA_DIR, 'Antai_hackathon_attr.csv'))
    
    item_cf = joblib.load(os.path.join(MODEL_DIR, 'item_cf.pkl'))
    item2vec = joblib.load(os.path.join(MODEL_DIR, 'item2vec.pkl'))
    ranker = LGBMRanker(LGBM_PARAMS)
    ranker.load(os.path.join(MODEL_DIR, 'lgbm_ranker.pkl'))
    
    user_hist_dict_B = df_B.groupby('buyer_admin_id')['item_id'].apply(list).to_dict()
    
    print("Building Features")
    df_rank_test = build_ranking_dataset(user_hist_dict_B, item_cf, item2vec, item_df, labels=None)
    
    print("Predicting Scores")
    df_rank_test['predict_score'] = ranker.predict(df_rank_test)
    
    print("Generating Top 30 Recommendations")
    df_rank_test = df_rank_test.sort_values(['buyer_admin_id', 'predict_score'], ascending=[True, False])
    top30_result = df_rank_test.groupby('buyer_admin_id').head(30)

    top30_items = top30_result.groupby('buyer_admin_id')['item_id'].apply(list)

    result_data = []
    for user_id, items in top30_items.items():
        while len(items) < 30:
            items.append(0)
        result_data.append([user_id] + items[:30])

    columns = ['buyer_admin_id'] + [f'predict {i}' for i in range(1, 31)]

    submission = pd.DataFrame(result_data, columns=columns)

    out_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
    submission.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
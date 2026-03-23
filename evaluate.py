import os
import pandas as pd
import numpy as np
from collections import defaultdict
from src.config import *
from src.data_loader.preprocessor import load_user_data, load_item_data, split_train_val
from src.recall.item_cf import ItemCF
from src.recall.item2vec import Item2VecRecall
from src.features.build_features import build_ranking_dataset
from src.ranking.lgbm_ranker import LGBMRanker
import joblib


def compute_ndcg_at_k(relevant_items, recommended_items, k):
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)

    if not relevant_items:
        return 0.0

    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_recall_at_k(relevant_items, recommended_items, k):
    recommended_topk = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0
    return len(recommended_topk & relevant_set) / len(relevant_set)


def compute_hit_rate_at_k(relevant_items, recommended_items, k):
    recommended_topk = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    return 1.0 if len(recommended_topk & relevant_set) > 0 else 0.0


def compute_mrr(relevant_items, recommended_items):
    relevant_set = set(relevant_items)
    for i, item in enumerate(recommended_items):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_model(df_val_hist, df_val_label, item_cf, item2vec, ranker, item_df, top_k_list=[10, 30]):
    print("Building validation features")
    user_hist_dict = df_val_hist.groupby('buyer_admin_id')['item_id'].apply(list).to_dict()
    user_label_dict = df_val_label.set_index('buyer_admin_id')['item_id'].to_dict()

    df_rank_val = build_ranking_dataset(user_hist_dict, item_cf, item2vec, item_df, labels=None)
    df_rank_val['predict_score'] = ranker.predict(df_rank_val)

    df_rank_val = df_rank_val.sort_values(['buyer_admin_id', 'predict_score'], ascending=[True, False])

    results = {f'NDCG@{k}': [] for k in top_k_list}
    results.update({f'Recall@{k}': [] for k in top_k_list})
    results.update({f'HitRate@{k}': [] for k in top_k_list})
    results['MRR'] = []

    user_count = 0
    for user_id in user_label_dict.keys():
        if user_id not in user_hist_dict:
            continue

        user_preds = df_rank_val[df_rank_val['buyer_admin_id'] == user_id]['item_id'].tolist()
        relevant_items = [user_label_dict[user_id]]

        for k in top_k_list:
            results[f'NDCG@{k}'].append(compute_ndcg_at_k(relevant_items, user_preds, k))
            results[f'Recall@{k}'].append(compute_recall_at_k(relevant_items, user_preds, k))
            results[f'HitRate@{k}'].append(compute_hit_rate_at_k(relevant_items, user_preds, k))
        results['MRR'].append(compute_mrr(relevant_items, user_preds))

        user_count += 1

    metrics = {}
    for metric_name, values in results.items():
        metrics[metric_name] = np.mean(values) if values else 0.0

    return metrics, user_count


def evaluate_recall_only(df_val_hist, df_val_label, item_cf, item2vec, top_k_list=[10, 30]):
    print("Quick evaluation")
    user_hist_dict = df_val_hist.groupby('buyer_admin_id')['item_id'].apply(list).to_dict()
    user_label_dict = df_val_label.set_index('buyer_admin_id')['item_id'].to_dict()

    results = {f'NDCG@{k}': [] for k in top_k_list}
    results.update({f'Recall@{k}': [] for k in top_k_list})
    results.update({f'HitRate@{k}': [] for k in top_k_list})
    results['MRR'] = []

    user_count = 0
    for user_id in user_label_dict.keys():
        if user_id not in user_hist_dict:
            continue

        user_seq = user_hist_dict[user_id]
        if not user_seq:
            continue

        item_cf_results = item_cf.recall(user_seq, top_k=max(top_k_list))
        item2vec_results = item2vec.recall(user_seq, top_k=max(top_k_list))

        all_scores = defaultdict(float)
        for item, score in item_cf_results.items():
            all_scores[item] += score
        for item, score in item2vec_results.items():
            all_scores[item] += score

        recommended_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, score in recommended_items]

        relevant_items = [user_label_dict[user_id]]

        for k in top_k_list:
            results[f'NDCG@{k}'].append(compute_ndcg_at_k(relevant_items, recommended_items, k))
            results[f'Recall@{k}'].append(compute_recall_at_k(relevant_items, recommended_items, k))
            results[f'HitRate@{k}'].append(compute_hit_rate_at_k(relevant_items, recommended_items, k))
        results['MRR'].append(compute_mrr(relevant_items, recommended_items))

        user_count += 1

    metrics = {}
    for metric_name, values in results.items():
        metrics[metric_name] = np.mean(values) if values else 0.0

    return metrics, user_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Model Evaluation (Recall + Ranking)')
    parser.add_argument('--sample', type=int, default=0)
    args = parser.parse_args()

    print("\nLoading Data")
    df_A = load_user_data(os.path.join(RAW_DATA_DIR, 'Antai_hackathon_train.csv'))
    item_df = load_item_data(os.path.join(RAW_DATA_DIR, 'Antai_hackathon_attr.csv'))

    print("Splitting Train/Val")
    df_train_hist, df_val_label = split_train_val(df_A)
    df_val_hist = df_train_hist.groupby('buyer_admin_id').tail(1)

    if args.sample > 0:
        print(f"Sampling {args.sample} users for evaluation")
        val_users = df_val_label['buyer_admin_id'].unique()[:args.sample]
        df_val_hist = df_val_hist[df_val_hist['buyer_admin_id'].isin(val_users)]
        df_val_label = df_val_label[df_val_label['buyer_admin_id'].isin(val_users)]
    else:
        print("Using full validation set")

    print("Loading Models")
    item_cf = joblib.load(os.path.join(MODEL_DIR, 'item_cf.pkl'))
    item2vec = joblib.load(os.path.join(MODEL_DIR, 'item2vec.pkl'))
    ranker = LGBMRanker(LGBM_PARAMS)
    ranker.load(os.path.join(MODEL_DIR, 'lgbm_ranker.pkl'))

    print("Evaluating Recall Models Only")
    recall_metrics, recall_user_count = evaluate_recall_only(df_val_hist, df_val_label, item_cf, item2vec)

    print("\nEvaluating Full Model (Recall + Ranking)")
    full_metrics, full_user_count = evaluate_model(df_val_hist, df_val_label, item_cf, item2vec, ranker, item_df)

    print(f"\nRecall + Ranking (Users: {full_user_count}) ---")
    for metric_name in ['MRR', 'NDCG@10', 'Recall@10', 'HitRate@10']:
        if metric_name in full_metrics:
            print(f"{metric_name:15s}: {full_metrics[metric_name]:.6f}")

    results_df = pd.DataFrame([
        {'Model': 'Recall Only', **recall_metrics},
        {'Model': 'Full (Recall+Ranking)', **full_metrics}
    ])
    result_path = os.path.join(SUBMISSION_DIR, 'evaluation_results.csv')
    results_df.to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()

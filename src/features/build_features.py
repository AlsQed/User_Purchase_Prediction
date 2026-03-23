import pandas as pd
from tqdm import tqdm

def build_ranking_dataset(user_seqs, recall_cf, recall_i2v, item_df, labels=None):

    dataset = []
    
    for user, hist in tqdm(user_seqs.items(), desc="Building Base Features"):
        if len(hist) == 0: continue
            
        cf_res = recall_cf.recall(hist)
        i2v_res = recall_i2v.recall(hist)
        
        candidates = set(cf_res.keys()) | set(i2v_res.keys())
        
        for item in candidates:
            feat = {
                'buyer_admin_id': user,
                'item_id': item,
                'cf_score': cf_res.get(item, 0.0),
                'i2v_score': i2v_res.get(item, 0.0),
                'item_pop': recall_cf.item_popular.get(item, 0),
                'user_seq_len': len(hist)
            }
            
            if labels is not None:
                feat['label'] = 1 if item == labels.get(user) else 0
                
            dataset.append(feat)
            
    df_rank = pd.DataFrame(dataset)
    
    print("Merging Item Attributes")
    df_rank = df_rank.merge(item_df, on='item_id', how='left')
    
    return df_rank
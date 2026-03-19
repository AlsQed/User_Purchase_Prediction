from collections import defaultdict
from tqdm import tqdm

def _create_nested_dict():
    """用于 defaultdict 的辅助函数，可被 pickle 序列化"""
    return defaultdict(int)

class ItemCF:
    def __init__(self):
        self.item_sim_matrix = defaultdict(_create_nested_dict)
        self.item_popular = defaultdict(int)
        
    def fit(self, df):
        """基于历史数据构建共现矩阵"""
        user_seqs = df.groupby('buyer_admin_id')['item_id'].apply(list).to_dict()
        
        for user, seq in tqdm(user_seqs.items(), desc="Training Item-CF"):
            for i in range(len(seq)):
                self.item_popular[seq[i]] += 1
                for j in range(i + 1, len(seq)):
                    weight = 1.0 / (j - i)  # 距离惩罚
                    self.item_sim_matrix[seq[i]][seq[j]] += weight
                    self.item_sim_matrix[seq[j]][seq[i]] += weight
                    
    def recall(self, user_seq, top_k=100):
        if not user_seq: return {}
        last_item = user_seq[-1]
        
        sim_items = self.item_sim_matrix.get(last_item, {})
        sorted_items = sorted(sim_items.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {item: score for item, score in sorted_items}
from gensim.models import Word2Vec

class Item2VecRecall:
    def __init__(self, vector_size=64, window=5):
        self.vector_size = vector_size
        self.window = window
        self.model = None
        
    def fit(self, df):
        """训练 Word2Vec 模型"""
        sentences = df.groupby('buyer_admin_id')['item_id'].apply(list).tolist()
        print("Training Item2Vec")
        self.model = Word2Vec(
            sentences, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=1, 
            sg=1, 
            workers=8
        )
        
    def recall(self, user_seq, top_k=100):
        if not user_seq or self.model is None: return {}
        
        recent_items = [item for item in user_seq[-3:] if item in self.model.wv]
        if not recent_items: return {}
        
        sim_items = self.model.wv.most_similar(positive=recent_items, topn=top_k)
        return {item: score for item, score in sim_items}
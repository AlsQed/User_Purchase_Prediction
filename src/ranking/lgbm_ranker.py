import lightgbm as lgb
import joblib

class LGBMRanker:
    def __init__(self, params):
        self.params = params
        self.model = None
        self.features = [
            'cf_score', 'i2v_score', 'item_pop', 'user_seq_len', 
            'cate_id', 'store_id', 'item_price'
        ]
        self.cat_features = ['cate_id', 'store_id']
        
    def fit(self, df_train):
        print("Training LightGBM")
        X_train = df_train[self.features]
        y_train = df_train['label']
        
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            categorical_feature=self.cat_features,
            free_raw_data=False
        )
        
        self.model = lgb.train(
            self.params, 
            train_data, 
            num_boost_round=300
        )
        
    def predict(self, df_test):
        X_test = df_test[self.features]
        return self.model.predict(X_test)
        
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)
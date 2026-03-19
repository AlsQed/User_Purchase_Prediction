import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SUBMISSION_DIR = os.path.join(DATA_DIR, 'submissions')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 确保目录存在
for d in [PROCESSED_DATA_DIR, SUBMISSION_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# 召回超参数
RECALL_TOP_K = 100
I2V_VECTOR_SIZE = 64
I2V_WINDOW = 5

# 排序模型超参数
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'n_jobs': -1,  
    'seed': 42
}
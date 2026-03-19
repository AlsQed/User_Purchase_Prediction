import pandas as pd

def load_user_data(file_path):
    """加载用户行为数据并按时间排序"""
    df = pd.read_csv(file_path, sep=',')
    
    # 剔除无用特征
    if 'buyer_country_id' in df.columns:
        df = df.drop(columns=['buyer_country_id'])

    df = df.sort_values(by=['buyer_admin_id', 'irank'], ascending=[True, False]).reset_index(drop=True)

    df['item_id'] = df['item_id'].astype(str)
    
    return df

def load_item_data(file_path):
    """加载商品属性数据"""
    df = pd.read_csv(file_path, sep=',')
    df['item_id'] = df['item_id'].astype(str)
    df['cate_id'] = df['cate_id'].astype('category')
    df['store_id'] = df['store_id'].astype('category')
    df['item_price'] = df['item_price'].astype(float)
    
    return df

def split_train_val(df):
    """切分训练集：每个用户的最后一条作为验证标签，其余作为历史序列"""
    val_df = df.groupby('buyer_admin_id').tail(1).copy()
    train_hist_df = df.drop(val_df.index).copy()
    return train_hist_df, val_df
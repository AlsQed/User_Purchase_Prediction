import pandas as pd

def load_user_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    
    if 'buyer_country_id' in df.columns:
        df = df.drop(columns=['buyer_country_id'])

    df = df.sort_values(by=['buyer_admin_id', 'irank'], ascending=[True, False]).reset_index(drop=True)

    df['item_id'] = df['item_id'].astype(str)
    
    return df

def load_item_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df['item_id'] = df['item_id'].astype(str)
    df['cate_id'] = df['cate_id'].astype('category')
    df['store_id'] = df['store_id'].astype('category')
    df['item_price'] = df['item_price'].astype(float)
    
    return df

def split_train_val(df):
    val_df = df.groupby('buyer_admin_id').tail(1).copy()
    train_hist_df = df.drop(val_df.index).copy()
    return train_hist_df, val_df
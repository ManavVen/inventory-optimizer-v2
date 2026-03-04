import pandas as pd
REQUIRED_COLS=['date','sku','units_sold','lead_time_days','unit_cost','on_hand_inventory']
def load_sales_data(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    miss=[c for c in REQUIRED_COLS if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {miss}")
    df=df.copy()
    df['date']=pd.to_datetime(df['date'])
    df['sku']=df['sku'].astype(str)
    for c in ['units_sold','lead_time_days','unit_cost','on_hand_inventory']:
        df[c]=pd.to_numeric(df[c])
    return df.sort_values(['sku','date']).reset_index(drop=True)
def train_test_split_by_date(df:pd.DataFrame,test_days:int=30):
    cut=df['date'].max()-pd.Timedelta(days=test_days)
    tr=df[df['date']<=cut].copy()
    te=df[df['date']>cut].copy()
    if tr.empty or te.empty: raise ValueError("Empty train/test split")
    return tr,te

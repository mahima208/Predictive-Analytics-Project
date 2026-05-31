import pandas as pd
from typing import Tuple


def load_sales_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # try to find a date column
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.rename(columns={date_cols[0]: 'date'})
    elif 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError('No date-like column found in the dataset')

    # try to find a sales column
    sales_cols = [c for c in df.columns if c.lower() in ('sales', 'amount', 'revenue')]
    if sales_cols:
        df = df.rename(columns={sales_cols[0]: 'sales'})
    elif 'value' in df.columns:
        df = df.rename(columns={'value': 'sales'})
    else:
        # fallback: try numeric columns other than date
        numeric = df.select_dtypes('number').columns.tolist()
        numeric = [c for c in numeric if c not in ('date',)]
        if not numeric:
            raise ValueError('No sales-like numeric column found')
        df = df.rename(columns={numeric[0]: 'sales'})

    df = df[['date', 'sales']].dropna()
    df = df.sort_values('date')
    return df


def preprocess_time_series(df: pd.DataFrame, freq: str = 'D') -> pd.Series:
    df = df.copy()
    df.set_index('date', inplace=True)
    # aggregate by day (or specified freq)
    ts = df['sales'].resample(freq).sum()
    # fill missing values with forward fill then zero
    ts = ts.ffill().fillna(0)
    return ts


def train_test_split_ts(ts: pd.Series, test_size: int = 30) -> Tuple[pd.Series, pd.Series]:
    if test_size <= 0 or test_size >= len(ts):
        raise ValueError('test_size must be >0 and < length of series')
    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]
    return train, test

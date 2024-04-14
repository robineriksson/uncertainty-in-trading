import polars as pl
import datetime

def load_data():
    path = '../../data/BTC-2021min.csv'

    df = pl.read_csv(path)


    df=(df
        .with_columns(pl.col('date').str.to_datetime())
        .with_columns(day = pl.col('date').dt.date())
        .filter(pl.col('day') == datetime.date(2022,2,28))
        .select(['date','close'])
        .sort(pl.col('date'))
        )
    return df

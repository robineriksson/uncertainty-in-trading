import polars as pl
import datetime

def load_data():
    path = '../../data/BTC-2021min.csv'

    df = pl.read_csv(path)


    df=(df
        .with_columns(pl.col('date').str.to_datetime())
        .with_columns(day = pl.col('date').dt.date(),
                      returns = pl.col('close').log().add(1).pct_change(),
                      ranges = pl.col('high')-pl.col('low'))
        .filter(pl.col('day') == datetime.date(2022,2,28))
        .select(['date','close','returns','ranges'])
        .sort(pl.col('date'))
        )
    return df

import numpy as np
import pandas as pd

"""
Work done here:

Retrieved the data from world bank website.

Used the already existing WDIdata3decades.xlsx to find out which columns have the least number of 
null values and included only those columns in the search field of the world bank website

Used the retrieved world bank csv file to find out countries that have the highest number of NaN values and removed
those rows.

"""
class WeightRelation:
    def __init__(self, train_features_set, validation_features_set, test_features_set):
        n_features = len(train_features_set[0])
        self.weights = [np.random.random() for _ in range(n_features)]

    @staticmethod
    def get_non_empty_features_indices(features):
        non_empty_features = []
        for i, f in enumerate(features):
            if f != "":
                non_empty_features.append(i)


def work1():
    df = pd.read_excel(
        "data/WDIdata3decades.xlsx",
        engine='openpyxl',
    )
    metadata_df = pd.read_excel(
        "data/WDImetadata.xlsx",
        engine='openpyxl',
        sheet_name="1990-1999 MetaData"
    )
    good_columns = []
    best_cols = df.isna().sum().sort_values()
    best_cols = best_cols[best_cols.lt(5)]
    for key, value in best_cols.iteritems():
        try:
            print(key, metadata_df[metadata_df["Code"] == key]["Indicator Name"].values)
        except BaseException as b:
            print(b)
            continue


def work2():
    df = pd.read_csv("data/reduced_dataset_v3.csv")
    bad_countries = ['MNP', 'BMU', 'ABW', 'SMR', 'GIB', 'SOM', 'TKM', 'INX', 'DMA', 'VIR', 'FSM', 'NRU', 'SXM', 'GUM',
                     'VGB', 'MHL', 'MAF', 'CUB', 'MAC', 'FRO', 'CUW', 'GRL', 'VEN', 'CYM', 'NCL', 'JPN', 'MCO', 'SSD',
                     'KNA', 'IMN', 'PYF', 'TCA', 'SRB', 'AND', 'PRK', 'EAS', 'ASM', 'XKX', 'HKG', 'ERI', 'SYR', 'TUV',
                     'LIE', 'CHI', 'ISR', 'FCS', 'NZL', 'PRI', 'PLW']
    df = df[~df["Country Code"].isin(bad_countries)]
    print(df)
    df.to_csv("data/reduced_dataset_v3.csv", index=False)

if __name__ == '__main__':
    work2()

import pandas as pd


class PreProcess:
    def __init__(self):
        pass

    @staticmethod
    def load_data(df_path_1, df_path_2, merged_file_path):
        """
        Takes 2 wdi data export files as input and converts them into a single merged file by skipping common
        columns. This single merged file can then be used for processing. If you want to obtain new files,
        you can do so by visiting the following website:
        https://databank.worldbank.org/source/world-development-indicators#
        In that website, select all countries, all series and not more than 5 years on the left hand side.
        Once you're done with that, export the files as csv and paste the path of the csv file to run this function
        :param df_path_1: Path of the first csv file
        :param df_path_2: Path of the second csv file
        :param merged_file_path: The location where the merged file should be stored
        :return: Exports a csv file to the chosen path
        """
        df_part_1 = pd.read_csv(df_path_1)
        df_part_2 = pd.read_csv(df_path_2)
        if df_part_1.shape[0] != df_part_2.shape[0]:
            raise BaseException("Unable to import files because number of rows don't match")
        df_part_2 = df_part_2.iloc[:, 4:]
        df_merged = pd.concat([df_part_1, df_part_2], axis=1)
        df_merged = df_merged.replace("..", "")
        df_merged.to_csv(merged_file_path, index=False)

    def change_columns_to_source(self, merged_df):
        cols = ["Year", "Country Code"]
        cols.extend(list(set(merged_df["Series Code"].values.tolist()[1:])))
        wdi_new = pd.melt(merged_df, id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
                          var_name="Year", value_name="Value")
        print(wdi_new.head())
        # new_df = pd.DataFrame(columns=[])


def execute_load_data():
    path_1 = "/Users/csatyajith/Datasets/ML_proj/wdi_data/wdi_ml_data_part_1.csv"
    path_2 = "/Users/csatyajith/Datasets/ML_proj/wdi_data/wdi_ml_data_part_1.csv"
    storage_path = "data/merged_wdi_data.csv"
    PreProcess.load_data(path_1, path_2, storage_path)


def execute_transform():
    pp = PreProcess()
    df = pd.read_csv("data/merged_wdi_data.csv")
    pp.change_columns_to_source(df)


if __name__ == '__main__':
    execute_transform()

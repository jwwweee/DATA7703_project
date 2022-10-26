import pandas as pd
from itertools import chain

import os

path = 'data/'
output_path = 'data/'
hour = 24

speed_column_name = 'Wind Speed (m/s)'

def convert(input_file, output_file):

    data = pd.read_pickle(input_file)
    new_data = []


    for index, row in data.iterrows():
        extra_rows = pd.date_range(index, periods=hour, freq="H")
        new_y = data.loc[pd.date_range(index, periods=2, freq="H")]
        try:
            if len(data.loc[extra_rows]) < 12:
                break
        except:
            break

    #     new_y.append([cur_row[speed_column_name]])
        flat_list = list(chain(*data.loc[extra_rows, data.columns[:-1]].values.tolist()))
        flat_list.append(new_y.iloc[1][speed_column_name])
        flat_list.append(index)
        new_data.append(flat_list)
    #
    new_data_col = list(chain(*[data.columns[:-1]] * hour))
    new_data_col.append(speed_column_name)
    new_data_col.append('gk')
    # print(len(new_data_col))
    new_table = pd.DataFrame(new_data, columns=new_data_col)
    new_table.set_index('gk', inplace=True)
    # print(new_table.head(20).to_string())
    new_table.to_csv(output_file)


if __name__ == '__main__':
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        o = os.path.join(output_path, filename)
        convert(f, o.replace('.pkl', "_{hour}.csv".format(hour=hour)))






# a command line tool that processes raw csv
# use list comprehensions to clean missing data values
# use NumPy to calculate mean, median, and standard deviation of columns.

import pandas as pd

def analyze(csv_file: str):
    loaded_file = pd.read_csv(csv_file)
    new_prices = [405, 50, 200, 90, 770]
    loaded_file.loc[0:4, "Price"] = new_prices
    loaded_file.to_csv(csv_file, index=False)
    return loaded_file.head(5)
    # used to delete 5 prices from first 5 row
    #rows_to_remove = loaded_file.head(5)
    #loaded_file.loc[rows_to_remove.index, "Price"] = None
    #loaded_file.to_csv(csv_file, index=False)


result = analyze("/Users/mac/Downloads/products-100.csv")
print(result)

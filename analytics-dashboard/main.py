# a command line tool that processes raw csv
# use list comprehensions to clean missing data values
# use NumPy to calculate mean, median, and standard deviation of columns.

import pandas as pd

def analyze(csv_file: str):
    loaded_file = pd.read_csv(csv_file)
    rows_to_clean = loaded_file.head(5)
    new_prices = [200, 5, 200, 10, 770]
    for price in rows_to_clean:
        if price == "Price":

    # used to delete 5 prices from first 5 row
    #rows_to_remove = loaded_file.head(5)
    #loaded_file.loc[rows_to_remove.index, "Price"] = None
    #loaded_file.to_csv(csv_file, index=False)


analyze("/Users/mac/Downloads/products-100.csv")

## TEST READ

import pandas as pd

try:
    df = pd.read_csv("lung_cancer_data.csv", sep="\s+")
    print(df.head())
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
except Exception as e:
    print("ERROR:", e)



import pandas as pd

data = pd.read_csv("lung_cancer_data.csv")
data.columns = data.columns.str.strip()   # remove spaces
print(data.columns.tolist())



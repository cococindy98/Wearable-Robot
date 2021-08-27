import pandas as pd
import numpy as np
import os

IMAGE_PATH = "/home/vision/sy_ws/data2/JPEGImages/"


def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split("p")[0])

    return pd.DataFrame({"id": name}, index=np.arange(0, len(name)))


df = create_df()
df.to_csv("data_name.csv", index=False, encoding="cp949")

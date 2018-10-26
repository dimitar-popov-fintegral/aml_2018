import os
import pandas as pd

import data as dt
import matplotlib.pyplot as plt



grid_search = pd.read_csv(os.path.join(dt.output_dir(), 'svm_all_train.scale_grid_search'), header=None, index_col=None)


x = grid_search.iloc[:, 0]
y = grid_search.iloc[:, 1]
z = grid_search.iloc[:, 2]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z)
plt.show()

print("DONE")


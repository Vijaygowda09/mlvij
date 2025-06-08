from .p1 import p1
from .p2 import p2
from .p3 import p3
from .p4 import p4
from .p5 import p5
from .p6 import p6
from .p7 import p7
from .p8 import p8
from .p9 import p9
from .p10 import p10
p1='''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame
nf = df.select_dtypes(include=[np.number]).columns
for col in nf:
     sns.histplot(df[col], kde=True)
     plt.title(col)
     plt.show()
     sns.boxplot(x=df[col])
     plt.title(col)
     plt.show()
for col in nf:
      Q1, Q3 = df[col].quantile([0.25, 0.75])
      IQR = Q3 - Q1
      outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
      print(f"{col}: {len(outliers)} outliers")
print(df.describe())'''

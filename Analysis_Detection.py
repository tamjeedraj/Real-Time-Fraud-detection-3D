
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

data = pd.read_csv("C:/Users/Acer/Desktop/Project_0.2/Fraud detection project/Dataset/AIML Dataset.csv")
data.head(5)

#data.info()

data.shape
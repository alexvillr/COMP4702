import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel(
    io="./Datasets/Rel_2_Nutrient_file.xlsx", sheet_name="All solids & liquids per 100g"
)
df = df.drop(["Public Food Key", "Food Name"], axis=1)

corrMatrix = df.corr(min_periods=0)
plt.matshow(corrMatrix)
plt.show()

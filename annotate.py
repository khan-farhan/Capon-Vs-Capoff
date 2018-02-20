import os
import pandas as pd
import glob

Capoff_path = os.getcwd() + "/Dataset/Capoff"
CapON_path = os.getcwd() + "/Dataset/CapON"


os.chdir(Capoff_path)

ImageName = []
for x in glob.glob("*.jpg"):
	ImageName.append(x)


df = pd.DataFrame({'image' : ImageName , 'label' : "0"})

df = df.sample(frac=1).reset_index(drop=True)
print(df)


df.to_csv("../Capoff.csv", index = False)


os.chdir(CapON_path)

ImageName = []
for x in glob.glob("*.jpg"):
	ImageName.append(x)


df = pd.DataFrame({'image' : ImageName , 'label' : "1"})
df = df.sample(frac=1).reset_index(drop=True)
print(df)

df.to_csv("../CapoN.csv", index = False)

os.chdir(nothing_path)


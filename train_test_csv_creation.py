import pandas as pd 
import os

path = os.getcwd() + "/Dataset/"

df_capoff = pd.read_csv(os.path.join(path,"Capoff.csv"))
df_capon = pd.read_csv(os.path.join(path,"CapON.csv"))


#df_test =df_capoff.ix[182:186]

#df_test = df_test.append(df_capoff.ix[154:159], ignore_index = False)

#df_test = df_test.append(df_capon.ix[120:124], ignore_index = False)

#df_test = df_test.append(df_capon.ix[396:401], ignore_index = False)



df = df_capon.tail(10)
df_test = df.append(df_capoff.tail(10), ignore_index = False)



df_train = pd.concat([df_capoff,df_capon,df_test]).drop_duplicates(keep = False)



df_train  = df_train.sample(frac=1).reset_index(drop=True)
df_test  = df_test.sample(frac=1).reset_index(drop=True)




df_train.to_csv(os.path.join(path,"train.csv"),index = False)
df_test.to_csv(os.path.join(path,"test.csv"),index = False)

print(df_test.shape,df_train.shape)
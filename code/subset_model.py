# -*- coding: utf-8 -*-
# ------------------------------------
#  Author  : PC_KQM
#  FileName: subset_model.py
#  Time    : 2023-12-01 15:42
# ------------------------------------
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
import pandas as pd
import numpy as np
import os
import joblib

def regularity_z(df, feature):
    """
    :param df:    需要标准化的数据框
    :return: 两个数据框（标准化后的数据框、均值方差数据框）
    """
    mean_x = []
    std_x = []
    df_scale = pd.DataFrame(index=df.index)
    columns = df[feature].columns.tolist()
    for c in columns:
        d = df[c]
        MEAN = d.mean()
        STD = d.std()
        mean_x.append(MEAN)
        std_x.append(STD)
        df_scale[c] = (d - MEAN) / STD
    mean_std = pd.DataFrame([mean_x, std_x], columns=columns)

    return df_scale, mean_std


all_df = pd.read_excel("E:/kong/肺结节/2023-10-30/0_data/LC011_2023-10-31_new.xlsx")

all_df = all_df[(all_df["DISEASE_Tag"] == 2) | (all_df["DISEASE_Tag"] == 3) | (all_df["DISEASE_Tag"] == 4)].reset_index(drop=True)

df2 = all_df[all_df["DISEASE_Tag"] == 2].reset_index(drop=True)
df3 = all_df[all_df["DISEASE_Tag"] == 3].reset_index(drop=True)
df4 = all_df[all_df["DISEASE_Tag"] == 4].reset_index(drop=True)

df3_sample = df3.sample(n=90)
df4_sample = df4.sample(n=90)

model_df = pd.concat([df2, df3_sample, df4_sample], axis=0, ignore_index=True)

feature = ['utility_subset_02', 'utility_subset_16', 'utility_subset_21',
            'utility_subset_24', 'utility_subset_28', 'utility_subset_32',
            'utility_subset_34', 'utility_subset_35', 'utility_subset_59',
            'utility_subset_62', 'utility_subset_65', 'utility_subset_67']

# model_df0 = all_df[['utility_subset_02', 'utility_subset_16', 'utility_subset_21',
#             'utility_subset_24', 'utility_subset_28', 'utility_subset_32',
#             'utility_subset_34', 'utility_subset_35', 'utility_subset_59',
#             'utility_subset_62', 'utility_subset_65', 'utility_subset_67','DISEASE_Tag',"UNIQUE_ID"]].reset_index(drop=True)

# model_df = model_df0[(model_df0["DISEASE_Tag"] == 2) | (model_df0["DISEASE_Tag"] == 3)].reset_index(drop=True)
print(model_df.shape)
model_df["DISEASE_Tag"][model_df["DISEASE_Tag"]==2] = 1
model_df["DISEASE_Tag"][model_df["DISEASE_Tag"]==3] = 0
model_df["DISEASE_Tag"][model_df["DISEASE_Tag"]==4] = 0
model_df["DISEASE_Tag"] = model_df["DISEASE_Tag"].astype(int)

model_df_scale,mean_std = regularity_z(model_df, feature=feature)
model_df_scale = pd.concat([model_df_scale, model_df["DISEASE_Tag"]], axis=1)
print(model_df_scale.shape)

score_cutoff = 0
save_dir = "../2_result_subset_1/"

if os.path.exists(save_dir)==False:
    os.makedirs(save_dir)

fold = StratifiedKFold(n_splits=10, shuffle=True)
for train ,test in fold.split(X = model_df_scale[feature], y=model_df_scale["DISEASE_Tag"]):
    x_train = model_df_scale.loc[train,feature]
    x_test = model_df_scale.loc[test,feature]
    y_train = model_df_scale.loc[train,"DISEASE_Tag"]
    y_test = model_df_scale.loc[test, "DISEASE_Tag"]

    rfc = RandomForestClassifier(n_estimators=500, max_depth=7,
                             min_samples_leaf=5, bootstrap=False)

    rfc.fit(x_train,y_train)

    train_score = rfc.score(x_train,y_train)
    test_score = rfc.score(x_test,y_test)

    print(f"train_score:{train_score}")
    print(f"test_score:{test_score}")
    print("*****************************")

    if test_score >= score_cutoff:
        model = rfc
        score_cutoff = test_score

joblib.dump(model, save_dir + "train.pkl")
model = joblib.load(save_dir + "train.pkl")

model_df0_scale = (all_df[feature] - mean_std.iloc[0,:]) / mean_std.iloc[1,:]

proba = model.predict_proba(model_df0_scale)
# proba = rfc.predict_proba((model_df[feature] - mean_std.iloc[0,:]) / mean_std.iloc[1,:])

proba_df = pd.DataFrame(proba,columns=["dis_risk","health_risk"])

save_df = pd.concat([proba_df, all_df[["UNIQUE_ID","DISEASE_Tag"]]], axis=1)

df = pd.read_excel("../2_result/predict_df1.xlsx")
save_df = pd.merge(save_df, df[["UNIQUE_ID", "到样日期","group"]], how="left", on="UNIQUE_ID")

save_df.to_excel(save_dir + "predict_df.xlsx", index=0)
mean_std.to_excel(save_dir + "mean_std.xlsx", index=0)


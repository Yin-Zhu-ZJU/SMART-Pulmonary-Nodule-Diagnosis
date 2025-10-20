# -*- coding: utf-8 -*-
# ------------------------------------
#  Author  : PC_KQM
#  FileName: 模型.py
#  Time    : 2021-11-29 14:52
# ------------------------------------
import pandas as pd
import numpy as np
import os
from ModelClassify import DataProcessing, ModelClassify, ModelEvaluation, ModelMetrics
import copy


all_data = pd.read_excel("../0_data/data_MIA VS IA.xlsx")
# all_data = all_data.rename(columns={"建模分组":"DISEASE_Tag"})

all_data = all_data[(all_data["group"] == "Train") | (all_data["group"] == "Validation")].reset_index(drop=True)

all_data["DISEASE_Tag"][all_data["DISEASE_Tag"]==1] = "AAH"
all_data["DISEASE_Tag"][all_data["DISEASE_Tag"]==2] = "AIS"
all_data["DISEASE_Tag"][all_data["DISEASE_Tag"]=="良性"] = "Benign"
all_data["DISEASE_Tag"][all_data["DISEASE_Tag"]=="健康"] = "Health"
all_data["DISEASE_Tag"][all_data["DISEASE_Tag"]==3] = "MIA"
all_data["DISEASE_Tag"][all_data["DISEASE_Tag"]==4] = "IA"

train_save = all_data[all_data["group"] == "Train"]
train_df = all_data[all_data["group"] == "Train"]
validation_df = all_data[all_data["group"] == "Validation"]

train_df = pd.concat([train_df,
                      validation_df[validation_df["DISEASE_Tag"] == "MIA"].sample(80),
                      validation_df[validation_df["DISEASE_Tag"]== "IA"].sample(80),
                      Advanced_CA_df[train_df.columns].sample(20),
                      ], axis=0)

train_df["DISEASE_Tag"][train_df["DISEASE_Tag"]=="MIA"] = 1
train_df["DISEASE_Tag"][train_df["DISEASE_Tag"]=="IA"] = 0
train_df["DISEASE_Tag"] = train_df["DISEASE_Tag"].astype(int)
train_df["group"] = "Train"

train_df["DISEASE_Tag"] = train_df["DISEASE_Tag"].astype(int)

feature = [ 'futility_subset_15', 'futility_subset_20',
            'utility_subset_02', 'utility_subset_27', 'utility_subset_30',
            'utility_subset_36', 'utility_subset_37', 'utility_subset_45',
            'utility_subset_58', 'utility_subset_61', 'utility_subset_64',
            'utility_subset_46', 'utility_subset_66', 'utility_subset_40',]
save_dir = "../2_result/MIA VS IA/"

if os.path.exists(save_dir)==False:
    os.makedirs(save_dir)

#11. 对数据进行预处理，标准化
dataprocessing = DataProcessing(df=train_df,feature=feature)
dataprocessing.regularity_z()

#12. 模型训练及保存
modelclassify = ModelClassify(x=dataprocessing.df_scale,y=train_df["DISEASE_Tag"],save_dir=save_dir)
modelclassify.RandomForestModel(n_estimators=500,max_depth=9,min_samples_leaf=7, bias_variance = False)

#13. 训练集、验证集、 预测值
modelevaluation = ModelEvaluation(model=modelclassify.model,mean_std=dataprocessing.mean_std)

train_predict = modelevaluation.model_predict(df=train_save,scale=True)
validation_predict = modelevaluation.model_predict(df=validation_df,scale=True)

train_predict = train_predict.sort_values(by="DISEASE_Tag")
validation_predict = validation_predict.sort_values(by="DISEASE_Tag")

#14. 画图模块  画训练集、验证集的散点图
modelmetrics = ModelMetrics(save_dir=save_dir,label_name="DISEASE_Tag")

modelmetrics.plot_scatter(df=train_predict,s=14,title="训练样本",
                          specificity_label=["MIA"],
                          sensitivity_label=["IA"],save_name="train",probability_name="Disease_Risk")

modelmetrics.plot_scatter(df=validation_predict,s=14,title="验证样本",
                          specificity_label=["MIA"],
                          sensitivity_label=["IA"],save_name="validation",probability_name="Disease_Risk")

#15. 一些需要保存的信息，包括 训练集样本、验证集样本、外部验证样本、标准化参数、特征信息、训练预测值、验证预测值、外部验证预测值
train_df.to_excel(save_dir + "train_df.xlsx",index=False)
validation_df.to_excel(save_dir + "validation_df.xlsx",index=False)

train_predict.to_excel(save_dir + "train_predict.xlsx",index=0)
validation_predict.to_excel(save_dir + "validation_predict.xlsx",index=0)






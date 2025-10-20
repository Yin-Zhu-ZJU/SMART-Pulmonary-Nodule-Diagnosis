# -*- coding: utf-8 -*-
# ------------------------------------
#  Author  : PC_KQM
#  FileName: fenlei.py
#  Time    : 2021-11-19 10:58
# ------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import joblib
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestClassifier
# from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier


class DataProcessing(object):

    def __init__(self,df,feature):
        self.df = df
        self.mean_std = None
        self.df_scale = None
        self.feature = feature

    def regularity_z(self):
        """
        :param df:    需要标准化的数据框
        :return: 两个数据框（标准化后的数据框、均值方差数据框）
        """
        mean_x = []
        std_x = []
        self.df_scale = pd.DataFrame(index=self.df.index)
        columns = self.df[self.feature].columns.tolist()
        for c in columns:
            d = self.df[c]
            MEAN = d.mean()
            STD = d.std()
            mean_x.append(MEAN)
            std_x.append(STD)
            self.df_scale[c] = (d - MEAN) / STD
        self.mean_std = pd.DataFrame([mean_x, std_x], columns=columns)

    def save_feature(self, save_dir, subset_file):
        subset = pd.read_excel(subset_file)
        unique_id = subset["Unique_ID"]
        feature_dict = {}
        for i in self.feature:
            feature_dict.update(
                {i: subset["Gate_Name"][subset["Unique_ID"] == i].to_list()[0] if i in unique_id.to_list() else i})

        feature_dict_df = pd.DataFrame.from_dict(feature_dict, orient="index", columns=["Gate_Name"])
        feature_dict_df = feature_dict_df.sort_index()
        feature_dict_df = feature_dict_df.reset_index().rename(columns={"index": "Unique_ID"})
        feature_dict_df.to_excel(save_dir + "feature.xlsx", index=False)


class ModelClassify(object):


    def __init__(self,x,y,save_dir):
        """

        :param x:
        :param y:
        :param save_dir:
        """
        self.x = x
        self.y = y
        self.random_state = None
        self.model = None
        self.n_split = 10
        self.save_dir = save_dir


    def RandomForestModel(self,n_estimators,max_depth,min_samples_leaf, bias_variance = False):
        """

        :param n_estimators: 随机森林模型中的基分类器个数
        :param max_depth:
        :param min_samples_leaf:
        :return:
        """


        fold = StratifiedKFold(n_splits=self.n_split, shuffle=True, random_state=self.random_state)
        x = np.array(self.x)
        y = np.array(self.y)
        Train_scores = []
        Test_scores = []
        score_test = 0
        score_train = 0
        for train, test in fold.split(x, y):
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]

            # rfc = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,random_state=123)
            # rfc = BaggingClassifier(base_estimator=tree,n_estimators=n_estimators,random_state=123,bootstrap=False)

            rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf, bootstrap=False)
            rfc.fit(x_train, y_train)
            train_score = rfc.score(x_train, y_train)
            test_score = rfc.score(x_test, y_test)

            if bias_variance:
                print("!!!!!!!")

                # avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(estimator=rfc, X_train=x_train,
                #                                                             y_train=y_train,
                #                                                             X_test=x_test, y_test=y_test)
                # print("**********" * 3)
                # print("train_score:", train_score)
                # print("test_score:", test_score)
                # print('Average expected loss: %.3f' % avg_expected_loss)
                # print('Average bias: %.3f' % avg_bias)
                # print('Average variance: %.3f' % avg_var)
                # print("**********" * 3)
                # Train_scores.append(train_score)
                # Test_scores.append(test_score)
            else:
                print("**********" * 3)
                print("train_score:", train_score)
                print("test_score:", test_score)
                print("**********" * 3)
                Train_scores.append(train_score)
                Test_scores.append(test_score)

            if score_test <= test_score:
                self.model = rfc
                score_test = test_score
                score_train = train_score
        joblib.dump(self.model, self.save_dir + "train.pkl")

    def AdaboostSVMModel(self,n_estimators,learning,C,gamma,kernel):

        fold = StratifiedKFold(n_splits=self.n_split, shuffle=True, random_state=self.random_state)
        x = np.array(self.x)
        y = np.array(self.y)
        Train_scores = []
        Test_scores = []
        score_test = 0
        score_train = 0
        for train, test in fold.split(x, y):
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]
            adbt = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning,
                                      base_estimator=svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=True),
                                      random_state=self.random_state)
            adbt.fit(x_train, y_train)
            train_score = adbt.score(x_train, y_train)
            test_score = adbt.score(x_test, y_test)
            print("**********" * 3)
            print("train_score:", train_score)
            print("test_score:", test_score)
            print("**********" * 3)
            Train_scores.append(train_score)
            Test_scores.append(test_score)
            if score_test <= test_score:
                self.model = adbt
                score_test = test_score
                score_train = train_score
        joblib.dump(self.model, self.save_dir + "train.pkl")


class ModelEvaluation(object):

    def __init__(self,model,mean_std):
        self.model = model
        self.mean_std = mean_std
        self.prediction = None
        self.score = None
        self.predict_tag = None
        self.specificity = None
        self.sensitivity = None
        self.precision = []
        self.rescall = []
        self.f1_score = []

    def model_score(self,x,y):
        self.score = self.model.score(x,y)

    def model_predict(self,df,scale=True):
        """
        概率预测，
        :param df: 传入df 包含 类别列 DISEASE_Tag 和 UNIQUE_ID 列
        :param scale: 标准化数据
        :return: dataframe n x 4 的维度，
        """
        df = df.reset_index(drop=True)
        feature = self.mean_std.columns.tolist()
        if scale:
            x_scale = (df[feature] - self.mean_std.iloc[0, :]) / self.mean_std.iloc[1, :]
        else:
            x_scale = df[feature]

        self.predict_tag = self.model.predict(x_scale)
        self.prediction = self.model.predict_proba(x_scale)
        self.prediction = pd.DataFrame(self.prediction, columns=["Disease_Risk", "Health_Risk"])
        self.prediction = pd.concat([df[["UNIQUE_ID","DISEASE_Tag","group"]],self.prediction],axis=1)

        return self.prediction
        # self.prediction.to_excel(self.save_dir + "/" + self.save_name + "_" + "probability.xlsx",index=False)

    def confusion_matrix_map(self,df):
        """
        画 混淆矩阵热力图
        :param df: 传入df 需要包括 DISEASE_Tag列
        :return:
        """
        sn.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(self.model.predict(df[self.mean_std.columns]), df["DISEASE_Tag"])
        sn.heatmap(C2, annot=True, ax=ax, fmt='d')  # 画热力图
        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('true')  # x轴
        ax.set_ylabel('predict')
        plt.show()

    def ROC(self,df):
        """
        画ROC曲线图
        :param df: 传入一个df
        :return:
        """
        plot_roc_curve(self.model, df[self.mean_std.columns], df["DISEASE_Tag"], response_method='decision_function')  # response_method='predict_proba'
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()

    def specificity_sensitivity(self,specificity_label,sensitivity_label,label_name):
        """
        计算特异性灵敏性
        :param specificity_label:
        :param sensitivity_label:
        :param label_name:
        :return:
        """
        specificity_df = []
        sensitivity_df = []
        for i in specificity_label:
            specificity_i = self.df[self.df[label_name]==i]
            specificity_df.append(specificity_i)
        for j in sensitivity_label:
            sensitivity_i = self.df[self.df[label_name]==j]
            sensitivity_df.append(sensitivity_i)

        specificity_df = pd.concat(specificity_df)
        sensitivity_df = pd.concat(sensitivity_df)

        self.specificity = round(((self.prediction["Disease_Risk"] < 0.5).sum()/len(specificity_df))*100,2)
        self.sensitivity = round(((self.prediction["Disease_Risk"] >= 0.5).sum()/len(sensitivity_df))*100,2)

    #precision、recall、F1 score
    def Calculate_Precision_Rescall(self,x,y,scale=True):
        """
        计算每个类的精准率、召回率和F1-Score
        :param df:
        :param scale: True or False 是否标准化，如果df已经标准化则为False,默认为True
        :return:
        """
        feature = self.mean_std.columns.tolist()
        if scale:
            x_scale = (x[feature] - self.mean_std.iloc[0, :]) / self.mean_std.iloc[1, :]
        else:
            x_scale = x[feature]

        predict_tag = self.model.predict(x_scale)

        for i in np.unique(y):
            #计算每个类的精率
            precision_i = (np.where(predict_tag.values==i) == np.where(
                            y.values==i)).sum() /(predict_tag == i).sum()

            print("类别{}的精率为：{}%".format(i,round(precision_i*100,2)))
            self.precision.append(precision_i)

            rescall_i = (np.where(predict_tag.values==i) == np.where(
                        y.values==i)).sum() /(y.values == i).sum()
            print("类别{}的召回率为：{}%".format(i,round(rescall_i*100,2)))
            self.rescall.append(rescall_i)

            f1_score_i = 2 * precision_i * rescall_i/(precision_i + rescall_i)
            print("类别{}的F1-Score为：{}%".format(i, round(f1_score_i * 100, 2)))
            self.f1_score.append(f1_score_i)


class ModelMetrics(object):
    def __init__(self,save_dir,label_name):
        self.save_dir = save_dir
        self.label_name = label_name

    def plot_scatter(self,df,s,title,specificity_label,sensitivity_label,save_name,probability_name="Disease_Risk"):

        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用于显示中文
        plt.rcParams["axes.unicode_minus"] = False  # 用于正常显示负号
        df = df.reset_index(drop=True)
        df["ID"] = np.arange(1, len(df) + 1)
        specificity_df = []
        sensitivity_df = []

        if specificity_label != None:
            for i in specificity_label:
                specificity_i = df[df[self.label_name]==i]
                specificity_df.append(specificity_i)
            specificity_df = pd.concat(specificity_df)
            specificity = round(((specificity_df[probability_name] < 0.5).sum() / len(specificity_df)) * 100, 2)

        if sensitivity_label != None:
            for j in sensitivity_label:
                sensitivity_i = df[df[self.label_name]==j]
                sensitivity_df.append(sensitivity_i)
            sensitivity_df = pd.concat(sensitivity_df)
            sensitivity = round(((sensitivity_df[probability_name] >= 0.5).sum()/len(sensitivity_df))*100,2)

        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        sn.scatterplot(x="ID", y=probability_name, hue=self.label_name, style=self.label_name, s=s, data=df)
        plt.hlines(y=0.2, xmin=1, xmax=df.shape[0], colors="g", linestyles="--")
        plt.hlines(y=0.5, xmin=1, xmax=df.shape[0], colors="g", linestyles="--")
        plt.hlines(y=0.8, xmin=1, xmax=df.shape[0], colors="g", linestyles="--")
        plt.ylim([0, 1])
        plt.title(title)
        if specificity_label != None:
            plt.text(x=len(df)+1, y=0.55, s="特异性={}%".format(specificity),fontdict={'weight':'heavy',"size":12})
        if sensitivity_label != None:
            plt.text(x=len(df)+1, y=0.5, s="灵敏性={}%".format(sensitivity),fontdict={'weight':'heavy',"size":12})

        plt.subplots_adjust(right=0.8)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        plt.savefig(self.save_dir + "/" + save_name + "_" + "plot.png")
        # plt.show()
        plt.close()


# -*- coding:utf-8 -*-

import math
import numpy as np
import pandas as pd

def calcStd(feature):
    feature_mean = np.sum(feature) / len(feature)
    feature_mean_rep = np.tile(feature_mean, (len(feature), 1))
    temp = feature - feature_mean_rep
    temp = temp * temp
    temp = np.sum(temp) / (len(feature) - 1)
    std = math.sqrt(temp)

    return std

def calcCov(feature_1, feature_2):
    feature_1_mean = np.sum(feature_1) / len(feature_1)
    feature_2_mean = np.sum(feature_2) / len(feature_2)
    feature_1_mean_rep = np.tile(feature_1_mean, (len(feature_1), 1))
    feature_2_mean_rep = np.tile(feature_2_mean, (len(feature_2), 1))
    temp = (feature_1 - feature_1_mean_rep) * (feature_2 - feature_2_mean_rep)
    cov = np.sum(temp) / (len(feature_1) - 1)
    return cov

def calcPearMat(data_set):
    data_set = np.array(data_set)
    lenth = len(data_set[0])
    pearson = np.zeros((lenth, lenth))
    for i in range(0, lenth):
        for j in range(0, lenth):
            pearson[i, j] = calcCov(data_set[:, i], data_set[:, j]) / (calcStd(data_set[:, i]) * calcStd(data_set[:, j]))
    return pearson

def calcCovMat(data_set):
    data_set = np.array(data_set)
    lenth = len(data_set[0])
    cov = np.zeros((lenth, lenth))
    for i in range(0, lenth):
        for j in range(0, lenth):
            cov[i, j] = calcCov(data_set[:, i], data_set[:, j])

    return cov

def calcFeatureVector(mat):
    feature_root_vector = []
    feature_root,feature_vector = np.linalg.eig(mat)
    for i in range(0, len(feature_root)):
       feature_root_vector.append([feature_root[i],feature_vector[i]])

    return feature_root_vector

def calcContributedRate(feature_root_vector):
    sorted_root_vector = sorted( feature_root_vector, key=lambda x: x[0], reverse=True)
    total_rate = sum([i for i,_ in sorted_root_vector])

    contribute_rate = np.zeros(len(sorted_root_vector))
    for i in range(0, len(sorted_root_vector)):
        contribute_rate[i] = sorted_root_vector[i][0] / total_rate
    return contribute_rate, sorted_root_vector

def calcAccumulatedContributedRate(feature_root_vector, choose_threshold):
    accumulated_contribute_rate = 0
    selected_root_vector = []
    contribute_rate, sorted_feature_rootandvector = calcContributedRate(feature_root_vector)
    for i in range(0,len(contribute_rate)):
        accumulated_contribute_rate += contribute_rate[i]
        selected_root_vector.append(sorted_feature_rootandvector[i])
        if accumulated_contribute_rate >= choose_threshold:
            break

    return selected_root_vector

def calcCoefficient(selcted_root_vector):
    Coefficient = {}
    for i in range(0,len(selcted_root_vector[0])):
        Coefficient[i] = [math.sqrt(selcted_root_vector[i][0]) * vector_element for  vector_element in selcted_root_vector[i][1]]

    return Coefficient


if __name__ == '__main__':
    import sys

    file_name = 'dataset.xlsx'
    df = pd.read_excel(file_name, sheet_name="Sheet1")
    df = df.iloc[:, 1:].values

    pearson_mat = calcPearMat(df)

    feature_root_vector = calcFeatureVector(pearson_mat)
    print("特征根：", feature_root_vector[0])
    print("特征向量：", feature_root_vector[1])
    contribute_rate = calcContributedRate(feature_root_vector)
    print("贡献率:", contribute_rate)
    choose_threshold = input("请输入累计贡0.5献率阈值")
    choose_threshold = float(choose_threshold)
    selected_result = calcAccumulatedContributedRate(feature_root_vector, choose_threshold)
    print('--'*50)
    print(selected_result)
    pca = calcCoefficient(selected_result)
    print("主成分:", pca)



# -*- coding:utf-8 -*-

import numpy as np

A = np.mat([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(A)

# U为左奇异向量，V为右奇异向量，sigma为奇异值的对角矩阵
U, sigma, VT = np.linalg.svd(A)
print("U", U)
print("sigma", sigma)
print("VT", VT)

# 降为1维，sigma_c，U_c，VT_c为保存的矩阵
sigma_c = np.diag(sigma[0:1])
print(sigma_c)
U_c = U[:,0:1]
print(U_c)
VT_c = VT[0:1,:]
print(VT_c)

# 还原
print("conv A", U_c * sigma_c * VT_c)
print(A * VT_c.T)
print(U_c * sigma_c * VT_c * VT_c.T)
print(VT_c * VT_c.T)
print(U_c * sigma_c)

# 3.3编程实现对率回归，并给出西瓜数据集3.0α上的结果


import numpy as np
import matplotlib.pyplot as plt


Data0 = np.array([[0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
                 [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103]])
Data = np.transpose(Data0)
Label = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
(n, m) = Data.shape


def l_value(beta):
    num = 0
    for i in range(0, n):
        num = num - Label[i]*
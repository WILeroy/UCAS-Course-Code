import numpy as np
import matplotlib.pyplot as plt
from tools import gaussion as gs

dataset1 = [(0, 0), (2, 0), (2, 2), (0, 2)]
mean1, cov1 = gs(dataset1)

#print(mean1, cov1)

dataset2 = [(4, 4), (6, 4), (6, 6), (4, 6)]
mean2, cov2 = gs(dataset2)

#print(mean2, cov2)

if ((cov1.all()==cov2.all())):
    # 计算判别界面的参数
    arg1 = np.dot((mean1-mean2), np.linalg.inv(cov1))
    arg2 = -0.5 * np.dot(np.dot(mean1, np.linalg.inv(cov1)), mean1.T) + 0.5 * np.dot(np.dot(mean2, np.linalg.inv(cov2)), mean2.T)
#    print(arg1, arg2)

    # 接受输入，使用所得判别界面进行分类
    print('请分别输入需要分类的模式向量（x1 x2）：')
    input_1 = int(input())
    input_2 = int(input())
    input_ = np.array((input_1, input_2))
    print(input_)
    r = np.dot(input_.T, arg1) + arg2
    if r < 0:
        print('第2类')
    elif r > 0:
        print('第1类')
    else:
        print('无法判别')

    # 绘制判别界面    
    x1 = np.linspace(-1, 7, 50)
    x2 = (x1*arg1[0] + arg2) / (-arg1[1])
    bayesfig = plt.figure()
    bayesfig.suptitle("Bayes")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(x1, x2, color="green")

    # 绘制样本点
    for i in range(len(dataset1)):
        plt.scatter(dataset1[i][0], dataset1[i][1], s=20, color="blue")
    for i in range(len(dataset2)):
        plt.scatter(dataset2[i][0], dataset2[i][1], s=20, color="red")
    plt.scatter(input_1, input_2, s=20, color="yellow")    

    # 绘制
    plt.show()
else:
    pass
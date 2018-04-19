import numpy as np
import matplotlib.pyplot as plt


N = 30
dataset1 = (20, 35, 30, 35, 27,20, 35, 30, 35, 27,20, 35, 30, 35, 27,20, 35, 30, 35, 27,20, 35, 30, 35, 27,20, 35, 30, 35, 27)
dataset2 = (25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25)
dataset3 = (25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25,25, 32, 34, 20, 25)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, dataset1, width)
p2 = plt.bar(ind, dataset2, width,
             bottom=dataset1)
p3 = plt.bar(ind, dataset3, width,
             bottom=np.array(dataset1)+np.array(dataset2))

plt.ylabel('Score')
plt.title('Classes')
plt.xticks(ind, ('0','1', '2','3','4', '5', '6','7','8','9','zero','one','two','three','four','five','six','seven','eight','nine','soon','nung','song','saam','see','har','hok','ched','paad','khaw'))
plt.yticks(np.arange(0, 81, 10)) #start stop increase
plt.legend((p1[0], p2[0],p3[0]), ('Model_1', 'Model_2','Model_3'))

plt.show()

def plotStackBar(prob1,prob2,prob3):
    N = 30
    dataset1 = prob1
    dataset2 = prob2
    dataset3 = prob3
    ind = np.arange(N)    # the x locations for the groups
    width = 0.3       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, dataset1, width)
    p2 = plt.bar(ind, dataset2, width,
                 bottom=dataset1)
    p3 = plt.bar(ind, dataset3, width,
                 bottom=np.array(dataset1)+np.array(dataset2))

    plt.ylabel('Score')
    plt.title('Classes')
    plt.xticks(ind, ('0','1', '2','3','4', '5', '6','7','8','9','zero','one','two','three','four','five','six','seven','eight','nine','soon','nung','song','saam','see','har','hok','ched','paad','khaw'))
    plt.yticks(np.arange(0, 81, 10)) #start stop increase
    plt.legend((p1[0], p2[0],p3[0]), ('Model_1', 'Model_2','Model_3'))

    plt.show()


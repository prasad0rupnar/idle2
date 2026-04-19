import numpy as np
x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
y  = np.array([0,0,0,1])
n = 1
n2 = 1
theta = 2
w1 = int(input("Enter the weight (w1)="))
w2 = int(input("Enter the weight (w2)="))
theta = int(input("Enter the threshold (theta1)="))
f = x1 * w1 + x2 * w2
y_predict = (f >= theta).astype(int)
if np.all(y == y_predict):
    print("correct weights and threshold")
    print("w1=", w1)
    print("w2=", w2)
    print("threshold=", theta)
else:
    print("change the weights/threshold")


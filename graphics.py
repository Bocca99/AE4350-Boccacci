import numpy as np
import matplotlib.pyplot as plt

#500 episodes, 300 duration, snakeEnv1
r_normal = np.load("average_test_normal.npy")
rnormal2 = np.load("average_test.npy")
r_gamma05 = np.load("average_test_gamma.npy")
r_gamma75 = np.load("average_test_gamma75.npy")
r_tau = np.load("average_test_tau.npy")
r_xtra = np.load("average_test_x.npy")
r_xtra1 = np.load("average_test_x1.npy")
rlrl = np.load("average_test_lr1.npy")
rlru = np.load("average_test_lr12.npy")
r_buffer = np.load("average_test_buffer.npy")
r_noise = np.load("average_test_noise.npy")
r_s = np.load("average_test_s1.npy")

plt.style.use('ggplot')
plt.plot(r_normal[0:99], color='r',label="Average reward")
plt.plot(r_s, color='b',label="Average reward")
#plt.plot(r_buffer, color='b',label="Average reward")
#plt.plot(rnormal2, color='r',label="Average reward")
#plt.plot(rlrl, color='b',label="Average reward")
#plt.plot(rlru, color='g',label="Average reward")
#plt.plot(r_gamma05, color='b',label="Average reward")
#plt.plot(r_gamma75, color='g',label="Average reward")
#plt.plot(r_tau, color='b',label="Average reward")
#plt.plot(r_xtra,color='b',label="Average reward")
#plt.plot(r_xtra1,color='g',label="Average reward")

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()
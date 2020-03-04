import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, erlang, f, gamma
from scipy.stats.kde import gaussian_kde

pre_stamps = ['00:00.0', '00:11.2', '00:12.5', '00:13.6', '00:15.0', '00:24.6',
              '00:33.9', '00:54.4', '00:56.0', '01:14.6', '01:16.3', '01:43.6',
              '01:48.7', '02:04.1', '02:26.2', '02:54.4', '02:57.7', '02:59.0',
              '03:01.1', '03:06.0', '03:09.0', '03:10.4', '03:20.5', '03:21.6',
              '03:34.4', '03:52.8', '04:17.2', '04:35.1', '04:35.5', '04:37.2',
              '04:37.8', '05:06.2', '05:12.9', '05:36.6', '05:41.9', '05:45.8',
              '06:03.3', '06:06.5', '06:11.3', '06:46.1', '06:46.8', '06:48.7',
              '06:49.9', '07:06.2', '07:07.0', '07:27.2', '07:36.2', '07:40.6',
              '07:41.3', '07:42.0', '07:46.8', '08:15.7', '08:22.6', '08:24.7',
              '08:25.5', '08:27.2', '08:42.9', '08:43.8']

reg_stamps = ['00:00.0', '00:09.2', '00:09.7', '00:11.3', '00:12.8', '00:14.3',
              '00:15.9', '00:26.9', '00:56.8', '01:21.1', '02:16.0', '02:23.6',
              '02:27.1', '02:42.7', '02:44.2', '03:05.6', '03:12.8', '03:15.3',
              '03:26.8', '03:29.3', '04:26.5', '04:42.4', '04:47.8', '04:59.6',
              '05:09.5', '05:40.1', '05:56.7', '06:00.4', '06:19.9', '07:38.8',
              '07:50.8', '08:00.8', '08:01.7', '08:04.2', '08:05.6', '08:17.6',
              '08:19.8', '08:20.9', '08:21.6', '08:32.8', '08:45.6', '08:54.1',
              '08:55.8', '09:13.2', '09:34.4', '09:52.9', '09:55.5']

idcheck_times = ['00:07.6', '00:14.7', '00:05.3', '00:11.9', '00:11.1', '00:14.8',
                 '00:10.0', '00:20.5', '00:09.2', '00:07.7', '00:08.8', '00:07.5',
                 '00:12.6', '00:10.9', '00:15.4', '00:11.9']

mw_stamps = ['00:08.8', '00:20.1', '00:32.5', '00:36.0', '00:43.5', '00:51.7', 
             '01:04.6', '01:15.6', '01:24.7', '01:32.4', '01:43.1', '01:58.2',
             '02:09.2', '02:26.2', '02:39.6', '02:47.3', '02:54.0', '03:05.6',
             '03:25.1', '03:33.6', '04:11.1', '04:22.4', '04:34.8', '04:46.8',
             '05:00.0', '05:11.1', '05:18.4', '05:26.5', '05:40.8', '05:47.9',
             '05:58.9', '06:08.5', '06:36.0', '06:44.6', '06:54.3', '07:05.3',
             '07:15.0', '07:23.6', '07:35.6', '07:42.6']

xray_stamps_1 = ['00:02.5', '00:05.6', '00:07.3', '00:09.3', '00:20.1', '00:22.4',
                 '00:24.5', '00:41.2', '01:07.1', '01:08.8', '01:17.9']
xray_stamps_2 = ['00:00.0', '00:01.5', '00:03.5', '00:11.0']

scanned_times = ['0:48', '0:45', '0:28', '0:25', '0:22', '0:24', '0:17', '0:33',
                 '0:08', '0:10', '0:26', '0:32', '0:21', '0:37', '1:08', '0:40',
                 '0:18', '0:26', '0:08', '0:21', '0:23', '0:28', '0:50', '0:28',
                 '0:48', '0:28', '0:36', '0:27', '0:05']

def Interval(times):
    """ 提取时间间隔序列.
    """
    intervals = []
    for i in range(len(times)):
        m, s = times[i].split(':')
        intervals.append(np.around(float(m)*60+float(s), 1))
    return intervals

def Stamp2Interval(stamps):
    """ 由时间戳序列计算时间间隔序列.
    """
    intervals = []
    for i in range(1, len(stamps)):
        m0, s0 = stamps[i-1].split(':')
        m1, s1 = stamps[i].split(':')
        intervals.append(np.around(float(m1)*60+float(s1)-float(m0)*60-float(s0), 1))
    return intervals

def Erlang(x, k, lam):
    """ 爱尔兰分布概率密度函数.
    """
    return erlang.pdf(x, k) * np.exp(-k * lam * x) * pow(k*lam, k) / np.exp(-x)

def Expon(x, lam):
    """ 指数分布概率密度函数.
    """
    return lam * np.exp(-lam * x)

# 指数分布拟合旅客到达时间间隔.
pre_intervals = Stamp2Interval(pre_stamps)
plt.subplot(1, 2, 2)
x = np.linspace(0, 80, 320)
pre_mean = np.mean(np.array(pre_intervals))
print('pre mean: ', pre_mean)
print('pre lambda: ', 1/pre_mean)
plt.plot(x, Expon(x, 1/pre_mean), label='E({:.5f})'.format(1/pre_mean))
plt.hist(pre_intervals, bins=10, density=True)
plt.title('Pre-Check Passenger\'s Arrival Interval')
plt.ylabel('Probability Density')
plt.xlabel('Interval (sec)')
plt.legend()

reg_intervals = Stamp2Interval(reg_stamps)
plt.subplot(1, 2, 1)
x = np.linspace(0, 80, 320)
reg_mean = np.mean(np.array(reg_intervals))
print('reg mean: ', reg_mean)
print('reg lambda: ', 1/reg_mean)
plt.plot(x, Expon(x, 1/reg_mean), label='E({:.5f})'.format(1/reg_mean))
plt.hist(reg_intervals, bins=10, density=True)
plt.title('Regular Passenger\'s Arrival Interval')
plt.ylabel('Probability Density')
plt.xlabel('Interval (sec)')
plt.legend()

plt.show()

# 阶段1
idcheck_intervals = Interval(idcheck_times)
print('idcheck intervals: ', idcheck_intervals)

idcheck_mean = np.mean(np.array(idcheck_intervals))
idcheck_var = np.var(np.array(idcheck_intervals))
lam = 1 / idcheck_mean
k = 1 / (idcheck_var * lam * lam)
print('lambda1: ', lam)
print('k1: ', k)

x = np.linspace(0, 30, 90)
pdf = Erlang(x, np.round(k), lam)

plt.subplot(1, 3, 1)
plt.title('stage 1')
plt.plot(x, pdf, label='Erlang({:.0f}, {:.5f})'.format(np.round(k), lam))
plt.hist(idcheck_intervals, bins=10, density=True, range=(0, 30))
plt.legend()
plt.xlabel('t1 (sec)')

scanned_intervals = Interval(scanned_times)
print('scaned intervals: ', scanned_intervals)

scanned_mean = np.mean(np.array(scanned_intervals))
scanned_var = np.var(np.array(scanned_intervals))
lam = 1 / scanned_mean
k = 1 / (scanned_var * lam * lam)
print('lambda2: ', lam)
print('k2: ', k)

x = np.linspace(0, 80, 160)
plt.subplot(1, 3, 2)
plt.title('stage 2')
plt.plot(x, Erlang(x, np.round(k), lam), label='Erlang({:.0f}, {:.5f})'.format(np.round(k), lam))
plt.hist(scanned_intervals, bins=10, density=True, range=(0, 70))
plt.legend()
plt.xlabel('t2 (sec)')

mw_intervals = Stamp2Interval(mw_stamps)
print('mw_intervals: ', mw_intervals)

mw_mean = np.mean(np.array(mw_intervals))
mw_var = np.var(np.array(mw_intervals))
lam = 1 / mw_mean
k = 1 / (mw_var * lam * lam)
print('lambdamw: ', lam)
print('kmw: ', k)

x = np.linspace(0, 40, 80)
plt.subplot(1, 3, 3)
plt.title('mw')
plt.plot(x, Erlang(x, np.round(k), lam), label='Erlang({:.0f}, {:.5f})'.format(np.round(k), lam))
plt.hist(mw_intervals, bins=10, density=True, range=(0, 40))
plt.legend()
plt.xlabel('tmw (sec)')
plt.show()

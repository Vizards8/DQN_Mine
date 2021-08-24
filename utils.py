#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2021-05-04 19:58:31
Discription: 
Environment: 
'''
import os
import numpy as np
from pathlib import Path

def save_results(rewards,ma_rewards,tag='train',path='./results'):
    '''save rewards and ma_rewards
    '''
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('results saved!')
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:05:31 2022

@author: Liquid
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


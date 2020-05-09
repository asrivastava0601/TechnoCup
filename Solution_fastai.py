#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:47:59 2020

@author: abhinavsrivastava
"""

import numpy as np
import pandas as pd

import fastai

from fastai.vision import *


df1 = pd.read_csv('/kaggle/input/challengedata/Trainingcopy.csv')
df1.rename(columns = {"file":"Image"}, inplace = True)

df1['path1'] = 'Image-'

df1['jpg'] = '.jpg'

df1['path'] =  df1['path1'] + df1['Image'].astype(str) + df1['jpg']

df1 = df1.drop(['Image','path1','jpg'], axis = 1)

df1 = df1[['path','label']]

path_img = ('/kaggle/input/asdfgh/Training_Images/')


data = ImageDataBunch.from_df(path_img, df1, ds_tfms= get_transforms(do_flip = False), size = 450, valid_pct =0.2)


learn = cnn_learner(data, models.resnet34, metrics = [accuracy], pretrained = True)

learn.fit_one_cycle(50)

cm = ClassificationInterpretation.from_learner(learn)

cm.plot_confusion_matrix()

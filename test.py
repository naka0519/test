import requests
import zipfile
import io
import os
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
from matplotlib import animation
import random
import functools

import csv
import pathlib
import warnings
import datetime as dt
import glob
import time


################## csvファイル名を取得し、そのデータを変数として自動的に取得できるかのテスト ############################3
'''

# 空ファイル(dataframe)の作成 手動
column = [] # ここでカラムを入れておくことで、r_cam.csvが逐次更新でもカラム名を持っておくことができる #########未実装########
f_file = pd.DataFrame(data=[], columns= column)
f_file.to_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam1_forward.csv", mode = "w")
f_file.to_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam2_forward.csv", mode = "w")


path = "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam?.csv"
forward_path = "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam?_forward.csv"
csvpath_list = glob.glob(path)
forwardpath_list = glob.glob(forward_path)
csv_list = [os.path.basename(p) for p in csvpath_list]
forward_list = [os.path.basename(f) for f in forwardpath_list]
print(csvpath_list)
print(csv_list)

cam_num = 2 # カメラがふえれば追記すること

for i in range(cam_num):
    csv = csvpath_list[i]
    csv_list[i] = pd.read_csv(csv) # csvデータの読み込み
    #print(type(csv_list[i]))
    print(csv_list[i].columns)
    #os.remove(csvpath_list[i])  # tailが止まってしまうためこれではダメ
    csv_list[i]["cam"] = cam_num - i
    
#print(csv_list)

# frameの繰り越し
print(forward_list)
for i in range(len(csv_list)):
    csv_list[i] = pd.concat([forward_list[i], csv_list[i]], ignore_index = True, axis = 0)
os.remove(forward for forward in forwardpath_list) # 削除

# frame数の把握と調整
frame_list = [csv.iloc[-1]["frame"] for csv in csv_list] # frame:1~
print(frame_list)
min_frame = min(frame_list)
for i in range(len(frame_list)):
    #dif_frame = csv_list[i].iloc[-1]["frame"] - min_frame
    index = csv_list[i][csv_list[i]["frame"] == min_frame].index.values[0] + 1
    forward_csv = csv_list[i].iloc[index:]
    csv_list[i] = csv_list[i].iloc[:index]
    #print(index)
    print(forward_csv)
    #print(csv_list[i])
    forward_csv.to_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam{}_forward.csv".format(i), mode = "w")

df = pd.concat([r_cam for r_cam in csv_list], ignore_index = True, axis = 0) # 行方向に追加
df.to_csv("./headpose_result/testconcat.csv", mode = "w")
'''
##########################################

################## os.removeの挙動確認 ##############################
'''
# 自動バージョン 11/20
path = "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam?.csv"
csvpath_list = glob.glob(path)
csv_list = [os.path.basename(p) for p in csvpath_list]
cam_num = 2 ############## カメラがふえれば追記すること ###############
print(csvpath_list)

for i in range(cam_num):
    csv = csvpath_list[i]
    csv_list[i] = pd.read_csv(csv)
    os.remove(csvpath_list[i]) # ファイル削除によってtailがされていない？
    csv_list[i]["cam"] = cam_num - i # cam2->cam1の順にcsv_listには格納される

df = pd.concat([r_cam for r_cam in csv_list], ignore_index = True, axis = 0) # 行方向に追加
df.to_csv("./headpose_result/testconcat.csv", mode = "w")
'''
########################################################3
'''
##################### tailのチェック ######################
i = 0
while True:
    with open("test.csv", "a") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow("aaa{}".format(i))
    print(i)
    if (i == 10):
        os.remove("./tailtest.csv") # tailは続かなくなる
    i = i + 1
    time.sleep(2)

###########################################################
'''
'''
############# tail の試行錯誤 ########################
i = 0
while True:
    with open("test.csv", "a") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow("aaa{}".format(i))
    print(i)
    i = i + 1
    if (i == 10):
        #os.remove("./tailtest.csv") # tailは続かなくなる
        # 中身を白紙にする →　解決
        with open("tailtest.csv", "w") as f:
            f.write("")

    time.sleep(2)


###############################################
'''
'''
############# csvの読み込み方の確認
test = pd.read_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam2_forward.csv")
test = pd.read_csv("./a.csv") # 空
print(test.values)
################
'''
################## if の処理手順確認　##########
x = 10
if x == 10:
    print("10")
elif (x / 2) == 5:
    print("5")
###############################################

#################
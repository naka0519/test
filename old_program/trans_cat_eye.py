################## 複数台のカメラデータがまとまったcsvに対する平均や必要カラムの作成　################
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

import datetime as dt

# 動的な最新ファイル名を取得する方法(更新されたことをどう把握するか)
trans = pd.read_csv("./headpose_result/trans.csv")

cam_num = 3 ##################### カメラ台数 ############################  
df = [] # カメラごとのdataframeのまとめリスト
frame = [] # frame数のまとめリスト
# csvの読み込みとカラム整理
for i in range(cam_num):
    #df.append(pd.read_csv("1124_three_cap{}.csv".format(i + 1)))
    df.append(trans[trans["cam_id"] == i + 1])
    df[i] = df[i].reset_index(drop = True)
    df[i] = df[i].drop(["Unnamed: 0"], axis = 1)
    #df[i]["cam"] = i + 1 # カメラ番号 headpose.pyで作成
    frame.append((df[i].iloc[-1]["frame"]).astype(int)) # 各frame数の取得

print(df[1])
# 複数データに対するframe差の補完方法
# はじめと終わりのどちらにも差が生まれている状態
# 案：はじめか終わりのframe差を観測する
#  はじめ：openfaceの出力開始部分でdateを出力　
#  おわり：データを格納した直後のdateを比較することによるframe割り出し. 割り出し部分を削ってずれている分は開始時間によるずれ
#         -> 最小のframeに合わせて、余剰分を繰り越し(初めの割り出し分も)

column = ["frame", "timestamp", "confidence", "success", "head_loc_x", "head_loc_y", "head_loc_z", "head_rot_x", "head_rot_y", "head_rot_z", "eye_rot_x", "eye_rot_y", "eye_rot_z", "dif_xy", "dif_xz", "use_cam"]
df_cat = pd.DataFrame(data=[], columns= column) # result
# 最小frameに合わせてある(headpose.py)
min_frame = min(frame)

# 平均などによる出力値データの生成 すべてが成功しているのが前提
df_cat["frame"] = (df[0]["frame"])
#print(df_cat)
df_cat["timestamp"] = (df[0]["timestamp"])

data_count = [0 for _ in range(min_frame)] # 使用してるカメラデータ数
use_cam = ["" for _ in range(min_frame)] # 使用しているデータのcam_idの文字列で表示
for j in range(cam_num):
    count = 0
    for success in df[j]["success"]:
        if success == 1:
            use_cam[count] = use_cam[count] + str(j + 1)
            data_count[count] = data_count[count] + 1
        count = count + 1

# 平均による値の抽出 推定失敗しているデータはすべて0で補完されているとする
new_df_setframe = [0 for _ in range(cam_num)]
for i in range(cam_num):
    new_df_setframe[i] = df[i].drop(["frame", "face_id", "timestamp", "dif_xy", "dif_xz", "cam_id"], axis=1) # sumする必要のない要素を除外
#print(df[0])
#print(new_df_setframe[0])
SUM = sum(new_df_setframe) # listでまとめたdataframeの各要素の和

# 出力カメラ台数割合
print("data:{}\n3cam:{}\n2cam:{}\n1cam:{}\n0cam:{}".format(min_frame, data_count.count(3), data_count.count(2), data_count.count(1), data_count.count(0)))

# 計算エラー回避
for i in range(min_frame):
    if data_count[i] == 0:
        data_count[i] = 1

df_cat["confidence"] = (SUM["confidence"] / data_count)                    # もしここで例外処理(推定失敗など)をするなら、headposeではしなくていい?
df_cat["success"] = (SUM["success"] / data_count)                          # 判定方法はsuccessかconfidence値による判定
df_cat["head_loc_x"] = (SUM["head_loc_x"] / data_count)
df_cat["head_loc_y"] = (SUM["head_loc_y"] / data_count)
df_cat["head_loc_z"] = (SUM["head_loc_z"] / data_count)
df_cat["head_rot_x"] = (SUM["head_rot_x"] / data_count)
df_cat["head_rot_y"] = (SUM["head_rot_y"] / data_count)
df_cat["head_rot_z"] = (SUM["head_rot_z"]/ data_count)
df_cat["eye_rot_x"] = (SUM["eye_rot_x"] / data_count)
df_cat["eye_rot_y"] = (SUM["eye_rot_y"] / data_count)
df_cat["eye_rot_z"] = (SUM["eye_rot_z"] / data_count)
df_cat["dif_xy"] = (round(abs(SUM["head_rot_z"] - SUM["eye_rot_z"]), 4))
df_cat["dif_xz"] = (round(abs(SUM["head_rot_y"] - SUM["eye_rot_y"]), 4))
df_cat["use_cam"] = (use_cam)

print(df_cat)
df_cat.to_csv("./trans_result/trans_ave.csv", mode="w") # 逐次更新用
df_cat.to_csv("./trans_result/trans_ave_all.csv", mode="a") # データ追記して蓄積用
#now = dt.datetime.now()
#time = now.strftime('%Y%m%d-%H%M%S')
#df_cat.to_csv("./trans_result/trans_ave{}.csv".format(time), mode="w")



''' 二台限定のプログラム
# 動的な最新ファイル名を取得する方法(更新されたことをどう把握するか)
df = pd.read_csv("./headpose_result/trans.csv")


df = df.drop(['Unnamed: 0'], axis = 1)
# どのカメラ情報を可視化に用いるか(現在の座標・角度情報として用いるか)
df1 = df[df["cam_id"] == 1]
df2 = df[df["cam_id"] == 2]
#print(df1)
#print(df2)

# frame数が一番小さいものに合わせる カメラが三台以上の時どうやって判断するか
dif_frame = abs(df1.iloc[-1]["frame"] - df2.iloc[-1]["frame"]).astype(int)
df11 = df1.drop(range(0, dif_frame))
df11["frame"] = range(1, (df11.iloc[-1]["frame"] - df11.iloc[0]["frame"] + 1).astype(int) + 1)
df11 = df11.reset_index().drop("index", axis = 1)
#print(df11)
df_con = pd.merge(df11, df2, on="frame")
#print(df_con)

column = ["frame", "timestamp", "confidence", "success", "head_loc_x", "head_loc_y", "head_loc_z", "head_rot_x", "head_rot_y", "head_rot_z", "eye_rot_x", "eye_rot_y", "eye_rot_z", "dif_xy", "dif_xz"]
df_cat = pd.DataFrame(data=[], columns= column)

df_cat["frame"] = df_con["frame"] # +1
df_cat["timestamp"] = df_con["timestamp_y"]
df_cat["confidence"] = (df_con["confidence_x"] + df_con["confidence_y"]) / 2
df_cat["success"] = (df_con["success_x"] + df_con["success_y"]) // 2
df_cat["head_loc_x"] = (df_con["head_loc_x_x"] + df_con["head_loc_x_y"]) / 2
df_cat["head_loc_y"] = (df_con["head_loc_y_x"] + df_con["head_loc_y_y"]) / 2
df_cat["head_loc_z"] = (df_con["head_loc_z_x"] + df_con["head_loc_z_y"]) / 2
df_cat["head_rot_x"] = (df_con["head_rot_x_x"] + df_con["head_rot_x_y"]) / 2
df_cat["head_rot_y"] = (df_con["head_rot_y_x"] + df_con["head_rot_y_y"]) / 2
df_cat["head_rot_z"] = (df_con["head_rot_z_x"] + df_con["head_rot_z_y"]) / 2
df_cat["eye_rot_x"] = (df_con["eye_rot_x_x"] + df_con["eye_rot_x_y"]) / 2
df_cat["eye_rot_y"] = (df_con["eye_rot_y_x"] + df_con["eye_rot_y_y"]) / 2
df_cat["eye_rot_z"] = (df_con["eye_rot_z_x"] + df_con["eye_rot_z_y"]) / 2
df_cat["dif_xy"] = round((df_cat["head_rot_z"] - df_cat["eye_rot_z"]).abs(), 4)
df_cat["dif_xz"] = round(abs(df_cat["head_rot_y"] - df_cat["eye_rot_y"]), 4)

#print(df_con["frame"])
#print(df_cat)
#now = dt.datetime.now()
#time = now.strftime('%Y%m%d-%H%M%S')
#df_cat.to_csv("./trans_result/trans_ave{}.csv".format(time), mode="w")
df_cat.to_csv("./trans_result/trans_ave.csv", mode="w")
'''


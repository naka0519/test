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

def pre_process():
    cam_num = 2 ################ カメラがふえれば追記すること ################################
    trans = pd.read_csv("./headpose_result/headpose.csv")
    df = [] # カメラごとのdataframeのまとめリスト
    frame_list = [] # 最終frame番号のまとめリスト
    data_num = [] # データ数の確認
    # csvの読み込みとカラム整理
    for i in range(cam_num):
        #df.append(pd.read_csv("1124_three_cap{}.csv".format(i + 1)))
        df.append(trans[trans["cam_id"] == i + 1])
        df[i] = df[i].sort_values(by=["frame"]) # frame順に並び替え
        df[i] = df[i].reset_index(drop = True)
        df[i] = df[i].drop(["Unnamed: 0"], axis = 1)
        #df[i]["cam"] = i + 1 # カメラ番号 headpose.pyで作成
        frame_list.append((df[i].iloc[-1]["frame"]).astype(int)) # 各frame数の取得
        data_num.append(df[i]["frame"].idxmax() + 1) # 最終frameのindexを取得してデータ数を格納
        print(df[i])
        print(len(df))

    forward_path = "/home/ubuntu/workdir/processe/forwardcsv/r_cam?_forward.csv" # headposeで処理済みのデータが蓄えられる
    forwardpath_list = glob.glob(forward_path)
    forward_list = [os.path.basename(f) for f in forwardpath_list]
    for i in range(cam_num):
        forward_list[i] = pd.read_csv(forwardpath_list[i], index_col=0) # 繰り越しデータの読み込み
        with open(forwardpath_list[i], "w") as f: # 白紙
            f.write("")

    # frameの繰り越し
    #print(forwardpath_list[0])
    for i in range(cam_num):
        warnings.simplefilter("ignore", DeprecationWarning)
        if forward_list[i].values == []: # 繰越されるframeがあるかないかを判定する(データを持たないもの) 
            df[i] = df[i].reset_index(drop=True)
        else:
            df[i] = pd.concat([forward_list[i], df[i]], ignore_index = True, axis = 0)
            df[i] = df[i].sort_values(by=["frame"]) # frame順に整列
        #print("headpose{}:\n{}".format(i+1, csv_list[i]))

    ##### dfには新しい変換データと繰り越し変換データがカメラごとに含まれている まだデータ数などはバラバラでグレーム順#####

    # 最終frame番号を調節
    print("frame_list:{}".format(frame_list))
    min_frame = min(frame_list)
    for i in range(len(frame_list)): ####### frame番号で同期 ###########
        index = df[i][df[i]["frame"] == min_frame].index.values[0] + 1
        forward_csv = df[i].iloc[index:]
        df[i] = df[i].iloc[:index]
        # カラムの設定
        #forward_csv = forward_csv.drop(["Unnamed: 0"], axis = 1)
        #forward_csv = forward_csv.drop(["gaze_0_z"], axis = 1)
        forward_csv.to_csv("/home/ubuntu/workdir/processe/forwardcsv/r_cam{}_forward.csv".format(len(df) - i), mode = "w") 
    
    ######## 最終フレーム番号がそろった #########

    # index数を調べて抜けているframe番号の特定とcamera.csvの参照・データ取得

    

    # 処理を行うデータセットdf
    df = pd.concat([r_cam for r_cam in df], ignore_index = True, axis = 0) # 行方向に追加
    now = dt.datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    df.to_csv("/home/ubuntu/workdir/processe/csv_list/preprocess{}.csv".format(time), mode = "w")
    df.to_csv("./headpose_result/preprocess.csv", mode = "w") # 記録用

    return df




trans = pd.read_csv("./headpose_result/headpose.csv")

cam_num = 2 ##################### カメラ台数 ############################  
df = [] # カメラごとのdataframeのまとめリスト
frame = [] # frame数のまとめリスト
data_num = [] # データ数の確認
# csvの読み込みとカラム整理
for i in range(cam_num):
    #df.append(pd.read_csv("1124_three_cap{}.csv".format(i + 1)))
    df.append(trans[trans["cam_id"] == i + 1])
    df[i] = df[i].sort_values(by=["frame"]) # frame順に並び替え
    df[i] = df[i].reset_index(drop = True)
    df[i] = df[i].drop(["Unnamed: 0"], axis = 1)
    #df[i]["cam"] = i + 1 # カメラ番号 headpose.pyで作成
    frame.append((df[i].iloc[-1]["frame"]).astype(int)) # 各frame数の取得
    data_num.append(df[i]["frame"].idxmax() + 1) # 最終frameのindexを取得してデータ数を格納
    print(df[i])

column = ["frame", "timestamp", "confidence", "success", "head_loc_x", "head_loc_y", "head_loc_z", "head_rot_x", "head_rot_y", "head_rot_z", "use_cam"]
df_cat = pd.DataFrame(data=[], columns= column) # result

# 最小frameに合わせてある(headpose.py)
print("frame:{}".format(frame))
min_frame = min(frame) # 複数カメラ中の最小frame

######## データ損失などの時に備えたframe数の同期(最終フレーム番号で同期) ##########
min_data = min(data_num) # データ数の最小
dif_data = [(data_num[i] - min_data) for i in range(cam_num)]
print("dif_data:{}".format(dif_data))
if np.all(dif_data == 0):
    pass
else: # 0以外が含まれた場合
    df_set = []
    for i in range(cam_num):
        df_set.append(df[i][df[i]["frame"] > min_frame - min_data]) # 最小フレーム数に合わせるとともに、最小以外のデータは最新データからフレーム数を数える(最後で同期)
        df_set[i].reset_index(drop=True, inplace=True)# indexのリセット
        #print(df_set[i])
    df = df_set


df = preprocess()

# 平均などによる出力値データの生成 すべてが成功しているのが前提
df_cat["frame"] = (df[0]["frame"])
#print(df_cat)
df_cat["timestamp"] = (df[0]["timestamp"])

data_count = [0 for _ in range(min_data)] # 使用してるカメラデータ数
use_cam = ["" for _ in range(min_data)] # 使用しているデータのcam_idの文字列で表示
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
    new_df_setframe[i] = df[i].drop(["frame", "face_id", "timestamp", "cam_id"], axis=1) # sumする必要のない要素を除外
#print(df[0])
#print(new_df_setframe[0])
SUM = sum(new_df_setframe) # listでまとめたdataframeの各要素の和
#print(SUM)

# 出力カメラ台数割合
print("data:{} 3cam:{} 2cam:{} 1cam:{} 0cam:{}".format(min_data, data_count.count(3), data_count.count(2), data_count.count(1), data_count.count(0)))

# 計算エラー回避
for i in range(min_data):
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
df_cat["use_cam"] = (use_cam)

#print(df_cat)
#df_cat.to_csv("./trans_result/trans_ave.csv", mode="w") # 逐次更新用
df_cat.to_csv("./trans_result/trans_ave_all.csv", mode="a") # データ追記して蓄積用
now = dt.datetime.now()
time = now.strftime('%Y%m%d-%H%M%S')
df_cat.to_csv("./csv_list/trans_ave{}.csv".format(time), mode="w")

print("trans_cat fi")

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


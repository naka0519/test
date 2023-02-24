######################## 各カメラデータに対する座標変換とその前処理としてのframe同期 ##############################
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

#data = $(tail -f ~.csv ) 
# openfaceは逐次的にrtspからcsvファイルを吐き出している状態(追記)とする
# /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/processed/rtsp1.csv 

def pre_process():
    now = dt.datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    # frame同期自動バージョン 11/24
    # 空ファイル(dataframe)の作成 手動
    #'''
    column = ["frame", " face_id", " timestamp", " confidence", " success", " gaze_0_x", " gaze_0_y", "gaze_0_z", " gaze_1_x", " gaze_1_y", " gaze_1_z", " gaze_angle_x", " gaze_angle_y", " eye_lmk_x_0", " eye_lmk_x_1", " eye_lmk_x_2", " eye_lmk_x_3", " eye_lmk_x_4", " eye_lmk_x_5", " eye_lmk_x_6", " eye_lmk_x_7", " eye_lmk_x_8", " eye_lmk_x_9", " eye_lmk_x_10", " eye_lmk_x_11", " eye_lmk_x_12", " eye_lmk_x_13", " eye_lmk_x_14", " eye_lmk_x_15", " eye_lmk_x_16", " eye_lmk_x_17", " eye_lmk_x_18", " eye_lmk_x_19", " eye_lmk_x_20", " eye_lmk_x_21", " eye_lmk_x_22", " eye_lmk_x_23", " eye_lmk_x_24", " eye_lmk_x_25", " eye_lmk_x_26", " eye_lmk_x_27", " eye_lmk_x_28", " eye_lmk_x_29", " eye_lmk_x_30", " eye_lmk_x_31", " eye_lmk_x_32", " eye_lmk_x_33", " eye_lmk_x_34", " eye_lmk_x_35", " eye_lmk_x_36", " eye_lmk_x_37", " eye_lmk_x_38", " eye_lmk_x_39", " eye_lmk_x_40", " eye_lmk_x_41", " eye_lmk_x_42", " eye_lmk_x_43", " eye_lmk_x_44", " eye_lmk_x_45", " eye_lmk_x_46", " eye_lmk_x_47", " eye_lmk_x_48", " eye_lmk_x_49", " eye_lmk_x_50", " eye_lmk_x_51", " eye_lmk_x_52", " eye_lmk_x_53", " eye_lmk_x_54", " eye_lmk_x_55", " eye_lmk_y_0", " eye_lmk_y_1", " eye_lmk_y_2", " eye_lmk_y_3", " eye_lmk_y_4", " eye_lmk_y_5", " eye_lmk_y_6", " eye_lmk_y_7", " eye_lmk_y_8", " eye_lmk_y_9", " eye_lmk_y_10", " eye_lmk_y_11", " eye_lmk_y_12", " eye_lmk_y_13", " eye_lmk_y_14", " eye_lmk_y_15", " eye_lmk_y_16", " eye_lmk_y_17", " eye_lmk_y_18", " eye_lmk_y_19", " eye_lmk_y_20", " eye_lmk_y_21", " eye_lmk_y_22", " eye_lmk_y_23", " eye_lmk_y_24", " eye_lmk_y_25", " eye_lmk_y_26", " eye_lmk_y_27", " eye_lmk_y_28", " eye_lmk_y_29", " eye_lmk_y_30", " eye_lmk_y_31", " eye_lmk_y_32", " eye_lmk_y_33", " eye_lmk_y_34", " eye_lmk_y_35", " eye_lmk_y_36", " eye_lmk_y_37", " eye_lmk_y_38", " eye_lmk_y_39", " eye_lmk_y_40", " eye_lmk_y_41", " eye_lmk_y_42", " eye_lmk_y_43", " eye_lmk_y_44", " eye_lmk_y_45", " eye_lmk_y_46", " eye_lmk_y_47", " eye_lmk_y_48", " eye_lmk_y_49", " eye_lmk_y_50", " eye_lmk_y_51", " eye_lmk_y_52", " eye_lmk_y_53", " eye_lmk_y_54", " eye_lmk_y_55", " eye_lmk_X_0", " eye_lmk_X_1", " eye_lmk_X_2", " eye_lmk_X_3", " eye_lmk_X_4", " eye_lmk_X_5", " eye_lmk_X_6", " eye_lmk_X_7", " eye_lmk_X_8", " eye_lmk_X_9", " eye_lmk_X_10", " eye_lmk_X_11", " eye_lmk_X_12", " eye_lmk_X_13", " eye_lmk_X_14", " eye_lmk_X_15", " eye_lmk_X_16", " eye_lmk_X_17", " eye_lmk_X_18", " eye_lmk_X_19", " eye_lmk_X_20", " eye_lmk_X_21", " eye_lmk_X_22", " eye_lmk_X_23", " eye_lmk_X_24", " eye_lmk_X_25", " eye_lmk_X_26", " eye_lmk_X_27", " eye_lmk_X_28", " eye_lmk_X_29", " eye_lmk_X_30", " eye_lmk_X_31", " eye_lmk_X_32", " eye_lmk_X_33", " eye_lmk_X_34", " eye_lmk_X_35", " eye_lmk_X_36", " eye_lmk_X_37", " eye_lmk_X_38", " eye_lmk_X_39", " eye_lmk_X_40", " eye_lmk_X_41", " eye_lmk_X_42", " eye_lmk_X_43", " eye_lmk_X_44", " eye_lmk_X_45", " eye_lmk_X_46", " eye_lmk_X_47", " eye_lmk_X_48", " eye_lmk_X_49", " eye_lmk_X_50", " eye_lmk_X_51", " eye_lmk_X_52", " eye_lmk_X_53", " eye_lmk_X_54", " eye_lmk_X_55", " eye_lmk_Y_0", " eye_lmk_Y_1", " eye_lmk_Y_2", " eye_lmk_Y_3", " eye_lmk_Y_4", " eye_lmk_Y_5", " eye_lmk_Y_6", " eye_lmk_Y_7", " eye_lmk_Y_8", " eye_lmk_Y_9", " eye_lmk_Y_10", " eye_lmk_Y_11", " eye_lmk_Y_12", " eye_lmk_Y_13", " eye_lmk_Y_14", " eye_lmk_Y_15", " eye_lmk_Y_16", " eye_lmk_Y_17", " eye_lmk_Y_18", " eye_lmk_Y_19", " eye_lmk_Y_20", " eye_lmk_Y_21", " eye_lmk_Y_22", " eye_lmk_Y_23", " eye_lmk_Y_24", " eye_lmk_Y_25", " eye_lmk_Y_26", " eye_lmk_Y_27", " eye_lmk_Y_28", " eye_lmk_Y_29", " eye_lmk_Y_30", " eye_lmk_Y_31", " eye_lmk_Y_32", " eye_lmk_Y_33", " eye_lmk_Y_34", " eye_lmk_Y_35", " eye_lmk_Y_36", " eye_lmk_Y_37", " eye_lmk_Y_38", " eye_lmk_Y_39", " eye_lmk_Y_40", " eye_lmk_Y_41", " eye_lmk_Y_42", " eye_lmk_Y_43", " eye_lmk_Y_44", " eye_lmk_Y_45", " eye_lmk_Y_46", " eye_lmk_Y_47", " eye_lmk_Y_48", " eye_lmk_Y_49", " eye_lmk_Y_50", " eye_lmk_Y_51", " eye_lmk_Y_52", " eye_lmk_Y_53", " eye_lmk_Y_54", " eye_lmk_Y_55", " eye_lmk_Z_0", " eye_lmk_Z_1", " eye_lmk_Z_2", " eye_lmk_Z_3", " eye_lmk_Z_4", " eye_lmk_Z_5", " eye_lmk_Z_6", " eye_lmk_Z_7", " eye_lmk_Z_8", " eye_lmk_Z_9", " eye_lmk_Z_10", " eye_lmk_Z_11", " eye_lmk_Z_12", " eye_lmk_Z_13", " eye_lmk_Z_14", " eye_lmk_Z_15", " eye_lmk_Z_16", " eye_lmk_Z_17", " eye_lmk_Z_18", " eye_lmk_Z_19", " eye_lmk_Z_20", " eye_lmk_Z_21", " eye_lmk_Z_22", " eye_lmk_Z_23", " eye_lmk_Z_24", " eye_lmk_Z_25", " eye_lmk_Z_26", " eye_lmk_Z_27", " eye_lmk_Z_28", " eye_lmk_Z_29", " eye_lmk_Z_30", " eye_lmk_Z_31", " eye_lmk_Z_32", " eye_lmk_Z_33", " eye_lmk_Z_34", " eye_lmk_Z_35", " eye_lmk_Z_36", " eye_lmk_Z_37", " eye_lmk_Z_38", " eye_lmk_Z_39", " eye_lmk_Z_40", " eye_lmk_Z_41", " eye_lmk_Z_42", " eye_lmk_Z_43", " eye_lmk_Z_44", " eye_lmk_Z_45", " eye_lmk_Z_46", " eye_lmk_Z_47", " eye_lmk_Z_48", " eye_lmk_Z_49", " eye_lmk_Z_50", " eye_lmk_Z_51", " eye_lmk_Z_52", " eye_lmk_Z_53", " eye_lmk_Z_54", " eye_lmk_Z_55", " pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz", " x_0", " x_1", " x_2", " x_3", " x_4", " x_5", " x_6", " x_7", " x_8", " x_9", " x_10", " x_11", " x_12", " x_13", " x_14", " x_15", " x_16", " x_17", " x_18", " x_19", " x_20", " x_21", " x_22", " x_23", " x_24", " x_25", " x_26", " x_27", " x_28", " x_29", " x_30", " x_31", " x_32", " x_33", " x_34", " x_35", " x_36", " x_37", " x_38", " x_39", " x_40", " x_41", " x_42", " x_43", " x_44", " x_45", " x_46", " x_47", " x_48", " x_49", " x_50", " x_51", " x_52", " x_53", " x_54", " x_55", " x_56", " x_57", " x_58", " x_59", " x_60", " x_61", " x_62", " x_63", " x_64", " x_65", " x_66", " x_67", " y_0", " y_1", " y_2", " y_3", " y_4", " y_5", " y_6", " y_7", " y_8", " y_9", " y_10", " y_11", " y_12", " y_13", " y_14", " y_15", " y_16", " y_17", " y_18", " y_19", " y_20", " y_21", " y_22", " y_23", " y_24", " y_25", " y_26", " y_27", " y_28", " y_29", " y_30", " y_31", " y_32", " y_33", " y_34", " y_35", " y_36", " y_37", " y_38", " y_39", " y_40", " y_41", " y_42", " y_43", " y_44", " y_45", " y_46", " y_47", " y_48", " y_49", " y_50", " y_51", " y_52", " y_53", " y_54", " y_55", " y_56", " y_57", " y_58", " y_59", " y_60", " y_61", " y_62", " y_63", " y_64", " y_65", " y_66", " y_67", " X_0", " X_1", " X_2", " X_3", " X_4", " X_5", " X_6", " X_7", " X_8", " X_9", " X_10", " X_11", " X_12", " X_13", " X_14", " X_15", " X_16", " X_17", " X_18", " X_19", " X_20", " X_21", " X_22", " X_23", " X_24", " X_25", " X_26", " X_27", " X_28", " X_29", " X_30", " X_31", " X_32", " X_33", " X_34", " X_35", " X_36", " X_37", " X_38", " X_39", " X_40", " X_41", " X_42", " X_43", " X_44", " X_45", " X_46", " X_47", " X_48", " X_49", " X_50", " X_51", " X_52", " X_53", " X_54", " X_55", " X_56", " X_57", " X_58", " X_59", " X_60", " X_61", " X_62", " X_63", " X_64", " X_65", " X_66", " X_67", " Y_0", " Y_1", " Y_2", " Y_3", " Y_4", " Y_5", " Y_6", " Y_7", " Y_8", " Y_9", " Y_10", " Y_11", " Y_12", " Y_13", " Y_14", " Y_15", " Y_16", " Y_17", " Y_18", " Y_19", " Y_20", " Y_21", " Y_22", " Y_23", " Y_24", " Y_25", " Y_26", " Y_27", " Y_28", " Y_29", " Y_30", " Y_31", " Y_32", " Y_33", " Y_34", " Y_35", " Y_36", " Y_37", " Y_38", " Y_39", " Y_40", " Y_41", " Y_42", " Y_43", " Y_44", " Y_45", " Y_46", " Y_47", " Y_48", " Y_49", " Y_50", " Y_51", " Y_52", " Y_53", " Y_54", " Y_55", " Y_56", " Y_57", " Y_58", " Y_59", " Y_60", " Y_61", " Y_62", " Y_63", " Y_64", " Y_65", " Y_66", " Y_67", " Z_0", " Z_1", " Z_2", " Z_3", " Z_4", " Z_5", " Z_6", " Z_7", " Z_8", " Z_9", " Z_10", " Z_11", " Z_12", " Z_13", " Z_14", " Z_15", " Z_16", " Z_17", " Z_18", " Z_19", " Z_20", " Z_21", " Z_22", " Z_23", " Z_24", " Z_25", " Z_26", " Z_27", " Z_28", " Z_29", " Z_30", " Z_31", " Z_32", " Z_33", " Z_34", " Z_35", " Z_36", " Z_37", " Z_38", " Z_39", " Z_40", " Z_41", " Z_42", " Z_43", " Z_44", " Z_45", " Z_46", " Z_47", " Z_48", " Z_49", " Z_50", " Z_51", " Z_52", " Z_53", " Z_54", " Z_55", " Z_56", " Z_57", " Z_58", " Z_59", " Z_60", " Z_61", " Z_62", " Z_63", " Z_64", " Z_65", " Z_66", " Z_67", " p_scale", " p_rx", " p_ry", " p_rz", " p_tx", " p_ty", " p_0", " p_1", " p_2", " p_3", " p_4", " p_5", " p_6", " p_7", " p_8", " p_9", " p_10", " p_11", " p_12", " p_13", " p_14", " p_15", " p_16", " p_17", " p_18", " p_19", " p_20", " p_21", " p_22", " p_23", " p_24", " p_25", " p_26", " p_27", " p_28", " p_29", " p_30", " p_31", " p_32", " p_33", " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r", " AU01_c", " AU02_c", " AU04_c", " AU05_c", " AU06_c", " AU07_c", " AU09_c", " AU10_c", " AU12_c", " AU14_c", " AU15_c", " AU17_c", " AU20_c", " AU23_c", " AU25_c", " AU26_c", " AU28_c", " AU45_c"]

    path = "/home/ubuntu/workdir/processe/rtspcsv/r_cam?.csv"
    forward_path = "/home/ubuntu/workdir/processe/forwardcsv/r_cam?_forward.csv"
    #path = "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam?.csv"
    #forward_path = "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam?_forward.csv"
    csvpath_list = glob.glob(path)
    forwardpath_list = glob.glob(forward_path)
    csv_list = [os.path.basename(p) for p in csvpath_list]
    forward_list = [os.path.basename(f) for f in forwardpath_list]
    #print(csvpath_list)
    #print(csv_list)

    cam_num = 3 ################ カメラがふえれば追記すること ################################

    for i in range(cam_num):
        csvpath = csvpath_list[i]
        csv_list[i] = pd.read_csv(csvpath) # csvデータの読み込み
        #''' 生データを見たいときはここをコメントアウト
        with open(csvpath, "w") as f: # csvを白紙に(データ削除) カラムの設定だけしたい ############ 1frameくらい誤消去がおきる
            #f.write("")
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(column) # カラムの再設定
        #'''
        #print(csv_list[i].columns)
        forward_list[i] = pd.read_csv(forwardpath_list[i], index_col=0) # 繰り越しデータの読み込み
        with open(forwardpath_list[i], "w") as f: # 白紙
            f.write("")

        cam_id = int(csvpath[-5])
        csv_list[i]["cam"] = cam_id # カメラ番号の付与
        csv_list[i].to_csv("/home/ubuntu/workdir/processe/csv_list/r_cam{}:{}.csv".format(cam_id, time), mode = "w")
        
    print(csv_list)
    print(forward_list)

    # frameの繰り越し
    #print(forwardpath_list[0])
    for i in range(len(csv_list)):
        warnings.simplefilter("ignore", DeprecationWarning)
        cam_id = csv_list[i].iloc[0]["cam"]
        if forward_list[i].values == []: # 繰越されるframeがあるかないかを判定する(データを持たないもの) 
            csv_list[i] = csv_list[i].reset_index(drop=True)
        else:
            csv_list[i] = pd.concat([forward_list[i], csv_list[i]], ignore_index = True, axis = 0) # 結合しているデータgcam_idで確かめられてない
            csv_list[i] = csv_list[i].sort_values(by=["frame"]) # frame順に整列
            csv_list[i].to_csv("/home/ubuntu/workdir/processe/csv_list/csv_list{}:{}.csv".format(cam_id, time), mode = "w")
        #print("headpose{}:\n{}".format(i+1, csv_list[i]))

    
    print(csv_list[0])
    print(csv_list[1])
    print(csv_list[2])

    # 最終frame番号の把握と調整    : 一回目で開始時のframeを合わせる必要性がある
    frame_last = np.array([csv.iloc[-1]["frame"] for csv in csv_list]) # frame:1~
    data_num = [] # データ数の確認
    print("frame_last:{}".format(frame_last))
    min_frame = min(frame_last)
    for i in range(len(frame_last)): ####### frame番号で同期 ###########
        #dif_frame = csv_list[i].iloc[-1]["frame"] - min_frame
        #print("cam:{}".format(csv_list[i]["cam"][0]))
        cam_id = int(csv_list[i].iloc[0]["cam"])
        index = csv_list[i][csv_list[i]["frame"] == min_frame].index.values[0] + 1
        forward_csv = csv_list[i].iloc[index:]
        csv_list[i] = csv_list[i].iloc[:index]
        csv_list[i] = csv_list[i].reset_index(drop=True)
        data_num.append((csv_list[i]["frame"]).astype(int).idxmax() + 1) # 最終frameのindexを取得してデータ数を格納
        # カラムの設定
        #forward_csv = forward_csv.drop(["Unnamed: 0"], axis = 1)
        #forward_csv = forward_csv.drop(["gaze_0_z"], axis = 1)
        forward_csv.to_csv("/home/ubuntu/workdir/processe/csv_list/r_cam{}_forward{}.csv".format(cam_id, time), mode = "w")
        forward_csv.to_csv("/home/ubuntu/workdir/processe/forwardcsv/r_cam{}_forward.csv".format(cam_id), mode = "w") #, index=False) 
    
    f_last = np.array([csv.iloc[-1]["frame"] for csv in csv_list]) # frame:1~
    f_start = np.array([csv.iloc[0]["frame"] for csv in csv_list])
    frame_num = f_last - f_start + np.array([1] * len(frame_last))
    print("frame_num:{}".format(frame_num))
    print("data_num:{}".format(data_num))

    # frame番号の欠損がないか確かめて、ある場合はcamera.csvから引き抜いて加える
    for i in range(len(data_num)):
        dif_num = frame_num[i] - data_num[i]
        plus_data = pd.DataFrame(data=[], columns= column)
        if dif_num != 0: # データ数と最終フレーム番号が一致するか
            print("frame欠損!!!")
            df = csv_list[i]
            cam_id = int(df.iloc[0]["cam"])
            count =0
            for j in range(data_num[i] - 1):
                dif = int(df.iloc[j+1]["frame"] - df.iloc[j]["frame"])
                if dif != 1:
                    print("cam_id:{} {} : {}".format(cam_id, df.iloc[j+1]["frame"], df.iloc[j]["frame"]))
                    for jj in range (dif - 1):
                        addframe = df.iloc[j]["frame"] + (jj + 1)
                        ##### addframe番号をcamera{cam_id}.csvから取得 #####  データ数が増えるほど処理が多くなるぞ
                        raw_path = "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera{}.csv".format(cam_id)
                        raw_data = pd.read_csv(raw_path)
                        data = raw_data[raw_data["frame"] == addframe]
                        plus_data = pd.concat([plus_data, data], ignore_index = True, axis = 0)
                        count = count + 1
                if count == dif_num:
                    break
        csv_list[i] = pd.concat([csv_list[i], plus_data], ignore_index = True, axis = 0)
        csv_list[i] = csv_list[i].sort_values(by=["frame"]) # frame順に整列

    ######### frameとデータ数がそろったcsv_listが完成 ###########
    #'''

    '''
    # 自動バージョン 11/20 -> 更新されるtail csvにカラムがないためエラー
    path = "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam?.csv"
    csvpath_list = glob.glob(path)
    csv_list = [os.path.basename(p) for p in csvpath_list]
    cam_num = 2 ############## カメラがふえれば追記すること ###############
    print(csvpath_list)

    for i in range(cam_num):
        csv = csvpath_list[i]
        csv_list[i] = pd.read_csv(csv)
        with open(csvpath_list[i], "w") as f: # csvを白紙に(データ削除)
            f.write("")
        csv_list[i]["cam"] = cam_num - i # cam2->cam1の順にcsv_listには格納される

    '''

    '''
    # 手動バージョン11/10
    #r_cam1 = pd.read_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/processed/two_cap/camera1.csv") # テスト用
    r_cam1 = pd.read_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam1.csv")
    #os.remove("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam1.csv")
    #r_cam2 = pd.read_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/processed/two_cap/camera2.csv") # テスト用
    r_cam2 = pd.read_csv("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam2.csv")
    #os.remove("/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam2.csv")
    
    r_cam1["cam"] = 1
    r_cam2["cam"] = 2
    '''
    # 処理を行うデータセットdf
    df = pd.concat([r_cam for r_cam in csv_list], ignore_index = True, axis = 0) # 行方向に追加
    df.to_csv("/home/ubuntu/workdir/processe/csv_list/preprocess{}.csv".format(time), mode = "w")
    df.to_csv("./headpose_result/preprocess.csv", mode = "w") # 記録用

    return df
 
def main():
    # tailのデータを取得した時間を保存(保存ファイル名として使う)
    #now = dt.datetime.now()
    
    df = pre_process() # return df
    #print(df.columns)

    # 部屋の座標系
    room_xyz = [0, 0, 0]
        
    ############### カメラの座標(mm)、向き(°) 部屋の座標系 #################### 追記すること
    camera1_xyz = [0, 0, 1500]
    camera2_xyz = [-50, 3000, 1500]
    camera3_xyz = [3000, 100, 1500]
    #camera4_xyz = []
    camera1_gaze = [0, 0, 45] # カメラは地面に水平で、垂直軸に対して45度反時計回転している
    camera2_gaze = [0, 0, 315]
    camera3_gaze = [0, 0, 135]
    #camera4_gaze = []  

    camera_xyz = [camera1_xyz, camera2_xyz, camera3_xyz]  # リストにして条件分岐で書きやすいようにする
    camera_gaze = [camera1_gaze, camera2_gaze, camera3_gaze]

    #########################################################################

    # 取得csvからカメラの座標を用いて、部屋の座標系で顔の向き(注目物体)を出力
    # 室内座標系にける、顔の位置と向きを出力->特定物体の注目の有無を検知
    #last_frame = df.iloc[-1]["frame"]
    last_index = df.index.tolist()[-1]   # 複数カメラデータを結合したもののデータ数
    #print(last_index)
    num = last_index + 1
    #num = last_index.astype(int)
    
    # 必要なカラムの設定
    column = ["frame", "cam_id", "face_id", "timestamp", "confidence", "success", "head_loc_x", "head_loc_y", "head_loc_z", "head_rot_x", "head_rot_y", "head_rot_z"] #,"eye_rot_x", "eye_rot_y", "eye_rot_z", "dif_xy", "dif_xz"]
    result = pd.DataFrame(data=[], columns= column)
    #result = pd.DataFrame(data=[], columns= ["head_rot_x", "head_rot_y", "head_rot_z"])
    #print(result)

    for i in range(num):
        frame = df.iloc[i]["frame"]
        cam_id = df.iloc[i]["cam"]
        face_id = df.iloc[i][" face_id"]
        timestamp = df.iloc[i][" timestamp"]
        confidence = df.iloc[i][" confidence"]
        success = df.iloc[i][" success"]
        
        # 推定が失敗した時
        if ((df.iloc[i][" success"] < 0.75)): #or (df.iloc[i][" pose_Rx"] >= 0.9) or (df.iloc[i][" pose_Ry"] >= 0.9)):
            success = 0 # 不成功とする data数把握に使うため 
            # すべての値を-1000で置換
            head_loc_x = 0 #-1000
            head_loc_y = 0 #-1000
            head_loc_z = 0 #-1000
            head_rot_x = 0 #-1000
            head_rot_y = 0 #-1000
            head_rot_z = 0 #-1000
            
            # 目線(保留：目線の精度は？)
            #eye_rot_x = 0 #-1000
            #eye_rot_y = 0 #-1000
            #eye_rot_z = 0 #-1000            
            
            #dif_xy = 0
            #dif_xz = 0
            
            S_data = pd.DataFrame(data = [[frame, cam_id, face_id, timestamp, confidence, success, head_loc_x, head_loc_y, head_loc_z, head_rot_x, head_rot_y, head_rot_z]], columns = column) #, eye_rot_x, eye_rot_y, eye_rot_z, dif_xy, dif_xz]], columns = column)
            #result = result.append(S_data ,ignore_index = True)
            result = pd.concat([result, S_data], ignore_index = True, axis = 0)
            
        elif abs(df.iloc[i][" pose_Rx"]) > 1.5 or abs(df.iloc[i][" pose_Ry"]) > 1.5: # 外れ値のデータを除外 
            success = 0 # 不成功とする data数把握に使うため 
            # すべての値を-1000で置換
            head_loc_x = 0 #-1000
            head_loc_y = 0 #-1000
            head_loc_z = 0 #-1000
            head_rot_x = 0 #-1000
            head_rot_y = 0 #-1000
            head_rot_z = 0 #-1000
            
            # 目線(保留：目線の精度は？)
            #eye_rot_x = 0 #-1000
            #eye_rot_y = 0 #-1000
            #eye_rot_z = 0 #-1000
            
            #dif_xy = 0
            #dif_xz = 0
            
            S_data = pd.DataFrame(data = [[frame, cam_id, face_id, timestamp, confidence, success, head_loc_x, head_loc_y, head_loc_z, head_rot_x, head_rot_y, head_rot_z]], columns = column)#, eye_rot_x, eye_rot_y, eye_rot_z, dif_xy, dif_xz]], columns = column)
            #result = result.append(S_data ,ignore_index = True)
            result = pd.concat([result, S_data], ignore_index = True, axis = 0)

        # 推定が成功した時
        else:
            # head_loc を鼻の頭とする(顔の向きにおいて、鼻は自由度をもたずに向きと一致すると考える) ()
            # カメラ座標系での顔座標((X,Y)は平面, Zは距離)
            head_loc_cx = df.iloc[i][" X_33"]
            head_loc_cy = df.iloc[i][" Y_33"]
            head_loc_cz = df.iloc[i][" Z_33"]
            
            # 顔の向き・角度(カメラ座標)   pose_Rの有効数字は三桁 
            c_hx = df.iloc[i][" pose_Rx"] * 90.0 # -90(上を向く)~+90(下を向く) cameraに対して
            c_hy = df.iloc[i][" pose_Ry"] * 90.0 # -90(左を向く)~+90(右を向く)
            c_hz = -1000.0                           # +(右にかしげる)         #不要                                         
            
            # camera情報の確認
            cam = int(cam_id - 1)

            # カメラは右手座標系でz軸回りの回転α=camera1_gaze[2], y軸回りでの回転β=camera1_gaze[1]のみと仮定する
            head_loc_x = camera_xyz[cam][0] + head_loc_cz * np.cos(np.radians(camera_gaze[cam][1])) * np.cos(np.radians(camera_gaze[cam][2])) + head_loc_cy * np.sin(np.radians(camera_gaze[cam][1])) * np.cos(np.radians(camera_gaze[cam][2])) + head_loc_cx * np.sin(np.radians(camera_gaze[cam][2]))
            head_loc_y = camera_xyz[cam][1] + head_loc_cz * np.cos(np.radians(camera_gaze[cam][1])) * np.sin(np.radians(camera_gaze[cam][2])) + head_loc_cy * np.sin(np.radians(camera_gaze[cam][1])) * np.sin(np.radians(camera_gaze[cam][2])) + (-1) * head_loc_cx * np.cos(np.radians(camera_gaze[cam][2]))
            head_loc_z = camera_xyz[cam][2] + head_loc_cz * np.sin(np.radians(camera_gaze[cam][1])) + (-1) * head_loc_cy * np.cos(np.radians(camera_gaze[cam][1]))
            
            # X軸上に顔がある時の回転軸を基準とする. 軸に対して反時計回りを正とする.
            # 顔の座標から、部屋の原点room_xyz の向きを回転の0度とする
            # カメラごとに軸を用意して計算し、その結果を平行移動・変換をして部屋座標系にすればよい
            # camera1(原点上)にあるときを基本としてできれば、あとは90°回転ごとのカメラ軸をとったものを想定すれば以下と同じ処理で済む
            
            # カメラが側転方向の傾きを持たないとき
            head_rot_x = c_hz # 首傾げ
            # 頷き
            head_rot_y = c_hx # -90~+90が定義域 カメラから見た定義域 可視化の際に注意が必要
            
            # 部屋座標の原点と顔の座標の(x, y)平面上での角度Θ=rot_xy
            #rot_xy = np.degrees(np.arctan(head_loc_y / head_loc_x)) 
            
            # 振り向き
            head_rot_z = camera_gaze[cam][2] + 180 - c_hy # camera_xyzが座標の原点の時

            # 複数データ統合時の数値エラー回避 
            if head_rot_y > 360:
                head_rot_y = head_rot_y - 360
            if head_rot_z > 360:
                head_rot_z = head_rot_z - 360
            
            warnings.simplefilter("ignore", FutureWarning)
            S_data = pd.DataFrame(data = [[frame, cam_id, face_id, timestamp, confidence, success, head_loc_x, head_loc_y, head_loc_z, head_rot_x, head_rot_y, head_rot_z]], columns = column) #, eye_rot_x, eye_rot_y, eye_rot_z, dif_xy, dif_xz]], columns = column)        
            #result = result.append(S_data ,ignore_index = True)
            result = pd.concat([result, S_data], ignore_index = True, axis = 0)
    
    #print(result)

    # データの保存
    now = dt.datetime.now()
    time = now.strftime('%Y%m%d-%H%M%S')
    result.to_csv("/home/ubuntu/workdir/processe/csv_list/headpose{}.csv".format(time), mode = "w")
    result.to_csv("/home/ubuntu/workdir/processe/headpose_result/headpose.csv", mode = "w")


    print("finish headpose.py")
    

# 処理のコンパクト化 関数の作成、リストにまとめる
# 複数台カメラの結果に対してこれらの処理を行う時、
# 複数のdfがある。
# 方法１：一つのcsvにカメラのラベルをつけてまとめる
#    問題：補完を行うための時間同期が必要。まったく同じ時間のものはないため、丸めかたも考えなければならない
# 

if __name__ == '__main__':
    main()





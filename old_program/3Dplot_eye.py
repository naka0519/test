# 以下のライブラリを使うので、あらかじめ読み込んでおいてください
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import functools

# 可視化するCSVを選択 動的なファイル名を取得

df = pd.read_csv("./trans_result/trans_ave.csv")
print(df)
df = df.drop(["Unnamed: 0"], axis = 1)
print(df)


fig= plt.figure()
ax = fig.add_subplot(projection='3d')

last_frame = df.iloc[-1]["frame"]
num = last_frame.astype(int)
#print(num)

def update(frame, res):
    ax.cla()
    
    x0 = res.iloc[frame]["head_loc_x"]
    y0 = res.iloc[frame]["head_loc_y"]
    z0 = res.iloc[frame]["head_loc_z"]
    y = []
    z = []
    # ｘの間隔
    xf = 100
    
    
    # 頭の回転
    if res.iloc[frame]["head_rot_x"] == 0:
        print("no data")
        x = np.linspace(0, 5000, 2)
        y = [y0, y0]
        z = [z0, z0]
        ax.plot(x, y, z, color="red")
    elif (0 < res.iloc[frame]["head_rot_z"] and res.iloc[frame]["head_rot_z"] < 90) or (270 < res.iloc[frame]["head_rot_z"] and res.iloc[frame]["head_rot_z"] <360):
        x = np.arange(x0, 5000, xf)  # 適当に動かす
        y = np.tan(np.radians(res.iloc[frame]["head_rot_z"])) * (x - x0) + y0
        z = np.tan(np.radians(res.iloc[frame]["head_rot_y"])) * (x - x0) + z0
        
        # 数値のバグ補正用コード
        for i in range(int((5000 - x0) // xf)):
            if y[i] < 0 or z[i] < 0:
                '''
                x[:i] = 0
                y[:i] = 0
                z[:i] = 0
                '''
                x = [x[i], x0] # x[i-1]とどっちがいいか
                y = [y[i], y0]
                z = [z[i], z0]
                break
            else:
                continue
        
        ax.plot(x, y, z, color='blue', label='head')
    elif 90 < res.iloc[frame]["head_rot_z"] and res.iloc[frame]["head_rot_z"] < 270:
        x = np.arange(0, x0, xf)  # 適当に動かす
        y = np.tan(np.radians(res.iloc[frame]["head_rot_z"])) * (x - x0) + y0
        #z = np.tan(np.radians(res.iloc[frame]["head_rot_y"])) * (x - x0) + z0
        z = np.tan(np.radians(res.iloc[frame]["head_rot_y"] + 180)) * (x - x0) + z0
        
        # 数値のバグ補正用コード
        for i in range(1, int(x0 // xf)): #xに同期
            if y[i] > 0 and z[i] > 0: # and
                '''
                x[:i] = 0
                y[:i] = 0
                z[:i] = 0
                '''
                x = [x[i-1], x0]
                y = [y[i-1], y0]
                z = [z[i-1], z0]
                #print(i)
                break
            else:
                continue
        
        #print("rot_z:{}".format(res.iloc[frame]["head_rot_z"]), "radian_z:{}".format(np.radians(res.iloc[frame]["head_rot_z"])), "x-y:{}".format(np.tan(np.radians(res.iloc[frame]["head_rot_z"]))))
        #print("x:{}".format(x), "y:{}".format(y))
        #print("z:{}".format(z))
        ax.plot(x, y, z, color="blue", label="head")
    else:
        print("90° error") 
        x = np.linspace(0, 5000, 2)
        y = [y0, y0]
        z = [z0, z0]
        ax.plot(x, y, z, color="red")

    # 目線バージョン
    '''
    if res.iloc[frame]["eye_rot_y"] == -1000:
        print("no data")
        xe = np.linspace(0, 5000, 2)
        ye = [y0, y0]
        ze = [z0, z0]
        ax.plot(xe, ye, ze, color="red")
    elif (0 < res.iloc[frame]["eye_rot_y"] and res.iloc[frame]["eye_rot_y"] < 90) or (270 < res.iloc[frame]["eye_rot_y"] and res.iloc[frame]["eye_rot_y"] <360):
        xe = np.arange(x0, 5000, xf)  # 適当に動かす
        ye = np.tan(np.radians(res.iloc[frame]["eye_rot_z"])) * (xe - x0) + y0
        ze = np.tan(np.radians(res.iloc[frame]["eye_rot_y"])) * (xe - x0) + z0
        
        for i in range(int((5000 - x0) // xf)):
            if ye[i] < 0 or ze[i] < 0:
                xe = [xe[i], x0] # x[i-1]とどっちがいいか
                ye = [ye[i], y0]
                ze = [ze[i], z0]
                break;
            else:
                continue;
        
        ax.plot(xe, ye, ze, color='green', label='eye')
    elif 90 < res.iloc[frame]["eye_rot_y"] and res.iloc[frame]["eye_rot_y"] < 270:
        xe = np.arange(0, x0, xf)  # 適当に動かす
        ye = np.tan(np.radians(res.iloc[frame]["eye_rot_z"])) * (xe - x0) + y0
        ze = np.tan(np.radians(res.iloc[frame]["eye_rot_y"])) * (xe - x0) + z0
        
        for i in range(int(x0 // xf)):
            if ye[i] > 0 and ze[i] > 0: # and
                xe = [xe[i] - 1, x0]
                ye = [ye[i] - 1, y0]
                ze = [ze[i] - 1, z0]
                break;
            else:
                continue;
        
        ax.plot(xe, ye, ze, color='green', label='eye')
    else:
        print("90° error") # -1000も拾ってる
        xe = np.linspace(0, 5000, 2)
        ye = [y0, y0]
        ze = [z0, z0]
        ax.plot(xe, ye, ze, color="red")
    ''' 
    #顔と目線の角度差を表示する データの計算はheadpose.pyでやるべき。処理が長い
    '''
    dif_xy = res.iloc[frame]["dif_xy"]
    dif_xz = res.iloc[frame]["dif_xz"]
    #print(x)
    vec_head = [x[0] - x[1], y[0] - y[1], z[0] - x[1]]
    vec_eye = [xe[0] - xe[1], ye[0] - ye[1], ze[0] - ze[1]]
    dif_rot = (vec_head[0] * vec_eye[0] + vec_head[1] * vec_eye[1] + vec_head[2] * vec_eye[2])\
              / (np.sqrt(pow(vec_head[0], 2) + pow(vec_head[1], 2) + pow(vec_head[2], 2)) * np.sqrt(pow(vec_eye[0], 2) + pow(vec_eye[1], 2) + pow(vec_eye[2], 2)))
    dif_rot = round(dif_rot, 4)
    txt_xy = "difference x_y:{}°"
    txt_xz = "difference x_z:{}°"
    txt_cos = "difference cos:{}"
    ax.text(-1500, 1500, 4200, txt_xy.format(dif_xy), fontsize=10)
    ax.text(-1500, 1500, 4000, txt_xz.format(dif_xz), fontsize=10)
    ax.text(-1500, 1500, 3800, txt_cos.format(dif_rot), fontsize=10)
    '''
    # 1111テスト用　###################
    '''
    # cameraと人を結んだ線分
    camera1_xyz = [0, 0, 1000]
    cx = [camera1_xyz[0], x0]
    cy = [camera1_xyz[1], y0]
    cz = [camera1_xyz[2], z0]
    ax.plot(cx, cy, cz, color="red", label = "")
    
    vec_c = [cx[0] - cx[1], cy[0] - cy[1], cz[0] - cz[1]]
    ch_rot = (vec_head[0] * vec_c[0] + vec_head[1] * vec_c[1] + vec_head[2] * vec_c[2])\
              / (np.sqrt(pow(vec_head[0], 2) + pow(vec_head[1], 2) + pow(vec_head[2], 2)) * np.sqrt(pow(vec_c[0], 2) + pow(vec_c[1], 2) + pow(vec_c[2], 2)))
    ch_rot = round(ch_rot, 4)
    ce_rot = (vec_eye[0] * vec_c[0] + vec_eye[1] * vec_c[1] + vec_eye[2] * vec_c[2])\
              / (np.sqrt(pow(vec_eye[0], 2) + pow(vec_eye[1], 2) + pow(vec_eye[2], 2)) * np.sqrt(pow(vec_c[0], 2) + pow(vec_c[1], 2) + pow(vec_c[2], 2)))
    ce_rot = round(ce_rot, 4)
    ax.text(-1500, 1500, 3600, "head_cam:{}".format(ch_rot), fontsize=10)
    ax.text(-1500, 1500, 3400, "eye_cam:{}".format(ce_rot), fontsize=10)
    '''
    #####################################
    
    ######## 使用カメラの表示 #################
    ax.text(-1500, 1500, 4200, "cam_id:{}".format(res.iloc[frame]["use_cam"]), fontsize=10)
    ax.text(-1500, 1500, 4000, "timestamp:{}".format(res.iloc[frame]["timestamp"]), fontsize=10)
    
    ax.legend()

    ax.plot(x0, y0, z0, marker='o', color='red')
    # 補助線
    subx1 = np.linspace(0, x0, 2)
    suby1 = [y0, y0]
    subz1 = [z0, z0]
    subx2 = [x0, x0]
    suby2 = np.linspace(y0, 5000, 2)
    subz2 = [z0, z0]
    subx3 = [x0, x0]
    suby3 = [y0, y0]
    subz3 = np.linspace(0, z0, 2)
    ax.plot(subx1, suby1, subz1, color="yellow")
    ax.plot(subx2, suby2, subz2, color="yellow")
    ax.plot(subx3, suby3, subz3, color="yellow")
    
    ax.set_xlim((0, 5000))
    ax.set_ylim((0, 5000))
    ax.set_zlim((0, 3000))
    plt.title("x-y-z 3D", fontsize=15)
    ax.set(
    xlabel='X [mm]',
    ylabel='Y [mm]',
    zlabel='Z [mm]',
    )


ani = animation.FuncAnimation(fig, functools.partial(update, res=df), frames = range(num), interval=50)  # patialでdfを適用した形でのupdate関数を設定

# gifのファイル名   ############ani_0000_test00.gif#########
#now = dt.datetime.now()
#time = now.strftime('%Y%m%d-%H%M%S')
ani.save("./animation/test.gif", writer="pillow")

print("3Dplot finish")
#%matplotlib inline
#html = display.HTML(ani.to_html5_video())
#display.display(html)
#HTML(ani.to_jshtml())
#plt.show()

# カメラをみているときの誤差が11°
#!/bin/bash
############ tailのテスト ##################
#source visual/bin/activate

#touch ./test.csv
#tail -f ./test.csv >> ./tailtest.csv &

#python3 test.py
#################################

############

#ffmpeg -i rtsp://10.35.153.108:8554/unicast -vf framerate=fps=15  -f mp4 "./cam1.mp4"
#ffmpeg -i rtsp://10.35.153.156:8554/unicast -f mp4 "${dir}/cor.mp4" 
#ffmpeg -i rtsp://10.35.153.196:8554/unicast -f mp4 "${dir}/f612.mp4" 

#####################

########## ファイル更新監視 #############
INTERVAL=1 # 監視間隔(秒単位)
UPDATE_FILE="/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera1.csv" # 監視する更新ファイル

# 監視対象ファイルの時間を取得
last=`ls --full-time $UPDATE_FILE | awk '{print $6"-"$7}'`

while true; 
do
    ### 対象ファイルの現在のタイムスタンプを取得 ###
    current=`ls --full-time $UPDATE_FILE | awk '{print $6"-"$7}'`

    ### 更新時間の変更をチェック ###
    if [ $last != $current ] ; then
        echo "updated: $current"
        last=$current
        # openfaceのデータ書き込み時間を取得 複数台分監視できるか？ 時間データを格納(last)
        break

    ### INTERVAL時間分スリープする
    sleep $INTERVAL

    fi
done
############################3

############  #####################
#!/bin/bash
set -eux # エラー出力：未定義警告：標準出力
set +x

# 途中中断時のデータ削除
function trap_sigint() { 
  echo "########## 中断 ############"
  #echo $(pgrep -f Feature)
  kill $(pgrep -f Feature) # FeatureExtractionのプロセス終了 plill -f FeatureExtraction
  kill $(pgrep -f python3)
  kill $(pgrep -f tail)
  echo "kill_finish"
  exit 0 # プロセス終了
  #sudo rm -rf /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp
  #mkdir -p /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp
  #sudo rm /home/ubuntu/workdir/processe/looksh.log
}

trap trap_sigint sigint

# レイテンシの計測
date "+%m/%d/%Y %H:%M:%S:%3N" >> time.log # シェルスクリプト開始時間

# cpu稼働率の確認 毎回上書きされます
top -i -b -d1 > looksh.log &

# 初期化
sudo rm -rf /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp
mkdir -p /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp -m 777
sudo rm -rf /home/ubuntu/workdir/processe/forwardcsv
mkdir -p /home/ubuntu/workdir/processe/forwardcsv -m 777
sudo rm -rf /home/ubuntu/workdir/processe/rtspcsv
mkdir -p /home/ubuntu/workdir/processe/rtspcsv -m 777

###### 必要なかった
cd /home/ubuntu/workdir/processe
source visual/bin/activate
echo "venv activate"

# synchlonyze 処理
python3 data_synchlonyze.py & # 出力をどうするか
syn=$!
######

# openfaceの開始
cd /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin

################ 必要なfile作成 ################################### 追記
touch /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera1.csv # tailのためにからファイルを作成
touch /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera2.csv
touch /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera3.csv
./FeatureExtraction -f rtsp://10.35.153.108:8554/unicast -cam_width 1980 -cam_height 1080 -out_dir ./rtsp -of "camera1" &  # スクリプト終了時にkillが必要
./FeatureExtraction -f rtsp://10.35.153.156:8554/unicast -cam_width 1980 -cam_height 1080 -out_dir ./rtsp -of "camera2" &
./FeatureExtraction -f rtsp://10.35.153.196:8554/unicast -cam_width 1980 -cam_height 1080 -out_dir ./rtsp -of "camera3" &

# test two_cap
#./FeatureExtraction -f "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/videos/two_cap/camera1.mp4"  -out_dir "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/processed/two_cap" -of "camera1" &
#./FeatureExtraction -f "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/videos/two_cap/camera2.mp4"  -out_dir "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/processed/two_cap" -of "camera2" &

# headpose.pyの処理が終わるまでの出力されるcsvだけを取得(追記) 上とつなげた方がいいか？ ######カラムが二回目以降なくなっている
tail -f /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera1.csv >> /home/ubuntu/workdir/processe/rtspcsv/r_cam1.csv &
tail -f /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera2.csv >> /home/ubuntu/workdir/processe/rtspcsv/r_cam2.csv &
tail -f /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera3.csv >> /home/ubuntu/workdir/processe/rtspcsv/r_cam3.csv &

# camera.csvへの書き込み開始を検知したい
# すべての書き込みが開始されたときの各frame番号がわかれば基準とframe差をheadposeに渡して処理が可能
# 更新時間を記録して、最も早いものと遅いものの時間差から大体のframe差を割り出す({s_f - l_f}/0.05) -> synchlonyze.pyが処理する(実行後のずれが大半を占めていることが判明)

#tail -f /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera1.csv >> /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam1.csv &
#tail -f /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera2.csv >> /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam2.csv &
#tail -f /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/camera3.csv >> /home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/rtsp/r_cam3.csv &

########################################################################

sleep 3 # 少し待つ

# 仮想環境の有効化
cd
cd /home/ubuntu/workdir/processe
#source visual/bin/activate
#echo "venv activate"

# synchlonyze 処理
##python3 data_synchlonyze.py & # 出力をどうするか
##syn=$!
wait $syn
echo "syn fi"

# file作成
python3 forward_filegene.py

######ここから新しいrtsp.csvに対して以下をループ#####
# 結合処理を行うところで、データを取得した時にcsvを削除する <- headpose.pyで os.remove("camera1.csv")
#<< "COMENT"
while true
do
  date "+%Y/%m/%d %H:%M:%S:%3N" >> time.log # 1回目
# headpose.pyによる座標変換 -> headpose_result/test.csv
  python3 headpose.py    # -> trans.csv
  date "+%Y/%m/%d %H:%M:%S:%3N" >> time.log
  python3 trans_cat.py        # -> trans/trans_ave.csv
  date "+%Y/%m/%d %H:%M:%S:%3N" >> time.log 
# 座標変換scriptの出力を可視化scriptの入力へ parisに持っていくか？ 別で蓄えといたものを
  #python3 3Dplot.py
  date "+%Y/%m/%d %H:%M:%S:%3N" >> time.log # 1回目終了
  #sleep 300 # 少し待つ
done
#COMENT

<< "COMENT"
# バックグラウンドのwaitによる待機
while true
do
  date "+%Y/%m/%d %H:%M:%S:%3N" >> time.log # 1回目
# headpose.pyによる座標変換 -> headpose_result/test.csv
  python3 headpose.py &   # -> trans.csv
  headpose=$!
  wait $headpose # 逐次的な処理として動かしている -> この処理の場合、並行処理できるが、headposeで出力されたファイル名をそれぞれ特定してtans_catにかける必要がある
  python3 trans_cat.py &       # -> trans/trans_ave.csv 
  trans=$!
# 座標変換scriptの出力を可視化scriptの入力へ parisに持っていくか? 別で蓄えといたものを
  #python3 3Dplot.py &
  wait $trans
  date "+%Y/%m/%d %H:%M:%S:%3N" >> time.log # 1回目終了
done
COMENT
# 並列して座標変換scriptの出力から注目検知(bool)

# 注目検知の出力に基づくalexaの操作コマンド


echo "finish"
cat # 標準入力永遠まち
#!/bin/bash
set -eux # エラー出力：未定義警告：標準出力
set +x

function trap_sigint() { 
  echo "########## 中断 ############"
  kill $(pgrep -f ffmpeg) 
  echo "kill_finish"
  exit 0 # プロセス終了
}

trap trap_sigint sigint

name="0224"
serial="1-1"
file="/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/videos/${name}/${serial}cam1.mp4"

ffmpeg -i rtsp://10.35.153.108:8554/unicast -f mp4 ${file} &
ffmpeg -i rtsp://10.35.153.132:8554/unicast -f mp4 "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/videos/${name}/${serial}cam2.mp4" &
ffmpeg -i rtsp://10.35.153.196:8554/unicast -f mp4 "/home/ubuntu/workdir/openface/OpenFace-OpenFace_2.2.0/build/bin/videos/${name}/${serial}cam3.mp4" &
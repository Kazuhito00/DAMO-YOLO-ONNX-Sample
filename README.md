# DAMO-YOLO-ONNX-Sample
[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)のPythonでのONNX推論サンプルです。<br>

https://user-images.githubusercontent.com/37477845/206065452-46720772-c657-48c0-b9e4-7dca9ef80e86.mp4

# Requirement 
* opencv-python 4.5.5.64 or later
* onnxruntime-gpu 1.12.0 or later

# Demo
デモの実行方法は以下です。
```bash
python sample.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：damoyolo/model/damoyolo_tinynasL20_T_418.onnx
* --score_th<br>
クラス判別の閾値<br>
デフォルト：0.4
* --nms_th<br>
NMSの閾値<br>
デフォルト：0.85

# Reference
* [tinyvision/DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
YOLOX-ONNX-TFLite-Sample is under [Apache-2.0 License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[ブラジル・サンパウロ 車で渋滞した通り](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002161417_00000)を使用しています。

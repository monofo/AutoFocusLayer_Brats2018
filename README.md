# Autofocus layer
ほとんどのコードは[著者の実装](https://github.com/yaq007/Autofocus-Layer)
から持ってきてきます。  

前処理やデータ分割は[ここ](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)を参考にしています。  

著者実装ではBrats2015で実験を行なっていますが、Brat2018データセットを使用しました。
Brats2015とBrats201８ではラベルが異なるので注意してください。
詳しくは[ここ](https://www.med.upenn.edu/sbia/brats2018/data.html)

また、検証の出力のshapeに関してですが、入力サイズと異なるところに注意してください。
著者実装の`test_full.py`を参考にしています。

* 著者実装の`test.py`は少し実装が異なりますが、著者はこちらを使用して論文のスコアを出したと書いています。


## Usage
[To Do] pathを絶対pathで書いているのであとで直す  

### データの標準化、マスク抽出   
`python preprocess.py`

### データの分割
`python split.py`

### 訓練
`config.py`に設定を書き込んでいます。  
`python train.py`

### テスト
`python test.py`


## 結果
Todo: 図が雑なので直す
- nii形式をgifに変更する[library](https://github.com/miykael/gif_your_nifti)

flairによる教師データ  
![flair](./src/Brats18_2013_5_1_flair.gif)  
ground truth  
![graund truth](./src//Brats18_2013_5_1_seg.gif)  
prediction  
![予測結果](./src/pred.gif)  
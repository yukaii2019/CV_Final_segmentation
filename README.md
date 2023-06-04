# Model 相關檔案
* module.py: 放model資料夾
* trainer.py: 改裡面用的model
* inference.py: 改裡面用的model

# Preprocessing 相關檔案
* data.py: 裡面有標註的地方拿掉或放上去
* inference.py: 裡面有標註的地方拿掉或放上去

# Data augmentation 相關檔案
* data.py: 裡面有標註的地方拿掉或放上去


# 其他說明：
* data.py目前的設定是當初那個model用的，或許可以把random crop打開，或是gamma correction打開
* 目前用來做validation的是每個sequence的後面20%，跟之前寫的有點不一樣，想調比例可以從data.py裡面找
* 新增的一個conf.json是存training data張眼閉眼的答案的，要喬一下.sh裡面的路徑，這邊用在train的時候要跳過閉眼的結果
* 我這邊都沒有真的train滿 200 epoch 我也只train過一次放了一個晚上90幾個epoch，這個可能自己靠感覺

# train
```shell script=
bash run_train.sh
```

# inference
```shell script=
bash run_inference.sh
```


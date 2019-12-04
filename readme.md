1、生成TFRecord
     1.图像+xml(注意xml中的文件名与图像名保持一致)
     2.在utils.voc_classname_encoder.py中填入类别
     3.运行utils.test_voc_utils.py，其中对应的图像，xml和结果地址填好
2、图像增强参数设置
     1.填写 'image_augmentor_config'
3、模型参数设置
     1.'config'填写：注意'mode'在训练时为'train'
4、开始训练Train_centernet.py
5、测试时'config'中'mode'为'test', centernet.load_weight()填写训练生成的模型参数，
   其他类别、阈值等参数相应调整。
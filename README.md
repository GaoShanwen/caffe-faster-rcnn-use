# caffe-faster-rcnn-use
caffe faster-rcnn  

---------------source information-----------------------
object  :py-faster-rcnn
author  :rbgirshick
index   :https://github.com/rbgirshick/py-faster-rcnn.git
user    :Gao Wenjie
runtime :2018/10/1

----------------notice information-----------------------
1:please get the code use this word /
 -"git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git"
2:solve the problem which caused by changes of /
 -the vision of caffe's or cuda's by using the way this wagepage Issue says /
 -(eg.#481,#483)
3:you can ingnore "5 Download pre-computed Faster R-CNN detectors" /
 -while it always appear error
4:you should in another way to get the per-train model
 -for"Download pre-trained ImageNet models" appear error
5:the error while run tool/demo.py(raise RuntimeError('Invalid DISPLAY variable')
 -can be solved by adding this code "plt.switch_backend('agg')" /
 -to the blew of import plt
6:the picture youself add need follow these rules
 -filename is consist with six number and '.jpg'/'jpeg'
 -please delete place or other word.

-----------------using information-----------------------
1 train:
 (using the alternating optimization algorithm from our NIPS 2015 paper)
 cd $FRCN_ROOT
    ( eg. cd /home/priv-lab1/workspace/gwj/caffe-project/py-faster-rcnn )
    ./sudo experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
    ( eg. sudo experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG16 pascal_voc )
 (using the approximate joint training method)

 cd $FRCN_ROOT(not solve the problem of run error)
    ./sudo experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
    ( eg. sudo experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 pascal_voc )
2 demo:
 (you need cp the file named caffemodel to the directroy /
 "/data/faster_rcnn_models")
 1-the root of the data of demo.py need is "FCRN-ROOT/data/demo"
 2-you also need to add the name of pictures to demo.py
 3-run code is :"sudo ./tools/demo.py "

---------------output infromation---------------------------
Trained Fast R-CNN networks are saved under:
    output/<experiment directory>/<dataset name> /
    (eg. /py-faster-rcnn/output/faster_rcnn_alt_opt/voc_2007_trainval)
Test outputs are saved under:
    output/<experiment directory>/<dataset name>/<network snapshot name> /
    (eg. /py-faster-rcnn/output/faster_rcnn_alt_opt/voc_2007_test)

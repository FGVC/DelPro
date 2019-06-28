
# Download the data. If a command line parameter is provided, it is assumed
# to be the desired location of the files, and they will all be moved there
# when the downloads finish
URL="https://lilablobssc.blob.core.windows.net/cvwc2019/train/"
wget ${URL}atrw_detection_train.tar.gz
wget ${URL}atrw_anno_detection_train.tar.gz
wget ${URL}atrw_pose_train.tar.gz
wget ${URL}atrw_pose_val.tar.gz
wget ${URL}atrw_anno_pose_train.tar.gz
wget ${URL}atrw_reid_train.tar.gz
wget ${URL}atrw_anno_reid_train.tar.gz

if [ -z $1 ]
then
    mv atrw_*.tar.gz $1
fi

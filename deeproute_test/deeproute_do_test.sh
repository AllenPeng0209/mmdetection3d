#$1 config 
#$2 workdir_lastest_path
#$3 save folder
#$4 mode

#generate model output(txt files) for deeproute 
./tools/dist_test.sh $1 $2 8 --format-only --options save_folder=$3

#staring evaluation 
python ./mmdet3d/core/evaluation/deeproute_utils/eval.py 1 $3 $4

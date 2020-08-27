./tools/dist_test.sh configs/benchmark/hv_pointpillars_secfpn_4x8_80e_pcdet_deeproute.py work_dirs/pp_secfpn_deeproute/latest.pth 8 --format-only

python ./mmdet3d/core/evaluation/deeproute_utils/eval.py


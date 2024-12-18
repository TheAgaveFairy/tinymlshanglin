### train
labels='AFb,AFt,SVT,VFb,VFt,VPD'

echo "Minor Six Classes: AFb,AFt,SVT,VFb,VFt,VPD. NO SR or VT!"

python3 train_1D.py  --lr 0.01 --batchsz 256 --ext _mm --path_indices ./multi_minor_indices/ --loss ce --labels AFb,AFt,SVT,VFb,VFt,VPD

echo "TRAINING COMPLETE. TESTING NOW."

### test
python3 test_1D.py --net_name saved_models/IEGM_net_mm.pkl -b 1 --path_indices ./multi_minor_indices/ --labels AFb,AFt,SVT,VFb,VFt,VPD

### convert 
#python3 pt2onnx.py IEGM_net_Sf1bz256lr0.01

### train

python3 train_1D.py  --lr 0.01 --batchsz 256 --ext _newidx --path_indices ./multi_indices_new/ --loss ce

echo "TRAINING COMPLETE. TESTING NOW."

### test
python3 test_1D.py --net_name saved_models/IEGM_net_newidx.pkl -b 1 --path_indices ./multi_indices_new/ --labels AFb,AFt,SR,SVT,VFb,VFt,VPD,VT

### convert 
#python3 pt2onnx.py IEGM_net_Sf1bz256lr0.01

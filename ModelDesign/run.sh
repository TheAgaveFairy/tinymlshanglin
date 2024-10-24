### train
python3 train_1D.py  --lr 0.01 --loss f1 --batchsz 256 --ext _newidx

echo "TRAINING COMPLETE. TESTING NOW."

### test
python3 test_1D.py --net_name saved_models/IEGM_net_newidx.pkl -b 1

### convert 
#python3 pt2onnx.py IEGM_net_Sf1bz256lr0.01

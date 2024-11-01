### train

python3 train_1D.py  --lr 0.01 --batchsz 256 --ext _three --path_indices ./threeclass_indices/ --loss ce

echo "TRAINING COMPLETE. TESTING NOW."

### test
python3 test_1D.py --net_name saved_models/IEGM_net_three.pkl -b 1 --path_indices ./threeclass_indices/

### convert 
#python3 pt2onnx.py IEGM_net_Sf1bz256lr0.01

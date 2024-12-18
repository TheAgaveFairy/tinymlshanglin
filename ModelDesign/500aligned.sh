### train

echo "Shorted and aligned data test, SR VT only. 500 Long."

python3 train_1D.py  --lr 0.01 --loss f1 --batchsz 256 --ext _temp --path_indices ./temp_indices/ --path_data ./temp_data/ --epoch 50 --labels SR,VT

echo "TRAINING COMPLETE. TESTING NOW."

### test
python3 test_1D.py --net_name saved_models/IEGM_net_temp.pkl -b 1 --path_indices ./temp_indices/ --path_data ./temp_data/ --labels SR,VT

### convert 
#python3 pt2onnx.py IEGM_net_Sf1bz256lr0.01

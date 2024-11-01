import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from help_code_demo import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # latter two added by PD
from models.model_1 import *


def main():
    seed = 222
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ----------
    ### load trained network
    net = torch.load(args.net_name)
    net = net.float().to(device)
    net.eval()
    # ----------

    testset = IEGM_DataSET(root_dir = args.path_data,
                           indice_dir = args.path_indices,
                           mode = 'test',
                           size = args.size,
                           transform = transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    true_label, pred_label = [], []
    for data_test in testloader:
        IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
        IEGM_test = IEGM_test.float().to(device)
        labels_test = labels_test.to(device)
        outputs_test = net(IEGM_test)
        _, predicted_test = torch.max(outputs_test.data, 1)

        if args.batch_size > 1:
            true_label.extend(labels_test.detach().cpu().numpy())
            pred_label.extend(predicted_test.detach().cpu().numpy())
        else:
            true_label.append(labels_test.detach().cpu().numpy())
            pred_label.append(predicted_test.detach().cpu().numpy())   

    #target_names = ['AFb', 'AFt', 'SR', 'SVT', 'VFb', 'VFt', 'VPD', 'VT']
    target_names = args.labels.split(',')
    C = confusion_matrix(true_label, pred_label)
    print(C)

    skl_acc = accuracy_score(np.array(true_label), np.array(pred_label), normalize=True)

    skl_report = classification_report(np.array(true_label), np.array(pred_label), labels=list(range(8)), target_names=target_names)

    print(skl_acc)
    print(skl_report)
    acc = np.diag(C).sum() / C.sum()
    true_pos = np.diag(C)
    false_pos = np.sum(C, axis=0) - true_pos
    false_neg = np.sum(C, axis=1) - true_pos
    precision = np.sum(true_pos / (true_pos + false_pos))
    recall = np.sum(true_pos / (true_pos + false_neg))
    f1_score = (2 * precision * recall) / (precision + recall)
    fb_score = (1+2**2) * (precision * recall) / ((2**2)*precision + recall)
           
    """
    acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
    precision = C[1][1] / (C[1][1] + C[0][1])
    sensitivity = C[1][1] / (C[1][1] + C[1][0])
    FP_rate = C[0][1] / (C[0][1] + C[0][0])
    PPV = C[1][1] / (C[1][1] + C[1][0])
    NPV = C[0][0] / (C[0][0] + C[0][1])
    
    

    F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)

    print("\nacc: {},\nprecision: {},\nsensitivity: {},\nFP_rate: {},\nPPV: {},\nNPV: {}".format(acc, precision, sensitivity, FP_rate, PPV, NPV))
    print(f"F1_score: {F1_score} \nFbeta_score: {F_beta_score}")
    """
    # # ----------------------------
    from torchsummary import summary
    summary(net, input_size=(1, args.size, 1))
    # # --------------------------------


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    #argparser.add_argument('--path_data', type=str, default='../../../Summer24/tinyml_contest_data_training/')
    argparser.add_argument('--path_data', type=str, default='../../tinyml_contest_data_training/')
    argparser.add_argument('--net_name', type=str, default='IEGM_net.pkl')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--batch_size', '-b', metavar='B', type=int, default=1, help='Batch size')
    argparser.add_argument('--labels', type=str, default='Healthy,Dying')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
    print("device is --------------", device)

    main()

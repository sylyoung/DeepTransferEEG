import os 
import time
import torch
import torch.nn as nn
import torch.quantization
from models.EEGNet import *
from models.losspredictor import *
from copy import deepcopy
# torch.backends.quantized.engine = 'fbgemm'
torch.set_num_threads(20)
torch.set_num_interop_threads(20)

import argparse
from utils.EA import *
from dropout import *
from augment import *
from quantization_utils import *
from utils.data_utils import *
from torch.quantization import QuantStub, DeQuantStub


class EEGNet_Block_quantized(nn.Module):
    def __init__(self, block1, block2):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.block1 = block1
        self.block2 = block2

    def forward(self, x):
        x = self.quant(x)  
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.dequant(x)  
        return x
    

def quantize_eegnet_feature(model, calibration_dataloader, device='cpu'):
    """
    Perform static quantization on the EEGNet_feature model.
    
    Args:
        model: Pre-trained EEGNet_feature model
        calibration_dataloader: Calibration dataloader
        device: Device
    
    Returns:
        quantized_model: Quantized model
    """
    
    quantized_model = EEGNet_Block_quantized(
        deepcopy(model.block1), 
        deepcopy(model.block2)
    )
    quantized_model.load_state_dict(model.state_dict(), strict=False)
    quantized_model.eval()
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quantized_model = torch.quantization.prepare(quantized_model)
    torch.quantization.fuse_modules(quantized_model, [
        ['block1.1', 'block1.2'],  # Conv2d + BatchNorm2d
        ['block1.3', 'block1.4'],  # Conv2d + BatchNorm2d  
        ['block2.2', 'block2.3']   # Conv2d + BatchNorm2d
    ], inplace=True)
    quantized_model = quantized_model.to(device)
    with torch.no_grad():
        for data, _ in calibration_dataloader:
            data = data.to(device)
            _ = quantized_model(data)
    
    quantized_model = torch.quantization.convert(quantized_model)
    
    return quantized_model



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
    data_name_list = ['Zhou2016']

    paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = \
    'MI', 4, 14, 2, 1251, 250, [119, 100, 100, 90]

    args = argparse.Namespace(trial_num=trial_num, time_sample_num=time_sample_num, 
                                sample_rate=sample_rate, N=N, chn=chn, 
                                class_num=class_num, paradigm=paradigm, data='Zhou2016')

    args.method = 'BFT'
    args.backbone = 'EEGNet'

    # whether to use EA
    args.align = True
    args.dropout_num = 2

    # cpu or cuda
    args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'

    # load data
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)
    data_subjects, labels_subjects = split_data_by_subject(X, y, args.trial_num)
    
    # EA
    for i in range(len(data_subjects)):
        data_subjects[i] = EA_offline(data_subjects[i], 1)

    all_acc_1 = []
    all_acc_2 = []
    all_acc_3 = []
    all_augment_t_1 = []
    all_augment_t_2 = []
    all_augment_t_3 = []
    all_forward_t_1 = []
    all_forward_t_2 = []
    all_forward_t_3 = []
    for SEED in [42, 42, 43, 44]:
        args.SEED = SEED
        fix_random_seed(SEED)

        idt = 0
        args.idt = 0
        src_data, src_label, tar_data, tar_label = get_test_train(data_subjects, labels_subjects, idt)

        eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
        X_train = src_data[:, :, :eeg_length]
        X_train = torch.tensor(X_train, dtype=torch.float32)
        labels_train = torch.tensor(src_label, dtype=torch.long)
        X_train = X_train.unsqueeze(1)
        data_train = torch.utils.data.TensorDataset(X_train, labels_train)
        loader_train = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True)

        model_target = EEGNet(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=eeg_length,
                        kernLength=int(args.sample_rate // 2),
                        F1=8,
                        D=2,
                        F2=16,
                        dropoutRate=0.25)
        PATH_EEGNET_MODEL = "/PATH/TO/SAVE/MODEL/"
        PATH_TO_LOSSPRE_MODEL_DROPOUT = "/PATH/TO/SAVE/MODEL/"  

        base_dir = PATH_EEGNET_MODEL + str(SEED) + '/EEGNet_pth/'
        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_200.pth'
        tar_model_dir_cc = tar_model_dir
        checkpoint = torch.load(tar_model_dir)
        model_target.load_state_dict(checkpoint)
        block_model = EEGNet_Block(model_target.block1, model_target.block2)
        classifier = EEGNet_Classifier(model_target.classifier_block)

        model_loss = EEGNetLossPredictor(F2=16, Samples=eeg_length) 
        base_dir = PATH_TO_LOSSPRE_MODEL_DROPOUT + str(SEED) + '/loss_model_dropout_new(batch=16)/'
        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_20.pth'
        checkpoint = torch.load(loss_model_dir)
        model_loss.load_state_dict(checkpoint)

        ################## quantization ##################
        quantized_feature_model = quantize_eegnet_feature(block_model, loader_train)
        # print(quantized_feature_model)
        # size_fp = model_bytes(block_model)
        # size_q = model_bytes(quantized_feature_model)
        # ratio = size_q / size_fp
        # print(f"Model-FP: {size_fp:.2f} B    Model-Q: {size_q:.2f} B    RATE(Q/FP) = {ratio:.3f}")
        # print_model_parameters(quantized_feature_model)
        ################## quantization ##################

        # EEGNet time
        acc_1, augment_t_1, forward_t_1 = test_q(quantized_feature_model, classifier, tar_data, tar_label, args)

        # BFT-A time
        acc_2, augment_t_2, forward_t_2 = test_augment_with_loss_q(model_loss, quantized_feature_model, classifier, tar_data, tar_label, args)

        # # BFT-D time
        acc_3, augment_t_3, forward_t_3 = test_dropout_with_loss_q(model_loss, quantized_feature_model, classifier, tar_data, tar_label, args)
        
        all_acc_1.append(acc_1)
        all_acc_2.append(acc_2)
        all_acc_3.append(acc_3)
        all_augment_t_1.append(augment_t_1)
        all_augment_t_2.append(augment_t_2)
        all_augment_t_3.append(augment_t_3)
        all_forward_t_1.append(forward_t_1)
        all_forward_t_2.append(forward_t_2)
        all_forward_t_3.append(forward_t_3)
        
    acc_mean_1 = np.mean(all_acc_1[1:])
    acc_mean_2 = np.mean(all_acc_2[1:])
    acc_mean_3 = np.mean(all_acc_3[1:])
    acc_std_1 = np.std(all_acc_1[1:])
    acc_std_2 = np.std(all_acc_2[1:])
    acc_std_3 = np.std(all_acc_3[1:])

    augment_mean_1 = np.mean(all_augment_t_1[1:])
    augment_mean_2 = np.mean(all_augment_t_2[1:])
    augment_mean_3 = np.mean(all_augment_t_3[1:])
    augment_std_1 = np.std(all_augment_t_1[1:])
    augment_std_2 = np.std(all_augment_t_2[1:])
    augment_std_3 = np.std(all_augment_t_3[1:])

    forward_mean_1 = np.mean(all_forward_t_1[1:])
    forward_mean_2 = np.mean(all_forward_t_2[1:])
    forward_mean_3 = np.mean(all_forward_t_3[1:])
    forward_std_1 = np.std(all_forward_t_1[1:])
    forward_std_2 = np.std(all_forward_t_2[1:])
    forward_std_3 = np.std(all_forward_t_3[1:])

    print("\n\n")
    print("EEGNet ACC: {:.2f} ± {:.2f}".format(acc_mean_1, acc_std_1))
    print("EEGNet Augment_t: {:.2f} ± {:.2f}".format(augment_mean_1, augment_std_1))
    print("EEGNet Forward_t: {:.2f} ± {:.2f}".format(forward_mean_1, forward_std_1))

    print("\n")
    print("BFT-A ACC: {:.2f} ± {:.2f}".format(acc_mean_2, acc_std_2))
    print("BFT-A Augment_t: {:.2f} ± {:.2f}".format(augment_mean_2, augment_std_2))
    print("BFT-A Forward_t: {:.2f} ± {:.2f}".format(forward_mean_2, forward_std_2))

    print("\n")
    print("BFT-D ACC: {:.2f} ± {:.2f}".format(acc_mean_3, acc_std_3))
    print("BFT-D Augment_t: {:.2f} ± {:.2f}".format(augment_mean_3, augment_std_3))
    print("BFT-D Forward_t: {:.2f} ± {:.2f}".format(forward_mean_3, forward_std_3))

        

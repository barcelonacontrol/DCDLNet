import os
import argparse
import torch
import torch.nn as nn

from networks.vnet_alb import VNet_alb
from networks.vnet_AMC import VNet_AMC
from networks.vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='BRATS', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='detail')
parser.add_argument('--testnum', type=int,  default=90, help='test')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./{}".format(FLAGS.dataset)
test_save_path = os.path.join(snapshot_path, "test/")

train_data_path = FLAGS.root_path + "/" + FLAGS.dataset


num_classes = 2




def test_calculate_metric(save_mode_path, root_path, save_test_path, args):
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
    #print(root_path + '/test.list')

    if args.dataset == 'LA' or  args.dataset == 'BRATS':
        with open(root_path + '/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
        patch_size = (112, 112, 80)
        stride_xy=18
        stride_z=4
    if args.dataset =='Pancreas':
        with open(root_path + '/flods/test0.list', 'r') as f:
            image_list = f.readlines()
        image_list = [root_path + "/" + item.replace('\n', '') for item in image_list]
        patch_size = (96, 96, 96)
        stride_xy = 16
        stride_z = 16
    image_list = image_list[:args.testnum]
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(args, net, image_list, num_classes=num_classes,
                               patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
                               save_result=True, test_save_path=save_test_path, metric_detail=FLAGS.detail)

    return avg_metric

def test_calculate_metric_memory(net,train_data_path, test_save_path, args):
    net.eval()
    if args.dataset == 'LA' or  args.dataset == 'BRATS':
        with open(train_data_path + '/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [train_data_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
        patch_size = (112, 112, 80)
        stride_xy = 18
        stride_z = 4
    if args.dataset == 'Pancreas':
        with open(train_data_path + '/flods/test0.list', 'r') as f:
            image_list = f.readlines()
        image_list = [train_data_path + "/" + item.replace('\n', '') for item in image_list]
        patch_size = (96, 96, 96)
        stride_xy = 16
        stride_z = 16
    image_list = image_list[:args.testnum]
    avg_metric = test_all_case(args, net, image_list, num_classes=num_classes,
                               patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
                               save_result=True, test_save_path=test_save_path, metric_detail=FLAGS.detail)
    net.train()
    return avg_metric

if __name__ == '__main__':
    save_mode_path = os.path.join(
        snapshot_path, 'best_model42.pth')
    metric = test_calculate_metric(save_mode_path, train_data_path , test_save_path, FLAGS)  # 6000
    print(metric)
    print(torch.backends.cudnn.version())

# if __name__ == '__main__':
#     snapshot_path = "../model/BRATS/UAMT_LUD2_SASS_mlwr_42labels_beta_0.3"
#     best_evl = [0, 0, 100000, 100000]
#     for i in range(3600, 13001, 200):
#         index = "vnet_iter_" + str(i) + ".pth"
#         save_mode_path = os.path.join(
#             snapshot_path, index)
#         metric = test_calculate_metric(save_mode_path, train_data_path , test_save_path, FLAGS)  # 6000
#         if metric[0] > best_evl[0]: best_evl[0] = metric[0]
#         if metric[1] > best_evl[1]: best_evl[1] = metric[1]
#         if metric[2] < best_evl[2]: best_evl[2] = metric[2]
#         if metric[3] < best_evl[3]: best_evl[3] = metric[3]
#         print("-------------------------------------")
#         print("--student--")
#         print(str(i))
#         print(metric)
#         index = "teacher_vnet_iter_" + str(i) + ".pth"
#         save_mode_path = os.path.join(
#             snapshot_path, index)
#         metric = test_calculate_metric(save_mode_path, train_data_path, test_save_path, FLAGS)  # 6000
#         if metric[0] > best_evl[0]: best_evl[0] = metric[0]
#         if metric[1] > best_evl[1]: best_evl[1] = metric[1]
#         if metric[2] < best_evl[2]: best_evl[2] = metric[2]
#         if metric[3] < best_evl[3]: best_evl[3] = metric[3]
#         print("--teacher--")
#         print(metric)
#         print("best: ")
#         print(best_evl)
#         print("-----------------------------")
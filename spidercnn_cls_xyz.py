# import tensorflow as tf
# from tensorflow.python.framework import ops
import torch
import numpy as np
import math
import sys
import os
import torch.nn.functional as F
import torch.nn as nn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/grouping'))
from torch.autograd import Variable 
# import tf_util
# from tf_grouping import query_ball_point, group_point, knn_point
# from tf_sampling import farthest_point_sample, gather_point

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx



class conv2d(nn.Module):
    def __init__(self,num_in_channels,num_output_channels,kernel_size,stride=[1, 1]):
        super(conv2d, self).__init__()
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        self.conv = nn.Conv2d(num_in_channels,num_output_channels,(kernel_h,kernel_w),stride=(stride_h, stride_w))
        self.bn = nn.BatchNorm2d(num_in_channels)
    def forward(self,inputs,bn=False,is_training=None,G=None):
    
        # kernel_shape = [kernel_h, kernel_w,
        #                 num_in_channels, num_output_channels]
        
        num_in_channels = inputs.shape[-1]
        outputs =self.conv(inputs.permute(0,3,1,2))
        if bn:
            outputs = self.bn(outputs)
        outputs = F.relu(outputs)
        return outputs
    
        # kernel = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(kernel_shape)))
        # kernel = _variable_with_weight_decay('weights',
        #                                     shape=kernel_shape,
        #                                     use_xavier=use_xavier,
        #                                     stddev=stddev,
        #                                     wd=weight_decay)

        # # tf.nn.conv2d(inputs, kernel,
        # #                         [1, stride_h, stride_w, 1],
        # #                         padding=padding)
        # biases = _variable_on_cpu('biases', [num_output_channels],
        #                         tf.constant_initializer(0.0))
        # outputs = tf.nn.bias_add(outputs, biases)

        # if bn:
        #     if is_multi_GPU:
        #         outputs = batch_norm_template_multiGPU(outputs, is_training,
                                                    
        #                                                 'bn', [0,1,2], bn_decay)
        #     else:
        #         outputs = batch_norm_template(outputs, is_training, 
        #                                     'bn', [0,1,2], bn_decay)
        # if gn:
        #     outputs = group_norm_for_conv(outputs, G=G, scope='gn')
        # if activation_fn is not None:
        #     outputs = activation_fn(outputs)


class SpiderConv(nn.Module):
    def __init__(self,in_conv,num_conv):
        super(SpiderConv, self).__init__()
        self.conv = conv2d( in_conv* 5,num_conv, kernel_size=[1,20],stride=[1,1])
    def forward(self,feat,
                idx,
                delta,
                taylor_channel,
                bn=False,
                is_training=None,
                bn_decay=None,
                gn=False,
                G=32,
                scope='taylor'):
        grouped_points = index_points(feat, idx)
        batch_size,num_point,K_knn,in_channels = grouped_points.shape
        shape = [1, 1, 1, taylor_channel]
        X = delta[:, :, :, 0]
        Y = delta[:, :, :, 1]
        Z = delta[:, :, :, 2]

        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
        Z = Z.unsqueeze(-1)

        w = torch.empty(shape)    
        w_x =  nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_y =  nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_z =  nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_xyz = nn.Parameter( torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)

        w_xy = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_yz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_xz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        bias = nn.Parameter(torch.nn.init.zeros_(w)).expand(batch_size, num_point, K_knn,-1)

        w_xx = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_yy = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_zz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
                            
        w_xxy = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_xyy = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_xxz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)

        w_xzz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_yyz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_yzz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)

        w_xxx = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_yyy = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)
        w_zzz = nn.Parameter(torch.nn.init.xavier_normal_(w)).expand(batch_size, num_point, K_knn,-1)

        g1 = w_x * X + w_y * Y + w_z * Z + w_xyz * X * Y * Z
        g2 = w_xy * X * Y + w_yz * Y * Z + w_xz * X * Z + bias
        g3 = w_xx * X * X + w_yy * Y * Y + w_zz * Z * Z
        g4 = w_xxy * X * X * Y + w_xyy * X * Y * Y + w_xxz * X * X * Z
        g5 = w_xzz * X * Z * Z + w_yyz * Y * Y * Z + w_yzz * Y * Z * Z
        g6 = w_xxx * X * X * X + w_yyy * Y * Y * Y + w_zzz * Z * Z * Z
        g_d = g1 + g2 + g3 + g4 + g5 + g6

        grouped_points = grouped_points.unsqueeze(-1)
        g_d = g_d.unsqueeze(3)
        g_d = g_d.expand(-1, -1, -1, in_channels, -1)
        grouped_points = grouped_points * g_d
        grouped_points = grouped_points.reshape(batch_size, num_point, K_knn, in_channels*taylor_channel)
        
        feat = self.conv(grouped_points,  bn=bn, is_training=is_training,G=G)
        
        return feat.squeeze(-1).transpose(2,1)


class SpiderCNN(nn.Module):
    def __init__(self,num_class):
        super(SpiderCNN, self).__init__()
        self.spiderconv1 = SpiderConv(3,32)
        self.spiderconv2 = SpiderConv(32,64)
        self.spiderconv3 = SpiderConv(64,128)
        self.spiderconv4 = SpiderConv(128,256)
        self.fc1 = nn.Linear(480, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, num_class)

    def forward(self,xyz, is_training, bn_decay=None, num_classes=40):
        # batch_size = xyz.get_shape()[0].value
        # num_point = xyz.get_shape()[1].value

        batch_size = xyz.shape[0]
        num_point  =xyz.shape[1]
        nsample = 20
        G = 16
        taylor_channel = 5
        radius = 1
        idx = query_ball_point(radius, nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx) 
        delta = grouped_xyz - xyz.unsqueeze(2).expand(-1,-1,nsample,-1)
        # with tf.variable_scope('delta') as sc:
        #     _, idx = knn_point(nsample, xyz, xyz)
            

        #     grouped_xyz = group_point(xyz, idx)   
        #     point_cloud_tile = tf.expand_dims(xyz, [2])
        #     point_cloud_tile = tf.tile(point_cloud_tile, [1, 1, nsample, 1])
        #     delta = grouped_xyz - point_cloud_tile


        feat_1 = self.spiderconv1(xyz, idx, delta, taylor_channel = taylor_channel, 
                                            gn=True, G=G)
        feat_2 = self.spiderconv2(feat_1, idx, delta,  taylor_channel = taylor_channel,
                                            gn=True, G=G)
        feat_3 = self.spiderconv3(feat_2, idx, delta, taylor_channel = taylor_channel, 
                                            gn=True, G=G)
        feat_4 = self.spiderconv4(feat_3, idx, delta,  taylor_channel = taylor_channel, 
                                            gn=True, G=G)
        

        # with tf.variable_scope('fanConv1') as sc:
        #     feat_1 = tf_util.spiderConv(xyz, idx, delta, 32, taylor_channel = taylor_channel, 
        #                                     gn=True, G=G, is_multi_GPU=True)

        # with tf.variable_scope('fanConv2') as sc:
        #     feat_2 = tf_util.spiderConv(feat_1, idx, delta, 64, taylor_channel = taylor_channel, 
        #                                     gn=True, G=G, is_multi_GPU=True)

        # with tf.variable_scope('fanConv3') as sc:
        #     feat_3 = tf_util.spiderConv(feat_2, idx, delta, 128, taylor_channel = taylor_channel, 
        #                                     gn=True, G=G, is_multi_GPU=True)

        # with tf.variable_scope('fanConv4') as sc:
        #     feat_4 = tf_util.spiderConv(feat_3, idx, delta, 256, taylor_channel = taylor_channel, 
        #                                     gn=True, G=G, is_multi_GPU=True)

        feat  = torch.cat([feat_1, feat_2, feat_3, feat_4],dim=2)
        # feat = tf.concat([feat_1, feat_2, feat_3, feat_4], 2)

        l3_fea =  torch.max(feat, 1)[0]
        x = l3_fea.view(batch_size, 480)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        #top-k pooling
        # net = tf_util.topk_pool(feat, k = 2, scope='topk_pool')
        # net = tf.reshape(net, [batch_size, -1])
        # net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
        #                             scope='fc1', bn_decay=bn_decay, is_multi_GPU=True)
        # net = tf_util.dropout(net, keep_prob=0.3, is_training=is_training,
        #                     scope='dp1')
        # net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
        #                             scope='fc2', bn_decay=bn_decay, is_multi_GPU=True)
        # net = tf_util.dropout(net, keep_prob=0.3, is_training=is_training,
        #                     scope='dp2')
        # net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        return x


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)

    return classify_loss


if __name__=='__main__':
    # with tf.Graph().as_default():
    #     inputs = tf.zeros((32,1024,3))
    #     outputs = get_model(inputs, tf.constant(True))
    #     print(outputs)
    inputs  = torch.zeros((32,1024,3))
    model = SpiderCNN(40)
    outputs = model(inputs,True)
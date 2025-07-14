import torch
import torch.nn as nn
from einops import rearrange
from model.module.trans import Transformer as Transformer_encoder
from model.module.trans_hypothesis import Transformer as Transformer_hypothesis
from typing import Tuple

from monty.collections import AttrDict

from torch_scae import cv_ops
from torch_scae.nn_ext import Conv2dStack, multiple_attention_pooling_2d
from torch_scae.nn_utils import measure_shape


#--------------------------SCAE-------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 out_channels,
                 kernel_sizes,
                 strides,
                 activation=nn.ReLU,
                 activate_final=True):
        super().__init__()
        self.network = Conv2dStack(in_channels=input_shape[0],
                                   out_channels=out_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   activation=activation,
                                   activate_final=activate_final)
        self.output_shape = measure_shape(self.network, input_shape=input_shape)

    def forward(self, image):
        return self.network(image)

#----------------------------if run 3dhp datasets the below parameters needs adjust---------------------------------------------
class CapsuleImageEncoder(nn.Module):
    def __init__(self,
                 input_shape=[3,64,64],
                 n_caps= 34,
                 n_poses= 6,
                 n_special_features: int = 344,
                 noise_scale: float = 4.,
                 similarity_transform: bool = False,
                ):

        super().__init__()
        self.input_shape = input_shape
        self.encoder = encoder
        self.n_caps = n_caps  # M
        self.n_poses = n_poses  # P
        self.n_special_features = n_special_features  # S
        self.noise_scale = noise_scale
        self.similarity_transform = similarity_transform
        self.encoder = CNNEncoder(input_shape= [1326,3,3], out_channels=[128,128,128,128], kernel_sizes=[3,3,3,3],strides=[2,2,1,1], activation=nn.ReLU, activate_final=True)

        self._build()

        self.output_shapes = AttrDict(
            pose=(n_caps, n_poses),
            presence=(n_caps,),
            feature=(n_caps, n_special_features),
        )

    def _build(self):
        self.img_embedding_bias = nn.Parameter(
            data=torch.zeros(self.encoder.output_shape, dtype=torch.float32),
            requires_grad=True
        )
        in_channels = self.encoder.output_shape[0]
        self.caps_dim_splits = [self.n_poses, 1, self.n_special_features]  # 1 for presence
        self.n_total_caps_dims = sum(self.caps_dim_splits)
        out_channels = self.n_caps * (self.n_total_caps_dims + 1)  # 1 for attention
        self.att_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)






#-------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## MHG
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        self.Transformer_encoder_1 = Transformer_encoder(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, args.frames, args.frames*2, length=2*args.n_joints, h=9)

        self.Caps = nn.Conv1d(args.value3, args.value4, kernel_size=1)#

        ## Embedding
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2*args.out_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(args.layers, args.channel, args.d_hid, length=args.frames)
        
        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape
        #print('x.shape',x.shape)
        x = x.reshape(B,-1,3,3)
        #--------------------------------------------------
        batch_size = image.shape[0]  # B

        img_embedding = self.encoder(image)  # (B, D, G, G)

        h = img_embedding + self.img_embedding_bias.unsqueeze(0)  # (B, D, G, G)
        h = self.att_conv(h)  # (B, M * (P + 1 + S + 1), G, G)
        h = multiple_attention_pooling_2d(h, self.n_caps)  # (B, M * (P + 1 + S), 1, 1)
        h = h.view(batch_size, self.n_caps, self.n_total_caps_dims)  # (B, M, (P + 1 + S))
        del img_embedding
        x = h

        #--------------------------------------------------

        #print('x1.shape',x.shape)
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()
        print('x.shape', x.shape)

        ## 
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))

        #-----------------Object Capsule Encoder-------------------------

        #-------------------------END------------------------------------
        
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        

        ## SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3) 
        x = self.Caps(x)

        ## Regression
        x = x.permute(0, 2, 1).contiguous() 
        x = self.regression(x) 
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x







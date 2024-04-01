#使用torch实现的3D渲染器
#实现了一个简单的3D物体类，可以进行平移、旋转、缩放等操作
#使用时需要继承这个类，实现build方法，返回一个顶点坐标矩阵
#矩阵乘法使用torch实现，可以使用GPU加速
#使用时需要将矩阵转换为torch.tensor

import pygame as pg
import random
import torch
from torch import nn

COLOR_LIST = [0xff6464, 0x95e1d3, 0xeaffd0, 0xfce38a, 0xf38181, 0xff2e63, 0xfcbad3,
              0xaa96da, 0x30e3ca, 0xe0f9b5, 0xfff5a5, 0xffaa64, 0xff8264, 0xa7ff83]


def get_RGB(color):
    return (color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff

class OBJECT():

    def __init__(self):
        # self.faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]])
        self.color = pg.Color(get_RGB(random.choice(COLOR_LIST)))# 随机颜色
        self.position = torch.tensor([0.0, 0.0, 0.0]) # 位置向量
        self.device = torch.device('cpu')

    def build(self):
        pass

    #写一个函数的插件，对于每一个转换函数都添加。如果self.device是GPU,那么matrix都需要to(device)
    def GPU_plugin(func):
        def wrapper(self, *args, **kwargs):
            #将args转为tensor
            args = [torch.tensor(arg, dtype=torch.float32) if type(arg) != torch.Tensor else arg for arg in args]
            matrix = func(self, *args, **kwargs)
            if self.device.type == 'cuda':
                matrix = matrix.to(self.device)
            self.vertices = self.vertices @ matrix
        return wrapper

    def translate(self, pos):
        if type(pos) != torch.Tensor:
            pos = torch.tensor(pos, dtype=torch.float32)
        self.position += pos

    @GPU_plugin
    def scale(self, scale_to):
        sx, sy, sz = scale_to
        scale_matrix = torch.tensor([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]], dtype=torch.float32)
        return scale_matrix

    @GPU_plugin
    def rotate_x(self, angle):
        c, s = torch.cos(angle), torch.sin(angle)
        rotate_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, c, s, 0],
            [0, -s, c, 0],
            [0, 0, 0, 1]], dtype=torch.float32)
        # self.vertices = self.vertices @ rotate_matrix
        return rotate_matrix

    @GPU_plugin
    def rotate_y(self, angle):
        c, s = torch.cos(angle), torch.sin(angle)
        rotate_matrix = torch.tensor([
            [c, 0, -s, 0],
            [0, 1, 0, 0],
            [s, 0, c, 0],
            [0, 0, 0, 1]], dtype=torch.float32)
        # self.vertices = self.vertices @ rotate_matrix
        return rotate_matrix

    @GPU_plugin
    def rotate_z(self, angle):
        c, s = torch.cos(angle), torch.sin(angle)
        rotate_matrix = torch.tensor([
            [c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=torch.float32)
        # self.vertices = self.vertices @ rotate_matrix
        return rotate_matrix

    def get_matrix(self):
        tx, ty, tz = self.position
        translate_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [tx, ty, tz, 1]], dtype=torch.float32)
        if self.device.type == 'cuda':
            translate_matrix = translate_matrix.to(self.device)
        return self.vertices @ translate_matrix

    def to_device(self, device):
        self.device = device
        self.vertices = self.vertices.to(device)
        # self.faces = self.faces.to(device)
        self.position = self.position.to(device)











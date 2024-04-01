import torch
import math
import pygame as pg
import numpy as np
from numba import njit

from viewing.camera import Camera
from viewing.projection import Projection

@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))


class Render:

    def __init__(self, shape) -> None:
        # self.camera = camera
        pg.init()
        self.WIDTH, self.HEIGHT = shape
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.device = torch.device("cpu")

        self.move = True
        self.showvertices = True
        self.objs = []

        # 初始化一个相机，视角对着z轴向下
        self.camera = Camera(self, position=torch.tensor([400, 300, 1000]))
        self.projection = Projection(self)

        self.screen = pg.display.set_mode(shape)
        self.background_color = pg.Color(116, 117, 155)
        self.clock = pg.time.Clock()

    def addObj(self, obj):
        self.objs.append(obj)

    def rotateObj(self, obj):
        if self.move:
            obj.rotate_y(-(pg.time.get_ticks() % 0.02))

    def Orthogonal_projection(self, vertices):
        vertices = vertices @ self.camera.camera_matrix()
        vertices = vertices @ self.projection.get_projection_matrix()
        vertices /= vertices[:, -1].reshape(-1, 1)
        vertices[(vertices > 2) | (vertices < -2)] = 0
        vertices = vertices @ self.projection.get_to_screen_matrix()
        vertices = vertices[:, :2]
        return vertices


    def draw(self):
        # print("drawing")
        self.screen.fill(self.background_color)
        for obj in self.objs:
            # 假设我们是z轴朝下看的正交投影
            vectors = obj.get_matrix()
            vectors = self.Orthogonal_projection(vectors)
            line_color = obj.color

            #输出vectors是torch.float32还是其他类型？
            # print(vectors.dtype)
            # print(obj.faces.dtype)
            #先在这将obj.faces转换为ndarray
            if self.device.type == 'cuda':
                vectors = vectors.cpu().numpy()
            else:
                vectors = vectors.numpy()
            # obj.faces = obj.faces.numpy()
            # print(vectors[obj.faces[0]])
            for face in obj.faces:
                polygon = vectors[face]
                if any_func(polygon, self.H_WIDTH, self.H_HEIGHT):
                    continue
                pg.draw.polygon(self.screen, line_color, polygon, 0)
                # pg.draw.polygon(self.screen, pg.Color("BLACK"), vectors[face], 2) #放这里的话时间可能混淆

            for face in obj.faces:  # 画完面再画线，保证这个时间差。
                pg.draw.polygon(self.screen, pg.Color("BLACK"), vectors[face], 1)

            if self.showvertices:
                for vector in vectors:
                    pg.draw.circle(self.screen, pg.Color('white'), vector, 2)

            self.rotateObj(obj)

    def render(self):
        while True:
            self.camera.control()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return
            self.draw()
            pg.display.flip()
            self.clock.tick(60)

    def GPU(self):
        self.device = torch.device("cuda")
        self.projection.device = self.device
        self.camera.device = self.device
        for obj in self.objs:
            obj.to_device(self.device)


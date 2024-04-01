import torch
import math
import pygame as pg


class Camera():
    def __init__(self, render, position):
        self.render = render
        self.device = render.device
        self.position = torch.tensor([*position, 1.0], dtype=torch.float32)
        self.forward = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        self.up = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        self.right = torch.tensor([1, 0, 0, 1], dtype=torch.float32)
        self.h_fov = math.pi / 3
        self.v_fov = self.h_fov * (render.HEIGHT / render.WIDTH)
        self.near_plane = 0.1
        self.far_plane = 100
        # self.moving_speed = 0.3
        self.moving_speed = 3
        self.rotation_speed = 0.015

        self.anglePitch = 0
        self.angleYaw = 0
        self.angleRoll = 0

    def control(self):
        key = pg.key.get_pressed()
        if key[pg.K_a]:
            self.position -= self.right * self.moving_speed
            # print(self.position)
        if key[pg.K_d]:
            self.position += self.right * self.moving_speed
        if key[pg.K_w]:
            self.position += self.forward * self.moving_speed
        if key[pg.K_s]:
            self.position -= self.forward * self.moving_speed
        if key[pg.K_q]:
            self.position += self.up * self.moving_speed
        if key[pg.K_e]:
            self.position -= self.up * self.moving_speed

        if key[pg.K_LEFT]:
            self.camera_yaw(-self.rotation_speed)
        if key[pg.K_RIGHT]:
            self.camera_yaw(self.rotation_speed)
        if key[pg.K_UP]:
            self.camera_pitch(-self.rotation_speed)
        if key[pg.K_DOWN]:
            self.camera_pitch(self.rotation_speed)

    def camera_yaw(self, angle):
        self.angleYaw += angle

    def camera_pitch(self, angle):
        self.anglePitch += angle

    def axiiIdentity(self):
        self.forward = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        self.up = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        self.right = torch.tensor([1, 0, 0, 1], dtype=torch.float32)

    def camera_update_axii(self):
        rotate = self.rotate_x(self.anglePitch) @ self.rotate_y(self.angleYaw)  # this concatenation gives right visual
        self.axiiIdentity()
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate

    def rotate_x(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([[1, 0, 0, 0],
                             [0, c, s, 0],
                             [0, -s, c, 0],
                             [0, 0, 0, 1]], dtype=torch.float32)

    def rotate_y(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return torch.tensor([[c, 0, -s, 0],
                             [0, 1, 0, 0],
                             [s, 0, c, 0],
                             [0, 0, 0, 1]], dtype=torch.float32)

    def translate_matrix(self):
        tx, ty, tz, w = self.position
        return torch.tensor([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [-tx, -ty, -tz, 1]], dtype=torch.float32)

    def rotate_matrix(self):
        rx, ry, rz, w = self.right
        fx, fy, fz, w = self.forward
        ux, uy, uz, w = self.up
        return torch.tensor([
            [rx, ux, fx, 0],
            [ry, uy, fy, 0],
            [rz, uz, fz, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

    def camera_matrix(self):
        self.camera_update_axii()
        matrix = self.rotate_matrix() @ self.translate_matrix()
        if self.device.type == 'cuda':
            return matrix.to(self.device)
        return matrix
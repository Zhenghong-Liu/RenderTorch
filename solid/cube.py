# from object import OBJECT
import torch
from .object import OBJECT

class CUBE(OBJECT):

    def __init__(self, position=torch.tensor([0.0, 0.0, 0.0]), size=torch.tensor([1.0, 1.0, 1.0])):
        super().__init__()
        self.build()
        self.translate(position)
        self.scale(size)

    def build(self):
        # [x, y, z, w]
        # 长宽高默认为2， 中心在原点
        self.vertices = torch.tensor([
            [-1, -1, -1, 1], [1, -1, -1, 1],
            [1, 1, -1, 1], [-1, 1, -1, 1],
            [-1, -1, 1, 1], [1, -1, 1, 1],
            [1, 1, 1, 1], [-1, 1, 1, 1],
        ], dtype=torch.float32)
        # 无向边
        self.faces = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                               [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]], dtype=torch.int32)


if __name__ == '__main__':
    cube = CUBE(position=torch.tensor([200.0, 100.0, 20.0]), size=torch.tensor([1.0, 1.0, 2.0]))
    print(cube.vertices)
    print(cube.faces)
    print(cube.get_matrix())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    cube.to_device(device)
    # cube.rotate_x(torch.tensor(0.5))
    cube.rotate_x(0.5)
    print(cube.get_matrix())


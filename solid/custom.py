from .object import OBJECT
import torch

class CUSTOM(OBJECT):
    def __init__(self, position=torch.tensor([0.0, 0.0, 0.0]), size=torch.tensor([1.0, 1.0, 1.0]), file_path = None) -> None:
        assert file_path is not None, "file_path is None"

        super().__init__()
        self.build(file_path)
        self.translate(position)
        self.scale(size)

    def build(self, file_path):
        self.vertices = []
        self.faces = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                if line.startswith("v "):
                    self.vertices.append([float(i) for i in line[2:].split() ] +[1])
                elif line.startswith("f "):
                    face = [int(i.split('/')[0] ) -1 for i in line[2:].split()]
                    self.faces.append(face)

        self.vertices = torch.tensor(self.vertices, dtype=torch.float32)

if __name__ == '__main__':
    custom = CUSTOM(position=torch.tensor([200.0, 100.0, 20.0]), size=torch.tensor([1.0, 1.0, 1.0]), file_path = "../resources/Rocket.obj")
    print(custom.vertices)
    print(custom.faces)
    print(custom.get_matrix())
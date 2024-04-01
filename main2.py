from render import Render
from solid.custom import CUSTOM
import math
import torch

if __name__ == "__main__":
    render = Render([800, 600])
    # render.camera.position = torch.tensor([0.0, 0.0, 0.0, 1.0])


    roket = CUSTOM(position=[400.0, 350.0, 100.0], size=[30.0, 30.0, 30.0], file_path = "./resources/Rocket.obj")
    roket.rotate_x(math.pi)
    render.addObj(roket)


    render.GPU()
    render.render()

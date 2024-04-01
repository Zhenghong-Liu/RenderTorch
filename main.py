from render import Render
from solid.cube import CUBE
from solid.pyramid import PYRAMID
from solid.cylinder import CYLINDER

if __name__ == "__main__":
    render = Render([800, 600])


    cube2 = CUBE(position=[600, 450, 0], size=[50, 50, 50])
    cube2.rotate_x(20)
    cube2.rotate_z(30)

    cube3 = CUBE(position=[100, 350, 0], size=[30, 30, 30])
    cube3.rotate_x(-20)
    cube3.rotate_z(-30)

    pyramid = PYRAMID(position=[100, 100, 300], size=[50, 50, 50])
    pyramid.rotate_x(180)

    cylinder = CYLINDER(position=[400, 300, 0], size=[100, 100, 100], step = 20)
    cylinder.rotate_x(60)

    cylinder2 = CYLINDER(position=[650, 100, 0], size=[20, 20, 60], step = 3)
    # cylinder2.rotate_local(0, -60)

    render.addObj(cube2)
    render.addObj(cube3)
    render.addObj(pyramid)
    render.addObj(cylinder)
    render.addObj(cylinder2)

    render.GPU()
    render.render()

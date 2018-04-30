import numpy as np
import matplotlib.pyplot as plt

# Parameters
KP = 5.0  # attractive potential gain
ETA = 1000.0  # repulsive potential gain
AREA_WIDTH = 101.0  # potential area width [m]

show_animation = False      #if you would like to see a graph, change this to True


def calc_potential_field(gx, gy, ox, oy, reso, rr):
    #minx = min(ox) - AREA_WIDTH / 2.0
    #miny = min(oy) - AREA_WIDTH / 2.0
    #maxx = max(ox) + AREA_WIDTH / 2.0
    #maxy = max(oy) + AREA_WIDTH / 2.0
    minx = 0.0
    miny = 0.0
    maxx = AREA_WIDTH
    maxy = AREA_WIDTH
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            ug = calc_attractive_potential(x, y, gx, gy)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i in range(len(ox)):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    # calc potential field
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr)

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)
    if show_animation:
        draw_heatmap(pmap)
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    some = 0
    listx = [sx]
    listy = [sy]
    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        for i in range(len(motion)):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]):
                p = float("inf")  # outside area
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)
#        print(ix, iy)
        if len(rx) == len(ry):
            some = len(rx)
        if show_animation:
            plt.plot(ix, iy, ".r")              #ix iy = position red dot
            plt.pause(0.01)
#    j = j-1
    #print(j)
    print("Goal!!")

    return rx, ry, some


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def point(start, goal,laserPosition=[50.0, 80.0]):
    print("potential_field_planning start")

    sx = start[0]  # start x position [cm]
    sy = start[1]  # start y positon [cm]
    sz = start[2]       #start z position
    gx = goal[0]  # goal x position [cm]
    gy = goal[1]  # goal y position [cm]
    gz= goal[2]
    grid_size = 1.0  # potential grid size [cm]
    robot_radius = 1.0  # robot radius [cm]

    ox = [laserPosition[0]]  # obstacle x position list [cm]
    oy = [laserPosition[1]]  # obstacle y position list [cm]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    rx, ry ,some = potential_field_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

 #   print(type(rx), len(rx), type(ry), len(ry))
 #   print('point', some)
    section = (gz-sz)/(some-1)

    if section != 0:
        rz = np.arange(sz, gz+(section/2), section)
    else :
        rz = [sz]*len(rx)

    sent = []
    for a, b, c in zip(rx, ry, rz):
        position = [a, b, c]
        sent.append(position)
#        print(position)
#    print(len(sent), sent)


    if show_animation:
        plt.show()

    return sent


def main():
    start = [80.0, 30.0, 20.0]
    goal = [40.0, 100.0, 80.0]
    sent = point(start, goal)
    print(len(sent), sent)


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
print(__file__ + " Done!!")

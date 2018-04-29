import numpy as np
import matplotlib.pyplot as plt

# Parameters
KP = 5.0  # attractive potential gain
ETA = 1000.0  # repulsive potential gain
AREA_WIDTH = 101.0  # potential area width [m]

show_animation = True


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
    a = 0
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
        listx.append(ix)
        listy.append(iy)
#        print(ix, iy)
        if len(listx) == len(listy):
            some = len(listx)
        if show_animation:
            plt.plot(ix, iy, ".r")              #ix iy = position red dot
            plt.pause(0.01)
#    j = j-1
    #print(j)
    print("Goal!!")

    return rx, ry, some, listx, listy


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    sx = 80.0  # start x position [m]
    sy = 30.0  # start y positon [m]
    sz = 20.0
    gx = 40.0  # goal x position [m]
    gy = 100.0  # goal y position [m]
    gz= 80.0
    grid_size = 1.0  # potential grid size [m]
    robot_radius = 1.0  # robot radius [m]

    #suan = (gz-sz)/j
    #print(suan)

    ox = [50.0]  # obstacle x position list [m]
    oy = [80.0]  # obstacle y position list [m]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    rx, ry ,some, listx, listy  = potential_field_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_radius)
    print('point qua', some)
    section = (gz-sz)/(some-1)

#    print(section)
#    print(len(listx), listx)
#    print(len(listy), listy)
    listz = np.arange(sz, gz+(section/2), section)
#    print(len(listz), listz)

    position = []
    sent = []
    for a, b, c in zip(listx, listy, listz):
        position = [a, b, c]
        sent.append(position)
#        print(position)
#    print(sent)


    if show_animation:
        plt.show()

    return sent



if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
print(__file__ + " Done!!")
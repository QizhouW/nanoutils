import numpy as np
from skimage.draw import polygon
import os

class grid():
    def __init__(self, shape, xrange, yrange, zrange=None):
        self.dim = shape.__len__()
        self.nm_l = shape
        ranges = [xrange, yrange, zrange]
        self.r = []
        for i, (s, r) in enumerate(zip(shape, ranges)):
            self.r.append(np.linspace(*r, s))

    def to_grid(self, *a, axis=-1):
        idxs = []
        for coord in a:
            if axis == -1:
                xyz = [np.abs(x - x0).argmin() for x, x0 in zip(self.r, coord)]
                idxs.append(xyz)
            else:
                ix = np.abs(self.r[axis] - coord).argmin()
                idxs.append(ix)
        return np.array(idxs)

    def __getitem__(self, slices):
        return [c[i] for i, c in zip(slices, self.r)]


def get_rect_points(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    p2 = [x0, y1]
    p3 = [x1, y0]
    return np.array([p0, p2, p1, p3])


def add_rect(canvas, xrange=(-1.0, 1.0), yrange=(-1.0, 1.0), w=(1.0, 1.0), xy_c=np.array((0, 0)), material_idx = 1):
    coords = grid(canvas.shape, xrange, yrange)
    p0 = xy_c - np.array(w) / 2.
    p1 = xy_c + np.array(w) / 2.
    points = coords.to_grid(*get_rect_points(p0, p1))
    points = np.array(points)
    px = points[:, 0]
    py = points[:, 1]
    rr, cc = polygon(px, py, shape=canvas.shape)
    canvas[rr, cc] = material_idx
    return canvas


def add_pyramid(canvas, xrange=(-1.0, 1.0), yrange=(-1.0, 1.0), zrange=(-1.0, 1.0), sizes=(1.0, 1.0, 1.0),
                xyz_c=np.array((0, 0, 0)), inf_eps=1e-3, material_idx=1):
    coords = grid(canvas.shape, xrange, yrange, zrange)
    # sizes of fundament
    w = sizes[:2]
    # height
    h = sizes[-1]
    h_0, h_l = coords.to_grid(xyz_c[-1], xyz_c[-1] + h, axis=2)
    nSlices = h_l - h_0
    wx_z = np.linspace(w[0], inf_eps, nSlices)
    wy_z = np.linspace(w[1], inf_eps, nSlices)
    for i, z in enumerate(range(h_0, h_l)):
        canvas[:, :, z] = add_rect(canvas[:, :, z], xrange=xrange, yrange=yrange, w=(wx_z[i], wy_z[i]), xy_c=xyz_c[:2], material_idx=material_idx)
    return canvas





class CustomGeometry():
    def __init__(self, ec, outdir='geometry/'):
        self.dir = outdir
        self.dim = ec.shape.__len__()
        if self.dim == 2:
            self.geometry = Geometry2d(ec)
        else:
            self.geometry = Geometry3d(ec)


    def write_to_files(self, fname_template='material'):
        grids = self.geometry.grids
        for k in grids.keys():
            fname = os.path.join(self.dir,fname_template+'-'+k+'.bin')
            grids[k].tofile(fname)

class Geometry2d():
    def __init__(self, ec):
        self.grids = {}
        self.grids['c'] = ec.copy()
        self.grids['x'] = self.get_Ex(ec)
        self.grids['z'] = self.get_Ez(ec)

    @staticmethod
    def get_Ex(ec):
        return np.pad(ec, [(0, 0), (1, 0)], 'edge')

    @staticmethod
    def get_Ez(ec):
        return np.pad(ec, [(1, 0), (0, 0)], 'edge')

class Geometry3d():
    def __init__(self, ec):
        self.grids = {}
        self.grids['c'] = ec.copy()
        self.grids['x'] = self.get_Ex(ec)
        self.grids['z'] = self.get_Ez(ec)
        self.grids['y'] = self.get_Ey(ec)

    @staticmethod
    def get_Ex(ec):
        return np.pad(ec,[(0,0),(0,1),[1,0]],'edge')

    @staticmethod
    def get_Ey(ec):
        return np.pad(ec,[(0,1),(0,0),[1,0]],'edge')

    @staticmethod
    def get_Ez(ec):
        return np.pad(ec, [(0, 1), (1, 0), [0, 0]], 'edge')

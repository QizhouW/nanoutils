# author: Qizhou Wang
# datetime: 5/30/22 2:19 PM
# email: imjoewang@gmail.com
"""
This module generate the 3d array geometry for simulation
"""
import numpy as np
from skimage import io
from skimage.draw import polygon, rectangle, ellipse
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from skimage.draw import circle, ellipse, rectangle, rectangle_perimeter
from shapely.affinity import skew, affine_transform, translate, rotate, scale
from skimage.draw import random_shapes
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from shapely.errors import TopologicalError
import os
from utils import mkdir, Logger
from PIL import Image
from fdtd_geometry import CustomGeometry


def to_array(l, dx):
    n = np.floor(l / dx)
    if n == l / dx:
        return n.astype(np.int)
    else:
        return (n + 1).astype(np.int)


def to_shapely_coords(coords):
    x0, x1 = coords[0]
    y0, y1 = coords[1]
    return [(int((x0 + x1) / 2), int((y0 + y1) / 2)), x1 - x0 + 1, y1 - y0 + 1]


def convert_labels(labels):
    for i, label in enumerate(labels):
        coords = label[1]
        coords = to_shapely_coords(coords)
        labels[i] = [label[0], coords]
    return labels


def continuous_rectangle(mesh, min_size=10, orientation='random'):
    '''
    this function is used to generate
    1. rectangle shapes that is continuous in x or y axis
    2. an additional smaller random rectangle
    the min_size is a weak constraint, the actual etching width might range down to min_size-1
    but it doesn't matter
    '''
    connected = False
    coords = []
    # randomly choose continuous orientation
    if orientation == 'random':
        orientation = np.random.choice(['x', 'y'])

    if orientation == 'x':
        x0 = 0
        x1 = mesh.shape[0] - 1
        y0 = 0  # periodic condition
        y_length = np.random.randint(min_size, mesh.shape[1] + 1)
        y1 = y0 + y_length - 1
        coords.append(((x0, x1), (y0, y1)))
        if np.random.randint(0, 3) != 0:  # 2/3 chance add 2nd shape
            if mesh.shape[1] - y_length > 3 * min_size:
                x0_new = np.random.randint(0, mesh.shape[0] - 2 * min_size)
                x1_new = np.random.randint(x0_new + min_size, mesh.shape[0] - min_size)
                if np.random.randint(0, 2) == 0:  # make second rectangle connected to the first
                    y0_new = y1 + 1
                    connected = True
                else:
                    y0_new = np.random.randint(np.int(y1 + min_size / 2 + 1), mesh.shape[1] - 2 * min_size)
                y1_new = np.random.randint(y0_new + min_size, mesh.shape[1] - np.int(min_size / 2))
                coords.append(((x0_new, x1_new), (y0_new, y1_new)))

    if orientation == 'y':
        y0 = 0
        y1 = mesh.shape[1] - 1
        x0 = 0  # periodic condition
        x_length = np.random.randint(min_size, mesh.shape[0] + 1)
        x1 = x0 + x_length - 1
        coords.append(((x0, x1), (y0, y1)))
        # Add bbox for second shape, this can be arbitary
        if np.random.randint(0, 3) != 0:  # 2/3 chance add 2nd shape
            if mesh.shape[0] - x_length > 3 * min_size:
                y0_new = np.random.randint(0, mesh.shape[1] - 2 * min_size)
                y1_new = np.random.randint(y0_new + min_size, mesh.shape[1] - min_size)
                if np.random.randint(0, 2) == 0:  # make second rectangle connected to the first
                    x0_new = x1 + 1
                    connected = True
                else:
                    x0_new = np.random.randint(np.int(x1 + min_size / 2 + 1), mesh.shape[0] - 2 * min_size)
                x1_new = np.random.randint(x0_new + min_size, mesh.shape[0] - np.int(min_size / 2))
                coords.append(((x0_new, x1_new), (y0_new, y1_new)))
    return coords, connected


def generate_rect_mask(mesh, min_size=10, max_shapes=3, allow_overlap=False):
    maxsize = np.min(mesh.shape)
    min_shapes = np.random.randint(1, 3)
    if min_shapes == 1:
        min_size = np.int(1.6 * min_size)
    image, labels = random_shapes(mesh.shape, min_shapes=min_shapes, shape='rectangle', max_shapes=max_shapes,
                                  multichannel=False, intensity_range=((1, 1)), min_size=min_size, max_size=maxsize,
                                  allow_overlap=allow_overlap)
    idx = image == 255
    image[idx] = 0
    labels = convert_labels(labels)
    return image, labels


def embed_to_rectangle(rect, shape_type, min_size):
    xy_c, wx, wy = rect
    if shape_type == 'rectangle':
        return (xy_c, wx, wy)
    elif shape_type == 'square':
        w = min(wx, wy)
        shift = np.array(
            (np.random.random() * (w - wx) - (w - wx) / 2., np.random.random() * (w - wy) - (w - wy) / 2.)).astype(
            np.int)
        xy_c = np.array(xy_c) + shift
        return [xy_c, w]
    elif shape_type == 'ellipse':
        return [xy_c, wx / 2, wy / 2.]
    elif shape_type == 'circle':
        w = min(wx, wy)
        shift = np.array(
            (np.random.random() * (w - wx) - (w - wx) / 2., np.random.random() * (w - wy) - (w - wy) / 2.)).astype(
            np.int)
        xy_c = np.array(xy_c) + shift
        return [xy_c, w / 2.]
    elif shape_type == 'ring':
        xy_c, wx_e, wy_e = embed_to_rectangle(rect, shape_type='ellipse', min_size=min_size)
        try:
            wx_i = np.random.randint(low=np.int(min_size / 2), high=wx_e - np.int(min_size / 2))
            wy_i = np.random.randint(low=np.int(min_size / 2), high=wy_e - np.int(min_size / 2))
        except ValueError:
            wx_i = 0
            wy_i = 0
        return xy_c, [wx_e, wy_e], [wx_i, wy_i]
    elif shape_type == 'polygon':
        x0, y0 = (np.array(xy_c) - np.array([wx, wy]) / 2.).astype(np.int)
        x1, y1 = (np.array(xy_c) + np.array([wx, wy]) / 2.).astype(np.int)
        random_rr = np.random.randint(low=x0, high=x1, size=5)
        random_cc = np.random.randint(low=y0, high=y1, size=5)
        points = np.stack([random_rr, random_cc], axis=-1)
        return [points]
    elif shape_type == 'shoe':
        wx_e, wy_e = wx, wy
        try:
            wx_i = np.random.randint(low=min_size, high=max(0, wx_e - min_size))
            wy_i = np.random.randint(low=min_size, high=max(0, wy_e - min_size))
        except:
            wx_i = int(wx_e / 2)
            wy_i = int(wy_e / 2)
        return xy_c, [wx_e, wy_e], [wx_i, wy_i]


class Canvas():
    def __init__(self, canvas, shapes=None, material_idx=1):
        self.mask = canvas
        self.shapes = []
        if shapes:
            self.add_shapes(shapes, material_idx=material_idx)

    def add_shape(self, shape, material_idx=1):
        mask = shape.mask == 1
        self.mask[mask] = material_idx
        self.shapes.append(shape)

    def add_shapes(self, shapes, material_idx=1):
        for shape in shapes:
            self.add_shape(shape, material_idx)

    def show(self):
        plt.imshow(self.mask)
        plt.show()

    def save(self, fname, save_labels=True):
        self.mask.tofile(fname + '-mask.bin')
        if save_labels:
            labels = []
            for shape in self.shapes:
                label = {shape.type: shape.params}
                labels.append(label)
            np.save(fname + '-labels', np.array(labels))


class Shape():
    def __init__(self, img_shape, labels):
        self.type, self.params = labels[0], list(labels[1])
        self.size = img_shape[:2]
        self.mask = np.zeros(img_shape[:2], dtype=np.int32)
        self.add_shape()

    @classmethod
    def embedded_into_rectangle(cls, img_shape, min_size, rect_labels, shape_type, mode='debug'):
        # when debug mode save the rectangular mask perimeter
        if mode == 'debug':
            # from (wx_c), wx, wy to start,extent
            xy_c, wx, wy = rect_labels
            start = (xy_c - np.array([wx, wy]) / 2.).astype(np.int)
            rr, cc = rectangle_perimeter(start=start, extent=(wx, wy))
            cls.rectangle_mask_contour = np.stack([rr, cc], axis=-1)
        params = embed_to_rectangle(rect_labels, shape_type, min_size)
        labels = [shape_type, params]
        return cls(img_shape, labels)

    def add_shape(self):
        if self.type == 'rectangle':
            self.add_rectangle(*self.params)
        elif self.type == 'ellipse':
            self.add_ellipse(*self.params)
        elif self.type == 'circle':
            self.add_circle(*self.params)
        elif self.type == 'square':
            self.add_square(*self.params)
        elif self.type == 'ring':
            self.add_ring(*self.params)
        elif self.type == 'shoe':
            self.add_shoe(*self.params)
        else:
            raise AttributeError('Could not find such a shape in a dictionary!')

    def add_rectangle(self, xy_c, wx, wy, idx=1):
        start = (np.array(xy_c) - np.array([wx, wy]) / 2.).astype(np.int)
        rr, cc = rectangle(start=start, extent=(wx, wy), shape=self.mask.shape)
        self.mask[rr, cc] = idx

    def add_square(self, xy_c, w, idx=1):
        self.add_rectangle(xy_c, w, w, idx=idx)

    def add_ellipse(self, xy_c, wx, wy, idx=1):
        rr, cc = ellipse(*xy_c, wx, wy, shape=self.mask.shape)
        self.mask[rr, cc] = idx

    def add_ring(self, xy_c, wxy_e, wxy_i):
        """Interior domain is greater than exterior"""
        if np.any(wxy_e < wxy_i):
            raise ValueError("Interior domain is greater than exterior")
        self.add_ellipse(xy_c, *wxy_e)
        self.add_ellipse(xy_c, *wxy_i, idx=0)

    def add_circle(self, xy_c, w, idx=1):
        self.add_ellipse(xy_c, w, w, idx=idx)

    def add_shoe(self, xy_c, wxy_e, wxy_i):
        try:
            wy_e = wxy_e[1]
        except IndexError:
            wy_e = wxy_e[0]
        try:
            wy_i = wxy_i[1]
        except IndexError:
            wy_i = wxy_i[0]
        """Interior domain is greater than exterior"""
        if np.any(wxy_e < wxy_i):
            raise ValueError("Interior domain is greater than exterior")
        self.add_rectangle(xy_c, *wxy_e)
        self.add_rectangle(np.array(xy_c) + np.array([0, (wy_e - wy_i) / 2 + 1]).astype(np.int), *wxy_i, idx=0)


class Grid():
    def __init__(self, grid_shape, bottom=None):
        self.mask = np.zeros(grid_shape, dtype=np.int32)
        self.shape = grid_shape
        if bottom:
            self.z0 = bottom['h']
            self.mask[:, :, :self.z0] = bottom['material_idx']

    def add_canvas(self, canvas, h, z0=None, material_idx=1):
        mask = canvas.mask
        if not z0:
            z0 = self.z0
        for z in range(z0, z0 + h):
            mask = mask == 1
            self.mask[mask, z] = material_idx


def generate_random_shapes(mesh, types='all', min_size=10, max_shapes=3, allow_overlap=False, denominator=20):
    if types == 'all':
        shape_types = ['rectangle', 'square', 'circle', 'ellipse', 'ring', 'shoe']
    else:
        shape_types = types

    canvas = Canvas(np.zeros_like(mesh, dtype=np.int32))
    shapes = []
    if allow_overlap:
        overlap = np.random.choice([False, False, True])
    else:
        overlap = False
    image, labels = generate_rect_mask(mesh, min_size,
                                       max_shapes=max_shapes,
                                       allow_overlap=overlap)

    if np.random.randint(0, denominator) == 0:
        tmp_coords = labels[0][1]
        labels[0][0] = 'slab'
        if np.random.randint(0, 2) == 0:
            tmp_coords[0] = (int(mesh.shape[0] / 2), tmp_coords[0][1])
            tmp_coords[1] = mesh.shape[0]
        else:
            tmp_coords[0] = (tmp_coords[0][0], int(mesh.shape[1] / 2))
            tmp_coords[2] = mesh.shape[1]
    area = []
    for label in labels:
        tmpcanvas = Canvas(np.zeros_like(mesh, dtype=np.int32))
        if label[0] == 'slab':
            shape_type = 'rectangle'
        else:
            shape_type = np.random.choice(shape_types)
            label[0] = shape_type
        shape = Shape.embedded_into_rectangle(mesh.shape, min_size, label[1], shape_type=shape_type)
        shapes.append(shape)
        tmpcanvas.add_shape(shape)
        area.append(np.sum(tmpcanvas.mask))

    canvas.add_shapes(shapes)
    if np.sum(area) > np.sum(canvas.mask):
        overlap = True
    if np.max(area) == np.sum(canvas.mask):
        overlap = False  # means only the biggest shape is present
        labels = [labels[np.argmax(area)]]  # in this case only keep the largest
    if len(np.unique(canvas.mask)) == 1:  # means the shape is not generated properly:
        print('This shape might not be correct')
    # canvas.show()
    # return generate_random_shapes(mesh, types, min_size, max_shapes, allow_overlap, denominator)
    return canvas, labels, overlap


def gen_shape_from_label(labels, dim):
    canvas = Canvas(np.zeros(dim, dtype=np.int32))
    shapes = []
    for label in labels:
        if label[0] == 'slab':
            label[0] = 'rectangle'
        shape = Shape.embedded_into_rectangle(dim, 10, label[1], shape_type=label[0])
        shapes.append(shape)
    canvas.add_shapes(shapes)
    return canvas.mask

def from_x_to_labels(x, dim=[100, 100]):
    labels = []
    input_shape=dim
    nx = input_shape[0]
    ny = input_shape[1]
    x = x.reshape(-1, 4)
    for i, rect in enumerate(x):
        xy_c = tuple(np.rint([rect[0] * nx, rect[1] * ny]).astype(int))
        wx, wy = np.rint([rect[2] * nx, rect[3] * ny]).astype(int)
        label = {'rectangle': [xy_c, wx, wy]}
        labels.append(label)
    return labels



def write_png_geometry(imgpath, savedir, dx=0.005, grid_shape=[200, 200, 200], h=0.2, z1=0.4):
    assert imgpath.endswith('png'), 'Imgpath not correct'
    # img=io.imread(imgpath)
    image = Image.open(imgpath)
    img = np.array(image)
    img = img[:, :, 0]
    if img.shape != grid_shape[:2]:
        img = np.array(image.resize(grid_shape[:2]))
        img = img[:, :, 0]
    mask = np.zeros(grid_shape[:2])
    mask[np.where(img == 255)] = 1
    canvas = Canvas(np.zeros(grid_shape[:2], np.int32))
    canvas.mask = mask
    h = to_array(h, dx)
    z_glass = to_array(z1, dx)
    grid = Grid(grid_shape=grid_shape, bottom={'h': z_glass, 'material_idx': 1})
    grid.add_canvas(canvas, h, material_idx=2)
    ful = grid.mask[:]
    mkdir(savedir)
    geometry = CustomGeometry(ful, outdir=savedir)
    geometry.write_to_files(fname_template='material')
    return


########### definition ends ###############
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("error")
    upml_gap = 0
    si_id = 2
    grid_shape = (100, 100, 100)
    h = 0.2
    dx = 0.005
    shape_types = ['rectangle', 'ellipse']
    mesh = np.zeros(grid_shape[:2], dtype=np.int32)
    types = 'all'
    min_size = 10
    max_shapes = 3
    allow_overlap = True
    denominator = 6

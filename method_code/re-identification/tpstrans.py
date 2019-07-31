'''Image warping using thin plate splines. Optimized for speed using numba.

I consulted https://github.com/zpincus/celltool/blob/master/celltool/numerics/image_warp.py
while writing this code, and compared the results of my functions to his to make sure
I was getting something reasonable. I also consulted Donato and Belongie's paper on
Approximate Thin Plate Spline Mappings.

Connor Anderson
'''

import numpy as np
from scipy.ndimage import map_coordinates
import numba
from PIL import Image
import torch
from torchvision import transforms
import math


class RandomSpatialTransform(object):
    '''Apply random spatial augmentation to the given PIL Image. 
    
    The augmentation includes random rotation, random resized cropping, and
    random thin-plate-spline warping. All three operations are rolled into a
    single image resampling operation.

    Args:
        size (int or tuple): expected output size. If a tuple, then
            (height, width). If an int, then output will be square.
        scale (tuple): range of fractional area of the input to crop out 
            (min_area, max_area).
        ratio (tuple): range of aspect ratio (width / height) for the cropping
            region (min_ratio, max_ratio).
        rprob (float): value between 0 and 1, the probability of rotating the
            image.
        theta (int or float): max angle of rotation in degrees. Rotation will
            be sampled uniformly in [-theta, theta].
        tps_points (int): number of points defining the thin-plate-spline
            warp. Each point will be randomly sampled within the image, and
            then perturbed to define the warping.
        redius (tuple): min and max perturbation radii. The tps_points will 
            be perturbed away from their original locations each by a random
            distance in (min_radius, max_radius), and at a random angle.
        interpolation (int): image resampling interpolation. 1 is bilinear,
            3 is bicubic. Recommended to be 3.
    '''
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                 rprob=0.5, theta=25, tps_points=10, radius=(0.01, 0.05),
                 interpolation=3):
        # size is (h, w)
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.min_r = radius[0]
        self.max_r = radius[1]

        self.scale = scale
        self.ratio = ratio
        self.rprob = rprob
        self.theta = theta
        self.n = tps_points
        self.interpolation = interpolation

    @staticmethod
    def get_tps_points(img, n, min_r, max_r):
        w, h = img.size
        anchor_x, anchor_y = np.meshgrid([0, w // 2, w], [0, h // 2, h])
        anchor_x, anchor_y = anchor_x.ravel(), anchor_y.ravel()
        na = anchor_x.size

        src = np.empty((n + na, 2))
        src[:na, 0] = anchor_y
        src[na:, 0] = np.random.randint(0, h, (n,))
        src[:na, 1] = anchor_x
        src[na:, 1] = np.random.randint(0, w, (n,))
        dst = src.copy()
        inds = np.arange(na, n + na)

        min_side = min(w, h)
        max_r = min_side * max_r
        min_r = min_side * min_r
        rad = np.random.rand(inds.size) * (max_r - min_r) + min_r
        angle = np.random.rand(inds.size) * 2 * np.pi
        dst[inds, 0] += rad * np.sin(angle)
        dst[inds, 1] += rad * np.cos(angle)

        return src, dst

    @staticmethod
    def sample_rotation(p, theta):
        if np.random.rand() > p:
            return 0
        theta = theta * np.pi / 180
        angle = np.random.uniform(-theta, theta)
        return angle

    @staticmethod
    def get_bbox(img, scale, ratio):
        # Adapted from
        # https://pytorch.org/docs/stable/_modules/torchvision/
        #                    transforms/transforms.html#RandomResizedCrop
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = np.random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[0] and h < img.size[1]:
                i = np.random.randint(0, img.size[1] - h)
                j = np.random.randint(0, img.size[0] - w)
                return i, j, i + h, j + w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(w / min(ratio))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(h * max(ratio))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, i + h, j + w

    @staticmethod
    def rotate_points(points, center, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        rmat = np.array([[ct, -st], [st, ct]])
        points = (points - center) @ rmat + center
        return points

    @staticmethod
    def get_tps_weights(warp_to, warp_from):
        L = _make_L(warp_to)
        v = np.empty((L.shape[0], 2))
        v[:-3] = warp_from
        v[-3:] = 0
        weights = np.linalg.pinv(L) @ v
        return weights

    def __call__(self, img, keypoints=None):
        w, h = img.size
        src, dst = self.get_tps_points(img, self.n, self.min_r, self.max_r)
        theta = self.sample_rotation(self.rprob, self.theta)
        bbox = self.get_bbox(img, self.scale, self.ratio)

        if theta != 0:
            center = np.array([h / 2, w / 2])
            dst = self.rotate_points(dst, center, theta)

        # Calculate the tps coefficients
        weights = self.get_tps_weights(dst, src)
        # Get the warped coordinates
        wc = _warp_dim_regular(weights, dst, bbox, *self.size)
        wc = wc.transpose(2, 0, 1)
        # Warp the image
        img = np.asarray(img)
        warped_image = np.empty((self.size[0], self.size[1], 3), img.dtype)
        for i in range(3):
            warped_image[:, :, i] = map_coordinates(img[:, :, i], wc,
                                                    mode='reflect',
                                                    order=self.interpolation,
                                                    prefilter=True)
        np.clip(warped_image, img.min(), img.max(), out=warped_image)

        if keypoints is not None:
            if theta:
                keypoints = self.rotate_points(keypoints, center, theta)
            keypoints = keypoints - bbox[:2]
            keypoints = keypoints * np.divide(bbox[2:], self.size)
            keypoints = keypoints.reshape(1, *keypoints.shape)
            keypoints = _warp_dim(weights, dst, keypoints).squeeze()
            return warped_image, keypoints

        return warped_image


class RandomTPSTranform(object):
    def __init__(self, r=(3, 6), n=10):
        if type(r) is int:
            self.min_r = 0
            self.max_r = r
        else:
            self.min_r = r[0]
            self.max_r = r[1]
        self.n = n

    def __call__(self, img, kp=None):
        w, h = img.size
        anchor_x = [0, w // w, w]
        anchor_y = [0, h // 2, h]
        anchor_x, anchor_y = np.meshgrid(anchor_x, anchor_y)
        anchor_x = anchor_x.ravel()
        anchor_y = anchor_y.ravel()
        na = anchor_x.size
        if kp is None:
            src = np.empty((self.n + na, 2))
            src[:na, 0] = anchor_y
            src[na:, 0] = np.random.randint(0, h, (self.n,))
            src[:na, 1] = anchor_x
            src[na:, 1] = np.random.randint(0, w, (self.n,))
            dst = src.copy()
            inds = np.arange(na, self.n + na)
        else:
            kp = np.reshape(kp, (15, 3))
            vis = np.argwhere(kp[:, -1] == 2).ravel()
            src = np.empty((vis.size + na, 2))
            src[:na, 0] = anchor_y
            src[:na, 1] = anchor_x
            src[na:] = kp[vis, 1::-1]
            dst = src.copy()
            k = np.random.random_integers(1, vis.size)
            inds = na + np.random.choice(vis.size, size=k, replace=False)
        rad = np.random.rand(inds.size) * (self.max_r - self.min_r) + self.min_r
        angle = np.random.rand(inds.size) * 2 * np.pi
        dst[inds, 0] += rad * np.sin(angle)
        dst[inds, 1] += rad * np.cos(angle)

        img = np.asarray(img)
        img = warp_image(img, src, dst)
        img = Image.fromarray(img)
        return img


def warp_image(image, source, dest, order=3):
    '''Warp an image using thin plate spline interpolation.

    `source` and `dest` define two corresponding sets of (row, column) points.
    The points from `source` will be warped to the corresponding points in
    `dest`, and all other points will be warped according to the thin plate
    spline interpolation.

    Args:
        image (array): [M x N x 3] array containing the image data.
        source (array): [P x 2] array containing the source points in the
            form (row, column).
        dest (array): [P x 2] array containing the destination points in
            the form (row, column)
        order (int): order of pixel value interpolation when doing reverse
            mapping. 1=BILINEAR, 3=BICUBIC. (Default 3).
    '''
    h, w = image.shape[:2]
    bbox = np.int64([0, 0, h, w])
    warped_coords = _warp_regular(dest, source, bbox)
    wc = warped_coords.transpose(2, 0, 1)
    warped_image = np.empty_like(image)

    # It's much faster to do these three 2d warps than to configure a 3d warp.
    # I tested it.
    for i in range(3):
        warped_image[:, :, i] = map_coordinates(image[:, :, i], wc,
                                                order=order, prefilter=True)
    np.clip(warped_image, image.min(), image.max(), out=warped_image)

    return warped_image


def _warp_regular(source, dest, bbox, stride=1):
    '''
    Args:
        source (array): [p x 2] array of source coordinates, first column
            is row coordinates, second is column coordinates.
        dest (array): [p x 2] array of destination coordinates, first
            column is row coordinates, second is column coordinates.
        bbox (array): bounding box defining output region
            (row1, col1, row2, col2). If you want the output region to be
            the same as the input region, pass [0, 0, img_height, img_width].
    '''
    # Solve for the tps weights
    L = _make_L(source)
    v = np.empty((L.shape[0], 2))
    v[:-3] = dest
    v[-3:] = 0
    weights = np.linalg.pinv(L) @ v
    # Get the tps interpolated coordinates
    warped = _warp_dim_regular(weights, source, bbox, stride)

    return warped


@numba.njit(cache=True)
def _make_L(source):
    '''
    Args:
        source (array): [p x 2] array of source coordinates, first column
            is row coordinates, second is column coordinates
    '''
    n = source.shape[0]
    L = np.empty((n + 3, n + 3))
    for i in range(n):
        r1 = source[i, 0]
        c1 = source[i, 1]
        for j in range(i + 1, n):
            r2 = source[j, 0]
            c2 = source[j, 1]
            rad = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2)
            lograd = np.log(rad) if rad > 0 else 0
            rad =  0.5 * rad * lograd
            L[i, j] = rad
            L[j, i] = rad
    for i in range(n):
        L[i, i] = 0
        L[i, -3] = 1
        L[i, -2:] = source[i]
        L[-3, i] = 1
        L[-2:, i] = source[i]
    L[-3:, -3:] = 0

    return L


@numba.njit(cache=True)
def _warp_dim_regular(weights, source, bbox, hnum=-1, wnum=-1):
    '''
    For warping full image coordinates, set bbox = [0, 0, shape[0], shape[1]]
    and stride = 1

    Args:
        weights (array): [p+3 x 2] array of weights, first column is row
            weights, second is column weights
        source (array): [p x 2] array of source coordinates, first column
            is row coordinates, second is column coordinates
        bbox (array): bounding box (r1, c1, r2, c2) defining set of
            coordinates to warp. The coordinates will form a regular grid
            bounded by bbox, and spaced according to stride
        hnum (int): number of evenly spaced points to evaluate between
            `bbox[0]` and `bbox[2]`. If -1, use `bbox[2] - bbox[0]`
            (default: -1).
        wnum (int): number of evenly spaced points to evaluate between
            `bbox[1]` and `bbox[3]`. If -1, use `bbox[3] - bbox[1]`
            (default: -1).
    '''
    w, a = weights[:-3], weights[-3:]
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    hnum = height if hnum == -1 else hnum
    wnum = width if wnum == -1 else wnum
    stepr = height / hnum
    stepc = width / wnum
    warped = np.empty((hnum, wnum, 2))

    for i in range(0, hnum):
        r = bbox[0] + i * stepr  # row coordinate
        for j in range(0, wnum):
            c = bbox[1] + j * stepc  # column coordinate
            valr = a[0, 0] + a[1, 0] * r + a[2, 0] * c
            valc = a[0, 1] + a[1, 1] * r + a[2, 1] * c
            for k in range(source.shape[0]):
                pr = source[k, 0]
                pc = source[k, 1]
                # R = ||(r, c) - (pr, pc)||
                rad = (r - pr) * (r - pr) + (c - pc) * (c - pc)
                lograd = np.log(rad) if rad > 0 else 0.0
                rad = 0.5 * rad * lograd  # U(R) = R^2 log(R)
                valr += weights[k, 0] * rad
                valc += weights[k, 1] * rad
            warped[i, j, 0] = valr
            warped[i, j, 1] = valc
    return warped


def _warp(source, dest, rowc, colc):
    '''
    Args:
        source (array): [p x 2] array of source coordinates, first column
            is row coordinates, second is column coordinates
        dest (array): [p x 2] array of destination coordinates, first
            column is row coordinates, second is column coordinates
        rowc (array): [M x N] array of destination row coordinates - these
            will be warped back to the source space
        colc (array): [M x N] array of destination column coordinates - these
            will be warped back to the source space
    '''
    L = _make_L(source)
    v = np.empty((L.shape[0], 2))
    v[:-3] = dest
    v[-3:] = 0
    weights = np.linalg.pinv(L) @ v
    coords = np.stack((rowc, colc), 2)
    warped = _warp_dim(weights, source, coords)

    return warped


@numba.njit(cache=True)
def _warp_dim(weights, source, coords):
    '''
    Args:
        weights (array): [p+3 x 2] array of weights, first column is row
            weights, second is column weights
        source (array): [p x 2] array of source coordinates, first column
            is row coordinates, second is column coordinates
        coords (array): [M x N x 2] array of points to warp. These correspond
            to destination coordinates that will be warped back into the
            source space. coords[:, :, 0] are the row coordinates, and
            coords[:, :, 1] are the column coordinates.
    '''
    w, a = weights[:-3], weights[-3:]
    warped = np.empty(coords.shape)

    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            r = coords[i, j, 0]  # row coordinate
            c = coords[i, j, 1]  # column coordinate
            valr = a[0, 0] + a[1, 0] * r + a[2, 0] * c
            valc = a[0, 1] + a[1, 1] * r + a[2, 1] * c
            for k in range(source.shape[0]):
                pr = source[k, 0]
                pc = source[k, 1]
                # R = ||(r, c) - (pr, pc)||
                rad = (r - pr) * (r - pr) + (c - pc) * (c - pc)
                lograd = np.log(rad) if rad > 0 else 0.0
                rad = 0.5 * rad * lograd  # U(R) = R^2 log(R)
                valr += weights[k, 0] * rad
                valc += weights[k, 1] * rad
            warped[i, j, 0] = valr
            warped[i, j, 1] = valc
    return warped


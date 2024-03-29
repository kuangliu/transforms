# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
See "Data Augmentation" tutorial for an overview of the system:
https://detectron2.readthedocs.io/tutorials/augmentation.html
"""

import torch
import torch.nn.functional as F

import math
import numpy as np

from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)

from PIL import Image

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "ExtentTransform",
    "ResizeTransform",
    "RotationTransform",
    "ColorTransform",
    "PILColorTransform",
    "ColorDistortTransform",
    "LetterboxTransform",
    "PerspectiveTransform",
    "CutoutTransform",
]


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.
    The fill color is used to map pixels from the source rect that fall outside
    the source image.
    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
        return ret

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize(
                (self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))),
                            abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(
                self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array(
                [self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h -
                                               self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])


class ColorTransform(Transform):
    """
    Generic wrapper for any photometric transforms.
    These transformations should only affect the color space and
        not the coordinate space of the image (e.g. annotation
        coordinates such as bounding boxes should not be changed)
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in an ndarray and returns an ndarray.
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.op(img)

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class ColorDistortTransform(Transform):
    """Color distortion transform used in YOLOX.

    Reference: https://github.com/ifzhang/ByteTrack/blob/main/yolox/data/data_augment.py#L150
    """

    def __init__(self, alpha, beta, h_delta, s_alpha):
        super().__init__()
        self._set_attributes(locals())

    def _convert(self, img, alpha=1, beta=0):
        tmp = img.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        img[:] = tmp

    def apply_image(self, img):
        img = img.copy()
        if self.beta is not None:
            self._convert(img, beta=self.beta)

        if self.alpha is not None:
            self._convert(img, alpha=self.alpha)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.h_delta is not None:
            tmp = img[:, :, 0].astype(int) + self.h_delta
            tmp %= 180
            img[:, :, 0] = tmp

        if self.s_alpha is not None:
            self._convert(img[:, :, 1], alpha=self.s_alpha)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class LetterboxTransform(Transform):
    """Letterbox transform used in YOLOX.

    Reference: https://github.com/ifzhang/ByteTrack/blob/main/yolox/data/data_augment.py#L150
    """

    def __init__(self, target_size, scale, dx, dy):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        img_h, img_w, _ = img.shape
        sw = int(self.scale * img_w)
        sh = int(self.scale * img_h)
        resized = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        out_w, out_h = self.target_size
        pad_img = np.ones((out_h, out_w, 3)) * 114.0
        dx, dy = self.dx, self.dy
        pad_img[dy:dy+sh, dx:dx+sw, :] = resized
        return pad_img

    def apply_coords(self, coords):
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] *= self.scale
        new_coords[:, 1] *= self.scale
        new_coords[:, 0] += self.dx
        new_coords[:, 1] += self.dy
        return new_coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class PerspectiveTransform(Transform):
    """Perspective transform used in YOLOX.

    Reference: https://github.com/ifzhang/ByteTrack/blob/main/yolox/data/data_augment.py#L54
    """

    def __init__(self, M, perspective, border):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2
        if self.perspective:
            img = cv2.warpPerspective(
                img, self.M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(
                img, self.M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        return img

    def apply_coords(self, coords):
        n = len(coords)
        xy = np.ones((n, 3))
        xy[:, :2] = coords
        xy = xy @ self.M.T
        if self.perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, -1)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, -1)
        return xy

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class PILColorTransform(ColorTransform):
    """
    Generic wrapper for PIL Photometric image transforms,
        which affect the color space and not the coordinate
        space of the image
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in a PIL Image and returns a transformed
                PIL Image.
                For reference on possible operations see:
                - https://pillow.readthedocs.io/en/stable/
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__(op)

    def apply_image(self, img):
        img = Image.fromarray(img)
        return np.asarray(super().apply_image(img))


class CutoutTransform(Transform):
    def __init__(self, region):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        H, W = img.shape[:2]
        x, y, w, h = self.region
        x2 = min(W-1, x+w)
        y2 = min(H-1, y+h)
        img[y:y2, x:x2, :] = 0
        return img

    def apply_coords(self, coords):
        return coords


def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.
    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.
    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) +
                                   np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) +
                                   np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(
        scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)

# not necessary any more with latest fvcore
NoOpTransform.register_type("rotated_box", lambda t, x: x)

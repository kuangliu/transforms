import cv2
import random
import numpy as np


class Pad:
    '''Pad images & boxes to target size.

    Args:
      out_size: (tuple) target size.
      pad_val: (number) padding value.
    '''

    def __init__(self, out_size, pad_val=0):
        self.out_size = out_size
        self.pad_val = pad_val

    def pad_img(self, img):
        h, w = img.shape[:2]
        ow, oh = self.out_size
        assert(ow >= w and oh >= h)
        canvas = np.ones((oh, ow, 3), dtype=img.dtype) * self.pad_val
        x = (ow - w) // 2
        y = (oh - h) // 2
        canvas[y:y+h, x:x+w, :] = img
        self.x, self.y = x, y
        return canvas

    def pad_boxes(self, boxes):
        padded = boxes.copy()
        padded[:, 0::2] += self.x
        padded[:, 1::2] += self.y
        return padded

    def __call__(self, imgs, boxes=None):
        if isinstance(imgs, list):
            imgs = [self.pad_img(x) for x in imgs]
            if boxes is not None:
                boxes = [self.pad_boxes(x) for x in boxes]
        else:
            imgs = self.pad_img(imgs)
            if boxes is not None:
                boxes = self.pad_boxes(boxes)
        return imgs if boxes is None else (imgs, boxes)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(out_size={self.out_size}) '
        s += f'(pad_val={self.pad_val})'
        return s


if __name__ == '__main__':
    img = cv2.imread('./img/test.jpg')
    boxes = np.array([[0, 0, 100, 100]]).astype(np.float)
    print(img.shape)
    print(boxes)

    pad = Pad((5000, 5000))
    print(pad)
    img, boxes = pad(img, boxes)
    print(img.shape)
    print(boxes)

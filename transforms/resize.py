import cv2
import numpy as np

from utils.misc import is_list_of


class Resize:
    '''Resize images and bboxes.

    Args:
      img_scale: (tuple) target image scale of (w,h).
      keep_ratio: (bool) whether to keep the aspect ratio when resizing.
    '''

    def __init__(self, img_scale, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio

    def resize_img(self, img):
        tw, th = self.img_scale
        h, w = img.shape[:2]
        if self.keep_ratio:
            sw, sh = tw / w, th / h
            s = min(sw, sh)
            ow, oh = int(s * w), int(s * h)
            self.scale_factor = s
        else:
            ow, oh = tw, th
            self.scale_factor = 1
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        return img

    def resize_boxes(self, boxes):
        return boxes * self.scale_factor

    def __call__(self, imgs=None, boxes=None):
        if isinstance(imgs, list):
            imgs = [self.resize_img(x) for x in imgs]
            if boxes is not None:
                boxes = [self.resize_boxes(x) for x in boxes]
        else:
            imgs = self.resize_img(imgs)
            if boxes is not None:
                boxes = self.resize_boxes(boxes)
        return imgs if boxes is None else (imgs, boxes)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(img_scale={self.img_scale}), '
        s += f'(keep_ratio={self.keep_ratio})'
        return s


if __name__ == '__main__':
    img = cv2.imread('./img/test.jpg')
    boxes = np.array([[0, 0, 100, 100]]).astype(np.float)
    print(img.shape)
    print(boxes)

    resize = Resize((100, 100), keep_ratio=True)
    print(resize)
    img, boxes = resize(img, boxes)
    print(img.shape)
    print(boxes)

    imgs, boxes = resize([img], [boxes])
    print(imgs[0].shape)
    print(boxes)

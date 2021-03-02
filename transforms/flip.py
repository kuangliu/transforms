import cv2
import random
import numpy as np


class RandomFlip:
    '''Flip images and bboxes horizontally.

    Args:
      flip_ratio: (number) flip ratio.
    '''

    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def flip_img(self, img):
        self.w = img.shape[1]
        return cv2.flip(img, 1)

    def flip_boxes(self, boxes):
        flipped = boxes.copy()
        flipped[..., 0::4] = self.w - boxes[..., 2::4]
        flipped[..., 2::4] = self.w - boxes[..., 0::4]
        return flipped

    def __call__(self, imgs=None, boxes=None):
        if random.random() > self.flip_ratio:
            return imgs if boxes is None else (imgs, boxes)

        if isinstance(imgs, list):
            imgs = [self.flip_img(x) for x in imgs]
            if boxes is not None:
                boxes = [self.flip_boxes(x) for x in boxes]
        else:
            imgs = self.flip_img(imgs)
            if boxes is not None:
                boxes = self.flip_boxes(boxes)
        return imgs if boxes is None else (imgs, boxes)


if __name__ == '__main__':
    img = cv2.imread('./img/test.jpg')
    boxes = np.array([[0, 0, 100, 100]]).astype(np.float)
    print(img.shape)
    print(boxes)

    flip = RandomFlip(flip_ratio=0.5)
    img, boxes = flip(img, boxes)
    print(img.shape)
    print(boxes)

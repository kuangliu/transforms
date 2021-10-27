import cv2
import numpy as np
import transforms as T


def test_brightness(img):
    augs = T.AugmentationList([
        T.RandomBrightness(0.5, 1.5),
    ])
    print(augs)
    data = T.AugInput(img)
    transform = augs(data)
    img_t = data.image
    cv2.imwrite('z.png', img_t)


def test_flip(img, boxes):
    augs = T.AugmentationList([
        T.RandomFlip(0.5),
    ])
    print(augs)
    data = T.AugInput(img, boxes=boxes)
    transform = augs(data)
    img_t = data.image
    boxes_t = data.boxes
    cv2.imwrite('z.png', img_t)
    print(boxes_t)


def test_crop(img, boxes):
    augs = T.AugmentationList([
        T.RandomCrop('absolute', (640, 640)),
    ])
    print(augs)
    data = T.AugInput(img, boxes=boxes)
    transform = augs(data)
    img_t = data.image
    boxes_t = data.boxes
    cv2.imwrite('z.png', img_t)
    print(boxes_t)


def test_resize(img, boxes):
    augs = T.AugmentationList([
        T.ResizeShortestEdge((800, 1000))
    ])
    print(augs)
    data = T.AugInput(img, boxes=boxes)
    transform = augs(data)
    img_t = data.image
    boxes_t = data.boxes
    cv2.imwrite('z.png', img_t)
    print(boxes_t)


def test_extent(img, boxes):
    print(img.shape)
    h, w = img.shape[:2]
    augs = T.AugmentationList([
        T.RandomExtent(scale_range=(3/4., 3/4.), shift_range=(0.2, 0.2)),
        T.ResizeShortestEdge(800, 1400),
    ])
    print(augs)
    data = T.AugInput(img, boxes=boxes)
    transform = augs(data)
    img_t = data.image
    print(img_t.shape)
    boxes_t = data.boxes
    cv2.imwrite('z.png', img_t)
    print(boxes_t)


def test_yolox_distort(img):
    augs = T.AugmentationList([
        T.RandomColorDistort(),
    ])
    print(augs)
    data = T.AugInput(img)
    transform = augs(data)
    img_t = data.image
    cv2.imwrite('z.png', img_t)


def test_letterbox(img, boxes):
    print(img.shape)
    h, w = img.shape[:2]
    augs = T.AugmentationList([
        T.RandomLetterbox((500, 500))
    ])
    print(augs)
    data = T.AugInput(img, boxes=boxes)
    transform = augs(data)
    img_t = data.image
    print(img_t.shape)
    boxes_t = data.boxes
    cv2.imwrite('z.png', img_t)
    print(boxes_t)


if __name__ == '__main__':
    img = cv2.imread('./img/test.jpg')
    boxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
    # test_brightness(img)
    # test_flip(img, boxes)
    # test_crop(img, boxes)
    # test_resize(img, boxes)
    # test_extent(img, boxes)
    # test_yolox_distort(img)
    test_letterbox(img, boxes)

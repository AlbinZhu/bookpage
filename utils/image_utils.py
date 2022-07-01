from __future__ import division
import sys, enum

sys.path.append("./")
import cv2
import numpy as np
import random, math

from PIL import Image, ImageEnhance, ImageOps

from skimage.transform import resize
from anchors.iou import compute_iou, compute_bbox_coverage

MAP_INTERPOLATION_TO_ORDER = {
    "nearest": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3
}


def _uniform(val_range):
    """
  Uniformly sample from the given range.

  Args
      val_range: A pair of lower and upper bound.
  """
    return np.random.uniform(val_range[0], val_range[1])


def adjust_rotation(min=0, max=0, prob=0.5):
    """
  Construct a homogeneous 2D rotation matrix.

  Args
    min: a scalar for the minimum absolute angle in radians
    max: a scalar for the maximum absolute angle in radians
  Returns
      the rotation matrix as 3 by 3 numpy array
  """
    random_prob = np.random.uniform()

    if random_prob > prob:
        # angle: the angle in radians
        _check_range([min, max])
        angle = _uniform((min, max))
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    else:
        identity_matrix = np.ones(shape=(3, 3))
        return identity_matrix


def _check_range(val_range, min_val=None, max_val=None):
    """
  Check whether the range is a valid range.

  Args
      val_range: A pair of lower and upper bound.
      min_val: Minimal value for the lower bound.
      max_val: Maximal value for the upper bound.
  """
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound > upper bound')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


def _clip(image):
    """
  Clip and convert an image to np.uint8.

  Args
      image: Image to clip.
  """
    return np.clip(image, 0, 255).astype(np.uint8)


def adjust_brightness(image, prob=0.5, min=0.8, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        # factor=0 返回全黑色, factor=1 返回原图
        factor = np.random.uniform(min, max)
    image = Image.fromarray(np.uint8(image[..., ::-1]))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def adjust_hue(image, delta):
    """
  Adjust hue of an image.

  Args
      image: Image to adjust.
      delta: An interval between -1 and 1 for the amount added to the hue channel.
              The values are rotated if they exceed 180.
  """
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    """
  Adjust saturation of an image.

  Args
      image: Image to adjust.
      factor: An interval for the factor multiplying the saturation values of each pixel.
  """
    image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
    return image


def adjust_solarize(image, prob=0.5, threshold=128.):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    image = Image.fromarray(np.uint8(image[..., ::-1]))
    image = ImageOps.solarize(image, threshold=threshold)
    image = np.array(image)[..., ::-1]
    return image


def adjust_sharpness(image, prob=0.5, min=0, max=2, factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        # 0 模糊一点, 1 原图, 2 清晰一点
        factor = np.random.uniform(min, max)
    image = Image.fromarray(np.uint8(image[..., ::-1]))
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def adjust_color(image, prob=0.5, min=0., max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        # factor=0 返回黑白色, factor=1 返回原图;
        factor = np.random.uniform(min, max)
    image = Image.fromarray(np.uint8(image[..., ::-1]))
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def adjust_contrast(image, prob=0.5, min=0.2, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        # factor=0 返回灰色, factor=1 返回原图
        factor = np.random.uniform(min, max)
    image = Image.fromarray(np.uint8(image[..., ::-1]))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def center_crop_and_resize(image,
                           image_size,
                           crop_padding=32,
                           interpolation="bicubic"):
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]
    crop_size = int((image_size / (image_size + crop_padding)) * min(h, w))
    offset_height = ((h - crop_size) + 1) // 2
    offset_width = ((w - crop_size) + 1) // 2
    image_crop = image[offset_height:crop_size + offset_height,
                       offset_width:crop_size + offset_width]
    resized_image = resize(image_crop, (image_size, image_size),
                           order=MAP_INTERPOLATION_TO_ORDER[interpolation],
                           preserve_range=True)
    return resized_image


class configUtils(object):
    """
  struct holding config params for image Utils
  Args
  all bool type to detemine if use by True or false
  """

    def __init__(self, use_contrast, use_brightness, use_sharpness, use_color,
                 use_hue, use_saturation, use_solarize):
        self.use_contrast = use_contrast
        self.use_brightness = use_brightness
        self.use_sharpness = use_sharpness
        self.use_color = use_color
        self.use_hue = use_hue
        self.use_saturation = use_saturation
        self.use_solarize = use_solarize
        self.aug_factory = [
            self.use_contrast, self.use_brightness, self.use_sharpness,
            self.use_color, self.use_hue, self.use_saturation,
            self.use_solarize
        ]

    def random_aug(self):
        self.reset()
        index = np.random.randint(0, 7)
        self.aug_factory[index] = True

    def reset(self):
        self.use_contrast = False
        self.use_brightness = False
        self.use_sharpness = False
        self.use_color = False
        self.use_hue = False
        self.use_saturation = False
        self.use_solarize = False

class VisualEffect:
    """
  Struct holding parameters and applying image color transformation.
  Args
      solarize_threshold:
      color_factor: A factor for adjusting color.
      brightness_factor: A factor for adjusting brightness.
      sharpness_factor: A factor for adjusting sharpness.
      contrast_factor: A factor for adjusting contrast. Should be between 0 and 3.
      brightness_delta: Brightness offset between -1 and 1 added to the pixel values.
      hue_delta: Hue offset between -1 and 1 added to the hue channel.
      saturation_factor: A factor multiplying the saturation values of each pixel.
  """

    def __init__(self,
                 config,
                 hue_delta=0.5,
                 saturation_factor=0.5,
                 color_factor=None,
                 contrast_factor=None,
                 brightness_factor=None,
                 sharpness_factor=None,
                 color_prob=0.5,
                 contrast_prob=0.5,
                 brightness_prob=0.5,
                 sharpness_prob=0.5,
                 solarize_prob=0.1,
                 solarize_threshold=128.0):
        self.config = config
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor
        self.color_factor = color_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.sharpness_factor = sharpness_factor
        self.color_prob = color_prob
        self.contrast_prob = contrast_prob
        self.brightness_prob = brightness_prob
        self.sharpness_prob = sharpness_prob
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold

    def __call__(self, image):
        """
    Apply a visual effect on the image.

    Args
        image: Image to adjust
    """
        self.config.random_aug()
        if self.config.use_color:
            image = adjust_color(image,
                                 prob=self.color_prob,
                                 factor=self.color_factor)
        if self.config.use_brightness:
            image = adjust_brightness(image,
                                      prob=self.brightness_prob,
                                      factor=self.brightness_factor)
        if self.config.use_sharpness:
            image = adjust_sharpness(image,
                                     prob=self.sharpness_prob,
                                     factor=self.sharpness_factor)

        if self.config.use_solarize:
            image = adjust_solarize(image,
                                    prob=self.solarize_prob,
                                    threshold=self.solarize_threshold)
        if self.config.use_contrast:
            image = adjust_contrast(image,
                                    prob=self.contrast_prob,
                                    factor=self.contrast_factor)

        if self.config.use_hue:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = adjust_hue(image, self.hue_delta)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if self.config.use_saturation:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image = adjust_saturation(image, self.saturation_factor)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


def cutImagealongPattern(src_image, patterns):
    pass


def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image line", image)
    cv2.waitKey(0)


def circle_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               1,
                               20,
                               param1=60,
                               param2=40,
                               minRadius=0,
                               maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv2.imshow("image line", image)
    cv2.waitKey(0)


def catch_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    cv2.imshow('thresh', thresh)
    #调用cv2.findContours()寻找轮廓,返回修改后的图像,轮廓以及他们的层次
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('image', image)

    print('contours[0]:', contours[0])
    print('len(contours):', len(contours))
    print('hierarchy.shape:', hierarchy.shape)
    print('hierarchy:', hierarchy)

    #调用cv2.drawContours()在原图上绘制轮廓
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('contours', img)
    cv2.waitKey(0)


def histogram_equalization(image):
    equal = cv2.equalizeHist(image)
    return equal


def adapt_equalhist(image):
    #clipLimit参数表示对比度的大小。
    #tileGridSize参数表示每次处理块的大小
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    dst = clahe.apply(image.astype(np.uint8))
    return dst


def gaussion_filters(image):
    gaussions = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1.5)
    return gaussions


def mean_squres(image, window_size):
    left_top, right_bottom = window_size
    x1 = left_top[0]
    y1 = left_top[1]
    x2 = right_bottom[0]
    y2 = right_bottom[1]
    window_image = image[y1:y2, x1:x2].reshape(-1)
    mean = np.mean(window_image)
    return mean


def variance_squres(image, window_size, log=False):
    left_top, right_bottom = window_size
    x1 = left_top[0]
    y1 = left_top[1]
    x2 = right_bottom[0]
    y2 = right_bottom[1]
    mean = mean_squres(image, window_size)
    window_image = image[y1:y2, x1:x2].reshape(-1)
    sub_squre = np.square(window_image - mean)
    variance = np.mean(sub_squre)
    if log:
        print("window_size: ", window_size, ", y2 - y1:", y2 - y1, ", x2 -x1:",
              x2 - x1, "image shape: ", image.shape, ", crop shape: ",
              image[y1:y2, x1:x2].shape, ", window shape: ",
              window_image.shape, ", sub_squre.shape: ", sub_squre.shape,
              ", image values: ", image[y1:y2, x1:x2], ", mean value: ", mean,
              ", variance vallue: ", variance)
    return variance


def get_adaptive_k_values(image, window_size, var_global, coefficient=0.5):
    var_window = variance_squres(image, window_size, log=False)
    #print("=========================\nvar_window: ", var_window)
    k = coefficient * (var_window / var_global - 1)
    return k


def exp_transform(image, coefficient=1):
    trans_image = image.copy()
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            # tmp = image[i, j]/255
            tmp = math.ceil(math.exp(image[i, j] / coefficient)) - 1
            if tmp >= 0 and tmp <= 255:
                trans_image[i, j] = tmp
            elif tmp > 255:
                trans_image[i, j] = 255
            else:
                trans_image[i, j] = 0
    return trans_image


def compute_noise_variance(image, kernel_size, interval=50):
    h, w = image.shape[:2]
    k_h, k_w = kernel_size
    n_h, n_w = math.ceil(h / k_h), math.ceil(w / k_w)
    noise = []
    for i in range(n_h):
        for j in range(n_w):
            x1 = (j * k_w) if j * k_w <= (w - 1) else (w - 1)
            x2 = ((j + 1) * k_w) if (j + 1) * k_w <= (w - 1) else (w - 1)
            y1 = (i * k_h) if i * k_h <= (h - 1) else (h - 1)
            y2 = ((i + 1) * k_h) if (i + 1) * k_h <= (h - 1) else (h - 1)
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
            noise.append(variance_squres(image, ([x1, y1], [x2, y2])))
    noise = np.array(noise)
    noise_max = np.amax(noise)
    noise_min = np.amin(noise)
    internal = (noise_max - noise_min) / interval
    noise = np.sort(noise)
    count = []
    map_cout = []
    for i in range(interval):
        min_value = i * internal
        max_value = (
            i + 1) * internal if (i + 1) * internal < noise_max else noise_max
        i_count = 0
        i_cout = []
        for value in noise:
            if value >= min_value and value <= max_value:
                i_count += 1
                i_cout.append(value)
        count.append(i_count)
        map_cout.append(i_cout)
    max_index = np.argmax(count)
    map_mean = np.mean(map_cout[max_index])
    return map_mean


def log_transform(image, coefficient):
    log_image = image.copy().astype(np.float32)
    rows, cols = log_image.shape[:2]
    for i in range(rows):
        for j in range(cols):
            log_image[i, j] = coefficient * math.log(1 + image[i, j])
    return log_image


def rotate_image(image):
    rotate_degree = np.random.uniform(low=-45, high=45)
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image,
                           M=M,
                           dsize=(new_w, new_h),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(128, 128, 128))
    return image


def preprocess(image, image_size, annos=None):
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    before_h = int(pad_h / 2)
    after_h = pad_h - before_h
    before_w = int(pad_w / 2)
    after_w = pad_w - before_w
    image = np.pad(image, [(before_h, after_h), (before_w, after_w), (0, 0)],
                   mode='constant')

    if annos is not None:
        annos *= scale
        annos[..., 0:2] += [before_w / image_size, before_h / image_size]
        return image, annos
    return image, scale


class ProcessType(enum.Enum):
    ENCODE_ANNO = 0
    DECODE_ANNO = 1


class BaseProcessImage(object):

    def __init__(self) -> None:
        super(BaseProcessImage, self).__init__()

    def __call__(self, image, input_size, process_type, anno=None):
        if process_type == ProcessType.ENCODE_ANNO:
            return self._encode_image_(image, input_size, anno)
        elif process_type == ProcessType.DECODE_ANNO:
            return self._decode_image_(image, input_size, anno)
        else:
            raise ValueError("do not support process_type: ", process_type)

    def __letter_image__(self, image, image_size):
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale_w = image_size / image_height
            resized_height = image_size
            resized_width = int(image_width * scale_w)
            scale = (resized_width / image_size, 1.)
        else:
            scale_h = image_size / image_width
            resized_height = int(image_height * scale_h)
            resized_width = image_size
            scale = (1, resized_height / image_size)

        resized_image = cv2.resize(image, (resized_width, resized_height))
        resized_image = resized_image.astype(np.float32)
        resized_image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resized_image -= mean
        resized_image /= std
        pad_h = image_size - resized_height
        pad_w = image_size - resized_width
        before_h = int(pad_h / 2)
        after_h = pad_h - before_h
        before_w = int(pad_w / 2)
        after_w = pad_w - before_w
        resized_image = np.pad(resized_image, [(before_h, after_h),
                                               (before_w, after_w), (0, 0)],
                               mode='constant')
        pads = [(before_h, after_h), (before_w, after_w)]
        return resized_image, scale, pads

    def __resize_image__(self, image, image_size):
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale_w = image_size / image_height
            resized_height = image_size
            resized_width = int(image_width * scale_w)
            scale = (resized_width / image_size, 1.)
        else:
            scale_h = image_size / image_width
            resized_height = int(image_height * scale_h)
            resized_width = image_size
            scale = (1, resized_height / image_size)

        resized_image = cv2.resize(image, (resized_width, resized_height))
        resized_image = resized_image.astype(np.float32)
        pad_h = image_size - resized_height
        pad_w = image_size - resized_width
        before_h = int(pad_h / 2)
        after_h = pad_h - before_h
        before_w = int(pad_w / 2)
        after_w = pad_w - before_w
        resized_image = np.pad(resized_image, [(before_h, after_h),
                                               (before_w, after_w), (0, 0)],
                               mode='constant')
        return resized_image

    def __decode__(self, image_shape, image_size):
        image_height, image_width = image_shape
        if image_height > image_width:
            scale_w = image_size / image_height
            resized_height = image_size
            resized_width = int(image_width * scale_w)
            scale = (resized_width / image_size, 1.)
        else:
            scale_h = image_size / image_width
            resized_height = int(image_height * scale_h)
            resized_width = image_size
            scale = (1, resized_height / image_size)

        pad_h = image_size - resized_height
        pad_w = image_size - resized_width
        before_h = int(pad_h / 2)
        after_h = pad_h - before_h
        before_w = int(pad_w / 2)
        after_w = pad_w - before_w

        pads = [(before_h, after_h), (before_w, after_w)]
        return scale, pads

    def _encode_image_(self, image, input_size, anno=None):
        raise NotImplementedError("should be implementd by sub Class")

    def _decode_image_(self, image_shape, input_size, anno=None):
        raise NotImplementedError("should be implementd by sub Class")


class polyGonProcessImage(BaseProcessImage):

    def __init__(self) -> None:
        super(polyGonProcessImage, self).__init__()

    def _encode_image_(self, image, input_size, anno=None):
        resize_image, scale, pads = self.__letter_image__(image, input_size)
        [(before_h, _), (before_w, _)] = pads

        if anno is not None:
            w_offset = before_w / input_size
            h_offset = before_h / input_size
            scale_w, scale_h = scale
            anno[..., 0::2] = (anno[..., 0::2] * scale_w + w_offset)
            anno[..., 1::2] = (anno[..., 1::2] * scale_h + h_offset)
            return resize_image, anno

        return resize_image, scale

    def resize_image(self, image, input_size):
        return self.__resize_image__(image, input_size)

    def _decode_image_(self, image_shape, input_size, anno=None):

        _, pads = self.__decode__(image_shape, input_size)
        [(before_h, after_h), (before_w, after_w)] = pads
        if anno is not None:
            w_offset = before_w / input_size
            h_offset = before_h / input_size
            w_end = after_w / input_size
            h_end = after_h / input_size
            anno[...,
                 0::2] = (anno[..., 0::2] - w_offset) / (1 -
                                                         (w_offset + w_end))
            anno[...,
                 1::2] = (anno[..., 1::2] - h_offset) / (1 -
                                                         (h_offset + h_end))
            return anno
        return None


class OcrProcessImage(BaseProcessImage):

    def __init__(self) -> None:
        super(OcrProcessImage, self).__init__()

    def _encode_image_(self, image, input_size, anno=None):
        resize_image, scale, pads = self.__letter_image__(image, input_size)
        [(before_h, _), (before_w, _)] = pads

        if anno is not None:
            w_offset = before_w / input_size
            h_offset = before_h / input_size
            scale_w, scale_h = scale
            process_anno = anno.copy()
            process_anno[..., 0::2] = (anno[..., 0::2] * scale_w + w_offset)
            process_anno[..., 1::2] = (anno[..., 1::2] * scale_h + h_offset)
            return resize_image, process_anno

        return resize_image

    def _decode_image_(self, image_shape=None, input_size=None, anno=None):
        if anno is not None:
            return anno
        return None

class DetProcessImage(BaseProcessImage):

    def __init__(self) -> None:
        super(DetProcessImage, self).__init__()

    def _encode_image_(self, image, input_size, anno=None):
        image, scale, pads = self.__letter_image__(image, input_size)
        [(before_h, _), (before_w, _)] = pads

        if anno is not None:
            xmin = anno[..., 0] - anno[..., 2] / 2
            ymin = anno[..., 1] - anno[..., 3] / 2
            xmax = anno[..., 0] + anno[..., 2] / 2
            ymax = anno[..., 1] + anno[..., 3] / 2
            w_offset = before_w / input_size
            h_offset = before_h / input_size
            scale_w, scale_h = scale
            xmin = (xmin * scale_w + w_offset)
            xmax = (xmax * scale_w + w_offset)
            ymin = (ymin * scale_h + h_offset)
            ymax = (ymax * scale_h + h_offset)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            b_width = xmax - xmin
            b_height = ymax - ymin

            new_anno = np.stack([cx, cy, b_width, b_height], axis=1)
            return image, new_anno

        return image, scale

    def _decode_image_(self, image_shape, input_size, anno=None):
        _, pads = self.__decode__(image_shape, input_size)
        [(before_h, after_h), (before_w, after_w)] = pads
        if anno is not None:
            w_offset = before_w / input_size
            h_offset = before_h / input_size
            w_end = after_w / input_size
            h_end = after_h / input_size
            xmin = anno[..., 0]
            ymin = anno[..., 1]
            xmax = anno[..., 2]
            ymax = anno[..., 3]
            xmin = (xmin - w_offset) / (1 - (w_offset + w_end))
            xmax = (xmax - w_offset) / (1 - (w_offset + w_end))
            ymin = (ymin - h_offset) / (1 - (h_offset + h_end))
            ymax = (ymax - h_offset) / (1 - (h_offset + h_end))
            return np.stack([xmin, ymin, xmax, ymax], axis=1)
        return None


def crop_image(image, sampled_bbox):
    img_H, img_W = image.shape[:2]
    xmin = int((sampled_bbox[0] - sampled_bbox[2] / 2) * img_W)
    xmax = int((sampled_bbox[0] + sampled_bbox[2] / 2) * img_W)
    ymin = int((sampled_bbox[1] - sampled_bbox[3] / 2) * img_H)
    ymax = int((sampled_bbox[1] + sampled_bbox[3] / 2) * img_H)
    return image[ymin:ymax, xmin:xmax]


def transform_bboxes(bboxes, labels=None):

    xmin = bboxes[..., 0] - bboxes[..., 2] / 2
    ymin = bboxes[..., 1] - bboxes[..., 3] / 2
    xmax = bboxes[..., 0] + bboxes[..., 2] / 2
    ymax = bboxes[..., 1] + bboxes[..., 3] / 2
    if labels is not None:
        assert bboxes.shape[0] == labels.shape[0]
        trans_bboxes = np.stack([xmin, ymin, xmax, ymax, labels], axis=1)
    else:
        trans_bboxes = np.array([xmin, ymin, xmax, ymax])
    return trans_bboxes


def ClipBBox(bbox):
    bbox[0] = max(min(bbox[0], 1.), 0.)
    bbox[1] = max(min(bbox[1], 1.), 0.)
    bbox[2] = max(min(bbox[2], 1.), 0.)
    bbox[3] = max(min(bbox[3], 1.), 0.)
    return bbox


def project_bbox(src_bbox, bboxes, labels):
    trans_bboxes = transform_bboxes(bboxes, labels)
    trans_src_bbox = transform_bboxes(src_bbox)
    truths_bboxes = []
    src_wdith = trans_src_bbox[2] - trans_src_bbox[0]
    src_height = trans_src_bbox[3] - trans_src_bbox[1]
    for gt_box in trans_bboxes:
        if gt_box[0] >= trans_src_bbox[2] or gt_box[2] <= trans_src_bbox[
                0] or gt_box[1] >= trans_src_bbox[3] or gt_box[
                    3] <= trans_src_bbox[1]:
            continue
        if compute_bbox_coverage(gt_box, trans_src_bbox, "left_right") < 0.5:
            continue
        new_xmin = (gt_box[0] - trans_src_bbox[0]) / src_wdith
        new_ymin = (gt_box[1] - trans_src_bbox[1]) / src_height
        new_xmax = (gt_box[2] - trans_src_bbox[0]) / src_wdith
        new_ymax = (gt_box[3] - trans_src_bbox[1]) / src_height
        new_xmin, new_ymin, new_xmax, new_ymax = ClipBBox(
            [new_xmin, new_ymin, new_xmax, new_ymax])
        center_x = (new_xmin + new_xmax) / 2
        center_y = (new_ymin + new_ymax) / 2
        width = new_xmax - new_xmin
        height = new_ymax - new_ymin
        label = gt_box[4]
        truths_bboxes.append([center_x, center_y, width, height, label])
    return np.array(truths_bboxes)


def satisfy_sample_constraint(sampled_bbox, object_bboxes, sample_constraint):
    jaccard_values = compute_bbox_coverage(object_bboxes, sampled_bbox)

    if np.sum(jaccard_values >= sample_constraint) > 0:
        return True
    else:
        return False


class CropImageParams(object):

    def __init__(self, min_scale, max_scale, min_aspect_ratio,
                 max_aspect_ratio, max_sample, max_tries, min_jaccard_overlap):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_sample = max_sample
        self.max_tries = max_tries
        self.min_jaccard_overlap = min_jaccard_overlap


def crop_params(n_count=7):
    params = []
    for i in range(n_count):
        param = CropImageParams(i * 0.1 + 0.3, 1.0, i * 0.1 + 0.3, 2.0, 1, 50,
                                0.3 + i * 0.1)
        params.append(param)
    random.shuffle(params)
    return params


class MosaicImages(object):

    def __init__(self, max_tris, images_group, annos_group, target_size):
        self.max_tris = max_tris
        self.images = images_group
        self.annos_group = annos_group
        assert len(self.images) == 4
        assert len(self.annos_group) == 4
        self.target_size = target_size
        self.jaccard_overlap = 0.6

    def merge_images(self):
        target_H, target_W = self.target_size
        split_cx = int(random.uniform(target_W / 2, target_W * 3 / 2))
        split_cy = int(random.uniform(target_H / 2, target_H * 3 / 2))
        mix_up_H, mix_up_W = 2 * target_H, 2 * target_W
        split_boxes = []
        split_boxes.append((0, 0, split_cx, split_cy))
        split_boxes.append((split_cx, 0, mix_up_W - split_cx, split_cy))
        split_boxes.append((0, split_cy, split_cx, mix_up_H - split_cy))
        split_boxes.append(
            (split_cx, split_cy, mix_up_W - split_cx, mix_up_H - split_cy))

        target_image = np.zeros((2 * target_H, 2 * target_W, 3),
                                dtype=np.uint8)
        target_Height, target_Width = target_image.shape[:2]
        target_annos = {
            "bboxes": np.empty((0, 4), dtype=float),
            "labels": np.empty((0, ), dtype=int)
        }
        for _, (image, split_box, annos) in enumerate(
                zip(self.images, split_boxes, self.annos_group)):
            gt_bboxes = annos["bboxes"]
            gt_labels = annos["labels"]
            img_H, img_W = image.shape[:2]
            off_w, off_h, split_W, split_H = split_box
            if split_W >= img_W:
                #off_w = split_W - img_W
                split_W = img_W
            if split_H >= img_H:
                #off_h = split_H - img_H
                split_H = img_H
            found = False
            for index in range(self.max_tris):
                center_x = (random.randint(0, img_W - split_W) +
                            split_W / 2) / img_W
                center_y = (random.randint(0, img_H - split_H) +
                            split_H / 2) / img_H
                width = split_W / img_W
                height = split_H / img_H
                #print(gt_bboxes)
                sampled_bbox = np.array([center_x, center_y, width, height])
                if (satisfy_sample_constraint(sampled_bbox, gt_bboxes,
                                              self.jaccard_overlap)):
                    # crop_image
                    xmin = int((center_x - width / 2) * img_W)
                    ymin = int((center_y - height / 2) * img_H)
                    xmax = xmin + split_W
                    ymax = ymin + split_H
                    # print("split_H: ", split_H, ",  split_W: ", split_W,
                    #       ", x_width: ", xmax - xmin, ", y_height: ",
                    #       ymax - ymin)
                    target_image[off_h:off_h + split_H,
                                 off_w:off_w + split_W] = image[ymin:ymax,
                                                                xmin:xmax]
                    # crop_anno_bboxes
                    crop_truth_bboxes = project_bbox(sampled_bbox, gt_bboxes,
                                                     gt_labels)
                    #print("crop_truth_bboxes: ", crop_truth_bboxes)
                    # map to the corrspondent target_image_location
                    crop_bboxes = crop_truth_bboxes[..., 0:4] * [
                        split_W, split_H, split_W, split_H
                    ]
                    crop_labels = crop_truth_bboxes[..., 4]
                    crop_bboxes[..., 0:2] += [off_w, off_h]
                    crop_bboxes[..., 0:4] /= [
                        target_Width, target_Height, target_Width,
                        target_Height
                    ]
                    target_annos["bboxes"] = np.concatenate(
                        [target_annos["bboxes"], crop_bboxes])
                    target_annos["labels"] = np.concatenate(
                        [target_annos["labels"], crop_labels])
                    found = True
                    break
            if not found:
                print("not find valid mosaic sub images, so use src image")
                return image, annos
        target_annos["labels"] = target_annos["labels"].astype(np.int)
        return target_image, target_annos


class ImageDataAugumentTransform(object):

    def __init__(
        self,
        image_data,
        annotations,
        Resized=False,
        Cropped=False,
        image_folder=None,
        anno_folder=None,
    ):
        self.image_data = image_data
        self.annos = annotations
        self.Resized = Resized
        self.cropped = Cropped
        self.image_folder = image_folder
        self.anno_folder = anno_folder
        self.processed_images = self.image_data
        self.processed_annos = self.annos

    def __call__(self):
        batch_images = []
        batch_annos = []
        if self.cropped:
            for _, (image, annos) in enumerate(zip(self.image_data,
                                                   self.annos)):
                crop_image, crop_annos = self.crop_transform(image, annos)
                batch_images.append(crop_image)
                batch_annos.append(crop_annos)
            self.processed_images = batch_images
            self.processed_annos = batch_annos
        if self.Resized:
            for _, (image, annos) in enumerate(
                    zip(self.processed_images, self.processed_annos)):
                crop_image, crop_annos = self.crop_transform(image, annos)
                batch_images.append(crop_image)
                batch_annos.append(crop_annos)
        return batch_images, batch_annos

    def crop_transform(self, image, annos):
        gt_bboxes = annos['bboxes']
        labels = annos['labels']
        params = crop_params()
        sampled_bboxes = []
        for param in params:
            found_sampled_bbox = []
            for _ in range(param.max_tries):
                if len(found_sampled_bbox) >= param.max_sample:
                    break
                scale = random.uniform(param.min_scale, param.max_scale)
                aspect_ratio = random.uniform(param.min_aspect_ratio,
                                              param.max_aspect_ratio)
                aspect_ratio = max(aspect_ratio, pow(scale, 2.))
                aspect_ratio = min(aspect_ratio, 1 / pow(scale, 2.))
                bbox_width = scale * math.sqrt(aspect_ratio)
                bbox_height = scale / math.sqrt(aspect_ratio)
                w_off = random.uniform(0, 1 - bbox_width)
                h_off = random.uniform(0, 1 - bbox_height)
                sampled_bbox = np.array([
                    w_off + bbox_width / 2., h_off + bbox_height / 2.,
                    bbox_width, bbox_height
                ])
                if (satisfy_sample_constraint(sampled_bbox, gt_bboxes,
                                              param.min_jaccard_overlap)):
                    found_sampled_bbox.append(sampled_bbox)
            sampled_bboxes.append(found_sampled_bbox)
        sampled_bboxes = np.array(sampled_bboxes).reshape(-1, 4)
        if sampled_bboxes.shape[0] == 0:
            return image, annos
        else:
            select_index = np.random.randint(0, sampled_bboxes.shape[0])
            selected_bbox = sampled_bboxes[select_index]
            cropped_image = crop_image(image, selected_bbox)

            cropped_bboxes = project_bbox(selected_bbox, gt_bboxes, labels)
            if cropped_bboxes.shape[0] == 0:
                return image, annos
            cropped_annos = {}
            cropped_annos["bboxes"] = cropped_bboxes[..., 0:4]
            cropped_annos["labels"] = cropped_bboxes[..., 4].astype(np.int)
            return cropped_image, cropped_annos

    def resize_transform(self):
        pass


class VisualDataAugumentTransform(object):

    def __init__(self,
                 image_data,
                 visualEffect,
                 annos,
                 taskType,
                 SavedToFile=False):
        self.image_data = image_data
        self.visualEffect = visualEffect
        self.annos = annos
        self.taskType = taskType
        self.SavedToFile = SavedToFile
        super(VisualDataAugumentTransform, self).__init__()

    def __call__(self, label_to_name):
        if self.taskType == "DetectionBox":
            anno_label, anno_boxes = self.annos
            for label, boxes in zip(anno_label, anno_boxes):
                xmin, xmax, ymin, ymax = boxes
                cv2.rectangle(self.image_data, (xmin, ymin), (xmax, ymax),
                              (182, 230, 43), 1)
                cv2.putText(self.image_data, label_to_name(label),
                            (xmin, xmax), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                            (54, 230, 43), 1)
        elif self.taskType == "PolyGonPoints":
            anno_label, anno_points = self.annos
            for label, points in zip(anno_label, anno_points):
                cv2.polylines(self.image_data, [points], True, (182, 230, 43),
                              1)
                cv2.putText(self.image_data, label_to_name(label), points[1],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (54, 230, 43), 1)

        elif self.taskType == "OCR":
            str_label = ""
            for label in self.annos:
                str_label += label_to_name(label)
            img_h, img_w = self.image_data.shape[:2]
            stand_points = (int(img_h / 2), int(img_w / 2))
            cv2.putText(self.image_data, label_to_name(label), stand_points,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (54, 230, 43), 1)
        else:
            raise ValueError("we do not support this task type {}".format(
                self.taskType))
        if self.SavedToFile:
            image_path = ""
            cv2.imwrite(image_path, self.image_data)


import unittest


class TestStringMethods(unittest.TestCase):

    def test_iou(self):
        truth = np.array([0.703125, 0.66049383, 0.22878086, 0.41769547])
        pred = np.array([0.19463735, 0.31044239, 0.00810185, 0.01286008])
        iou = compute_iou(truth, pred)
        print("iou: ", iou)
        self.assertAlmostEqual(iou, 0.0, 4)

    def test_project_bbox(self):
        src_bbox = np.array([0.5, 0.55, 0.6, 0.5])
        bboxes = np.array([[0.6, 0.5, 0.2, 0.2]])
        labels = np.array([1])
        box = project_bbox(src_bbox, bboxes, labels)
        print(box)
        self.assertAlmostEqual(
            box.all(),
            np.array([[0.6667, 0.4, 0.3333, 0.4, 1.]]).all(), 4)

    def test_bboxCoverage(self):
        src_bbox = np.array([0.2, 0.3, 0.8, 0.8])
        bboxes = np.array([[0.5, 0.4, 0.6, 0.7], [0.6, 0.4, 0.9, 0.6]])
        iou_coverage = compute_bbox_coverage(bboxes, src_bbox, "left_right")
        print("iou_coverage: ", iou_coverage)
        self.assertEqual(iou_coverage[0], 1)
        self.assertAlmostEqual(iou_coverage[1], 0.6667, 4)

    def test_preprocessed__polyGon_image(self):
        from utils import util as utils
        image_path = "/scanImages/new_images/1642572257213.jpg"
        anno_path = "/home/baseuser/workspace/annoDir/page_label/1642572257213.txt"
        anno = utils.load_polyGonAnnotations(anno_path)
        src_image = cv2.imread(image_path)
        print("src_anno: ", anno['points'])
        Processed_Func = polyGonProcessImage()
        processed_image, annos = Processed_Func(src_image, 1024,
                                                ProcessType.ENCODE_ANNO,
                                                anno["points"])
        print("processed_anno: ", annos)
        saved_path = "/home/baseuser/workspace/verifyDataset/123/processed_1642572257213.jpg"

        for p_anno in annos:
            x_anno = np.round(p_anno.reshape(-1, 2) * [1024, 1024]).astype(int)

            cv2.polylines(processed_image, [x_anno], True, (182, 230, 43), 1)
            cv2.fillPoly(processed_image, [x_anno], (182, 230, 43))

        cv2.imwrite(saved_path, processed_image)
        saved_path = "/home/baseuser/workspace/verifyDataset/123/after_1642572257213.jpg"

        src_annos = Processed_Func(src_image.shape[:2], 1024,
                                   ProcessType.DECODE_ANNO, annos)
        print("again_annos: ", src_annos)
        img_H, img_W = src_image.shape[:2]
        after_image = src_image.copy()
        for p_anno in src_annos:
            x_anno = np.round(p_anno.reshape(-1, 2) *
                              [img_W, img_H]).astype(int)
            cv2.polylines(after_image, [x_anno], True, (182, 230, 43), 1)
            cv2.fillPoly(after_image, [x_anno], (182, 230, 43))

        cv2.imwrite(saved_path, after_image)

        saved_path = "/home/baseuser/workspace/verifyDataset/123/1642572257213.jpg"

        img_H, img_W = src_image.shape[:2]
        anno = utils.load_polyGonAnnotations(anno_path)
        for p_anno in anno["points"]:
            x_anno = np.round(p_anno.reshape(-1, 2) *
                              [img_W, img_H]).astype(int)
            cv2.polylines(src_image, [x_anno], True, (182, 230, 43), 1)
            cv2.fillPoly(src_image, [x_anno], (182, 230, 43))

        cv2.imwrite(saved_path, src_image)

    def test_preprocessed_det_image(self):
        from utils import util as utils
        image_path = "/scanImages/new_images/1642572257213.jpg"
        anno_path = "/home/baseuser/workspace/annoDir/review_label/1642572257213.txt"
        anno = utils.load_BoxAnnotations(anno_path)
        src_image = cv2.imread(image_path)
        print("src_anno: ", anno['bboxes'])
        Processed_Func = DetProcessImage()

        ###----------------------processed_image---------------------###
        processed_image, annos = Processed_Func(src_image, 1024,
                                                ProcessType.ENCODE_ANNO,
                                                anno["bboxes"])
        print("processed_anno: ", annos)
        saved_path = "/home/baseuser/workspace/verifyDataset/123/box_encode_1642572257213.jpg"
        decode_annos = []
        for p_anno in annos:
            x_anno = np.round(p_anno * [1024, 1024, 1024, 1024]).astype(int)

            xmin = int(x_anno[0] - x_anno[2] / 2)
            ymin = int(x_anno[1] - x_anno[3] / 2)
            xmax = int(x_anno[0] + x_anno[2] / 2)
            ymax = int(x_anno[1] + x_anno[3] / 2)
            cv2.rectangle(processed_image, (xmin, ymin), (xmax, ymax),
                          (182, 230, 43), 1)
            decode_annos.append(
                [xmin / 1024, ymin / 1024, xmax / 1024, ymax / 1024])
        cv2.imwrite(saved_path, processed_image)
        decode_annos = np.array(decode_annos)


        ###--------------------back reverse original image----------------###
        saved_path = "/home/baseuser/workspace/verifyDataset/123/box_decode_1642572257213.jpg"
        src_annos = Processed_Func(src_image.shape[:2], 1024,
                                   ProcessType.DECODE_ANNO, decode_annos)
        print("again_annos: ", src_annos)
        img_H, img_W = src_image.shape[:2]
        after_image = src_image.copy()
        for p_anno in src_annos:
            x_anno = np.round(p_anno * (img_W, img_H, img_W, img_H))
            xmin = int(x_anno[0])
            ymin = int(x_anno[1])
            xmax = int(x_anno[2])
            ymax = int(x_anno[3])
            cv2.rectangle(after_image, (xmin, ymin), (xmax, ymax),
                          (182, 230, 43), 1)
        cv2.imwrite(saved_path, after_image)
        saved_path = "/home/baseuser/workspace/verifyDataset/123/box_1642572257213.jpg"

        ###----------------------original image control---------------------###
        img_H, img_W = src_image.shape[:2]
        anno = utils.load_BoxAnnotations(anno_path)
        for p_anno in anno["bboxes"]:
            x_anno = np.round(p_anno *
                              [img_W, img_H, img_W, img_H]).astype(int)
            xmin = int(x_anno[0] - x_anno[2] / 2)
            ymin = int(x_anno[1] - x_anno[3] / 2)
            xmax = int(x_anno[0] + x_anno[2] / 2)
            ymax = int(x_anno[1] + x_anno[3] / 2)
            cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax),
                          (182, 230, 43), 1)

        cv2.imwrite(saved_path, src_image)


if __name__ == "__main__":
    unittest.main()

import cv2
import numpy as np


class Pen(object):

    def __init__(self, color, thickness, font, font_scale):
        self.color = color
        self.thickness = thickness
        self.font = font
        self.font_scale = font_scale

    def set_color(self, color):
        self.color = color

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_font_scale(self, font_scale):
        self.font_scale = font_scale

    def set_font(self, font):
        self.font = font


class Draw(object):

    def __init__(self, pen) -> None:
        if not isinstance(pen, Pen):
            raise TypeError("pen should be object of Pen")

        self.pen = pen

    def set_color(self, color):
        self.pen.set_color(color)

    def set_thickness(self, thickness):
        self.pen.set_thickness(thickness)

    def set_font_scale(self, font_scale):
        self.pen.set_font_scale(font_scale)

    def set_font(self, font):
        self.pen.set_font(font)

    def draw_text(self, src_img, text, point):
        cv2.putText(src_img, text, point, self.pen.font, self.pen.font_scale,
                    self.pen.color, self.pen.thickness)

    def draw_rectangle(self, src_img, left_top_point, right_bottom_point):
        cv2.rectangle(src_img, left_top_point, right_bottom_point,
                      self.pen.color, self.pen.thickness)

    def draw_circle(self, src_img, point, radiness):
        cv2.circle(src_img, point, radiness, self.pen.color, -1)

    def draw_polygon(self, src_img, points, line_color, draw_text=True):
        self.set_color(line_color)
        if not isinstance(points, np.ndarray) and not isinstance(points, list):
            raise TypeError("points should be object of list or np.array")

        if isinstance(points, list):
            points = np.array(points)

        assert (points.shape[0] == 4
                ), "the num of polygon should be 4, but this is {}".format(
                    points.shape)

        for i in range(4):
            cv2.circle(src_img, (points[i][0], points[i][1]), 15,
                       self.pen.color, -1)
            if draw_text:
                self.draw_text(src_img, "poly_{}".format(i),
                              (points[i][0], points[i][1]))
            sec_index = i + 1
            if sec_index == 4:
                sec_index = 0
            cv2.line(src_img, (points[i][0], points[i][1]),
                     (points[sec_index][0], points[sec_index][1]),
                     self.pen.color, self.pen.thickness)

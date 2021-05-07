from threading import Thread

import pytesseract
import cv2
import math
import re


def rotate_image(angle, point, image):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(point, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))


def crop_image(initial_point, height, width, image):
    xi = initial_point[0] - width
    xf = initial_point[0]
    yi = initial_point[1]
    yf = initial_point[1] + height

    if image.shape[0] < yf:
        yf = image.shape[0]
        yi = yf - height

    if image.shape[1] < xf:
        xf = image.shape[1]
        xi = xf - width

    return image[yi:yf, xi:xf]


def normalize_text(text):
    return re.sub('[^A-Z0-9]', '', text)


def count_non_black_pixels(image):
    non_black_pixels = 0
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            if image[y, x] != 0:
                non_black_pixels += 1
    return non_black_pixels


def get_neighbours(img, y, x):
    neighbours = []
    if y + 1 < len(img):
        neighbours.append((y + 1, x))
    if y - 1 >= 0:
        neighbours.append((y - 1, x))
    if x + 1 < len(img[y]):
        neighbours.append((y, x + 1))
    if x - 1 >= 0:
        neighbours.append((y, x - 1))
    return neighbours


def paint_object(img, point):
    y, x = point
    img[y][x] = 255
    queue = [point]
    while queue:
        y, x = queue.pop()
        for neighbour in get_neighbours(img, y, x):
            y_v, x_v = neighbour
            color = img[y_v][x_v]
            if color == 0 and color != 255:
                img[y_v][x_v] = 255
                queue.append(neighbour)


def find_initial_edges(image, y, x):
    ix = x
    iy = y

    while image[iy+-1, ix] == 0 or image[iy, ix-1] == 0:
        if image[iy-1, ix] == 0:
            iy -= 1
        if image[iy, ix-1] == 0:
            ix -= 1

    return ix, iy


def find_final_edges(image, y, x):
    ix = x
    iy = y

    while image[iy+1, ix] == 0 or image[iy, ix+1] == 0:
        if image[iy+1, ix] == 0:
            iy += 1
        if image[iy, ix+1] == 0:
            ix += 1

    return ix, iy


def is_square(y1, x1, y2, x2):
    height = abs(y2 - y1)
    width = abs(x2 - x1)

    diff = abs(height - width)
    return diff < 35


def count_objects(image):
    squares = 0
    rectangle = 0
    for y in range(50, image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] == 0:
                iy, ix = find_initial_edges(image, y, x)
                my, mx = find_final_edges(image, y, x)
                if is_square(iy, ix, my, mx):
                    squares += 1
                else:
                    rectangle += 1
                paint_object(image, (y, x))

    return squares, rectangle


def map_template(path):
    image = cv2.imread(path, cv2.THRESH_BINARY)
    _, thresh_hold_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresh_hold_image


class ImageProcessor(Thread):
    color_blank_edge: int
    image_height: int
    image_width: int
    plate: str
    teressact_config: str
    tolerance: int
    path_templates: str

    def __init__(self, color_blank_edge, image_height, image_width, plate, teressact_config, tolerance, path_templates):
        Thread.__init__(self)
        self.color_blank_edge = color_blank_edge
        self.image_height = image_height
        self.image_width = image_width
        self.plate = plate
        self.teressact_config = teressact_config
        self.tolerance = tolerance
        self.path_templates = path_templates

    def run(self):
        template_map = {'AB01': map_template(self.path_templates + 'Gabarito AB01.jpg'),
                        'AB02': map_template(self.path_templates + 'Gabarito AB02.jpg'),
                        'AC01': map_template(self.path_templates + 'Gabarito AC01.jpg'),
                        'AC02': map_template(self.path_templates + 'Gabarito AC02.jpg')}

        image = cv2.imread(self.plate, cv2.THRESH_BINARY)
        # gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle, point1, point2 = self.discover_inclination(image)
        rotated_image = rotate_image(angle, point1, image)
        cropped_image = crop_image(point1, self.image_height, self.image_width, rotated_image)

        image_model_raw_text = (pytesseract.image_to_string(cropped_image, config=self.teressact_config))
        image_model_normalized_text = normalize_text(image_model_raw_text)

        template = template_map[image_model_normalized_text]
        _, thresh_hold_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)

        difference1 = cv2.subtract(thresh_hold_image, template)
        difference2 = cv2.subtract(template, thresh_hold_image)

        pixel_diff_count1 = count_non_black_pixels(difference1)
        pixel_diff_count2 = count_non_black_pixels(difference2)

        if pixel_diff_count1 < self.tolerance and pixel_diff_count2 < self.tolerance:
            print('plate: {:35s} passed'.format(self.plate))
            return

        original_squares, original_rectangles = count_objects(template)
        plate_squares, plate_rectangles = count_objects(thresh_hold_image)

        errors = []

        if original_rectangles != plate_rectangles:
            errors.append('has {} rectangles, expected {}'.format(plate_rectangles, original_rectangles))

        if original_squares != plate_squares:
            errors.append('has {} squares, expected {}'.format(plate_squares, original_squares))

        if original_rectangles == plate_rectangles and original_squares == plate_squares:
            errors.append('miss position')

        print('plate: {:35s} rejected, reason: {}'.format(self.plate, errors))

    def is_blank_color(self, color):
        return color < self.color_blank_edge

    def discover_inclination(self, image):
        height, width = image.shape[:2]

        xi = width
        point1 = None
        point2 = None
        is_angle_positive = False

        for y in range(0, height, 1):
            for x in range(0, width, 1):
                color = image[y, x]
                if (self.is_blank_color(color)) and (point1 is None):
                    point1 = (x, y)
                    xi = x
                    is_angle_positive = x > (width / 2)

                if (point1 is not None) and (self.is_blank_color(color)) and (x < xi) and is_angle_positive:
                    point2 = (x, y)
                    xi = x

                if (point1 is not None) and (self.is_blank_color(color)) and (x > xi) and not is_angle_positive:
                    point2 = (x, y)
                    xi = x

        angle = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
        if is_angle_positive:
            angle = math.degrees(angle)
        if not is_angle_positive:
            angle = math.degrees(angle) + 180
            point1, point2 = point2, point1

        return angle, point1, point2

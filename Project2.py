from os import listdir
from os.path import isfile

from project2 import image_processor
import pytesseract


def main():
    path_templates = 'images/templates/'
    path_plates = 'images/plates/'
    path_wring_plates = 'images/wrong_plates/'
    image_height = 295
    image_width = 602
    blank_color_edge = 190
    tolerance = 2100

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    tesseract_config = r'-c tessedit_char_whitelist=ABC012/'

    path = path_plates
    plates = []
    [plates.append(path + file) for file in listdir(path) if isfile(path + file)]
    # plates = [path + 'Slide3.JPG']

    path = path_wring_plates
    [plates.append(path + file) for file in listdir(path) if isfile(path + file)]
    # plates = [path + 'Slide4.JPG']

    processors = []
    for plate in plates:
        processor = image_processor.ImageProcessor(blank_color_edge, image_height, image_width, plate, tesseract_config,
                                                   tolerance, path_templates)
        processor.start()
        processors.append(processor)
        # break

    for processor in processors:
        processor.join()


if __name__ == '__main__':
    main()

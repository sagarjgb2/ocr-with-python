import cv2
import pytesseract
from pytesseract import image_to_string

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Computer\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

img = cv2.imread('img.png')

#get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(img,5)

#thresholding
def thresholding(image):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img = get_grayscale(img)
img = remove_noise(img)
img = thresholding(img)

print('===================================================START===================================================================')
print(ocr_core(img))
print('===================================================DONE===================================================================')

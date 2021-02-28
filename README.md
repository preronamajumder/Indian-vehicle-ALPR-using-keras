# Indian-vehicle-ALPR-using-keras

Do ALPR on a single image or multiple images in a folder to get the License Plate Number.
Input image required is that of a cropped license plate. Example images are provided in license_plate folder.
Character segmentation is performed first and then the recognition model is applied to the individual characters.
Keras sequential model used to perform OCR. Input size of each character is 28x28. and labels file is provided for all the characters to recognise.

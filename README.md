# Indian-vehicle-ALPR-using-keras

Perform OCR on a single image or multiple images in a folder to get the License Plate Number.
Input image required is that of a cropped license plate. Example images are provided in license_plate folder.

![75](https://user-images.githubusercontent.com/38746630/109415860-262cba00-79e1-11eb-8bbe-7b4bac7d5420.jpg)

Character segmentation is performed first and then the recognition model is applied to the individual characters.

![char0](https://user-images.githubusercontent.com/38746630/109415983-b2d77800-79e1-11eb-9f55-a04f8fc5e4b4.jpg)   ![char1](https://user-images.githubusercontent.com/38746630/109415986-b539d200-79e1-11eb-8a8c-f79704cab3ca.jpg)   ![char2](https://user-images.githubusercontent.com/38746630/109415999-c256c100-79e1-11eb-8886-d8808a4febf0.jpg)   ![char3](https://user-images.githubusercontent.com/38746630/109415990-b79c2c00-79e1-11eb-9e2b-48d94c479151.jpg)   ![char4](https://user-images.githubusercontent.com/38746630/109415991-b965ef80-79e1-11eb-8da3-a502a0ff5846.jpg)   ![char5](https://user-images.githubusercontent.com/38746630/109415995-bbc84980-79e1-11eb-860b-c66feaa3dc01.jpg)   ![char6](https://user-images.githubusercontent.com/38746630/109416093-2f6a5680-79e2-11eb-9135-93428a4b5a5f.jpg)   ![char7](https://user-images.githubusercontent.com/38746630/109416013-d3073700-79e1-11eb-8bf2-f8aca297270d.jpg)   ![char8](https://user-images.githubusercontent.com/38746630/109416015-d4d0fa80-79e1-11eb-9b76-058f057aee95.jpg)   ![char9](https://user-images.githubusercontent.com/38746630/109416016-d69abe00-79e1-11eb-8212-a426f6b3366a.jpg)

Keras sequential model is used to perform OCR. Input size of each character is 28x28. Label file is provided for all the characters to recognise.

<img width="231" alt="Screen Shot 2021-02-28 at 4 26 55 PM" src="https://user-images.githubusercontent.com/38746630/109416064-0c3fa700-79e2-11eb-9c45-2e7ffa2a70c0.png">

For ease of use a config file is provided for model path, label path, image path, folder path. if is_folder = 0, then, OCR is performed on single image. If is_folder = 1, then, OCR is performed on a collection of images present in a folder. 

For training your own model train_alpr.py is provided. Data is provided in ocr_char. Provision for augmentation also given.

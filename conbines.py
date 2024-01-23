
from PIL import Image
import os
import numpy as np
from skimage import io


folder_path = 'test_image_08/rggb_all/res'
angRes = 4

def conbines(input_path,output_path,angRes):
    # get file name
    image_path = os.listdir(folder_path)
    for path in image_path: 
        save_name = (path).split('.')[-1]
        # print(save_name)
        if  not os.path.exists(output_path):
            os.makedirs(output_path)
        save_path = output_path + save_name +'.tif'
        # print(save_path)
        image_files = [f for f in os.listdir(os.path.join(folder_path,path)) if os.path.isfile(os.path.join(folder_path, path, f))]
        # load the image's width and height
        frist_image_path = os.path.join(folder_path,path ,image_files[0])
        frist_image = Image.open(frist_image_path)
        width,height = frist_image.size
        #creat NumPY
        combined_image = np.zeros((angRes*angRes,height,width),dtype= np.uint8)
        if len(image_files) != angRes * angRes:
           print('the number of image wrong!')
        else:
           for i ,image_file in enumerate(image_files):
            # print(image_file)
            image_path = os.path.join(folder_path,path ,image_file)
            print(image_path)
            img = Image.open(image_path)
            img_data = np.array(img)
            combined_image[i ,: ,:] = img_data
        io.imsave(save_path,combined_image,plugin='tifffile')
    
if __name__ =='__main__':
    input_path =  'test_image_08/rggb_all/res'
    angRes = 4
    output_path = 'combined_image/08/res'
    PATH=conbines(input_path,output_path,angRes)
    print(PATH)
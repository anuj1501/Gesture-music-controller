import os
import os.path
import cv2
import glob

preprocessed_folder = 'Gesture_Preprocessed_Data'
extract_folder = 'Gesture Image Data'
#get a list of all the images that we need to preprocess

letters = [chr(i) for i in range(65,91)]
numbers = ['1','2','3','4','5','6','7','8','9']

folder_names = letters + numbers

for folder_name in folder_names:

    print("Status :: Preprocessing Folder {}/{}".format(folder_name,len(folder_names)))
    
    folder_path = os.path.join(extract_folder,folder_name)

    images = glob.glob(os.path.join(folder_path,"*"))
    
    for (i,image_path) in enumerate(images):

        if i>200:
            break
        
        print("Status :: Preprocessing Image {}/{}".format(i+1,len(images)))

        image_name = os.path.basename(image_path)
        
        name = image_name.split('.')

        img = cv2.imread(image_path)

        image = cv2.resize(img,(400,400),interpolation = cv2.INTER_AREA)

        image_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image_grey_final = cv2.threshold(image_grey,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

        img_rotate1 = cv2.rotate(image_grey,cv2.ROTATE_90_CLOCKWISE)

        image_rotate1_final = cv2.threshold(img_rotate1,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

        img_rotate2 = cv2.rotate(image_grey,cv2.ROTATE_90_COUNTERCLOCKWISE)

        image_rotate2_final = cv2.threshold(img_rotate2,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

        image_lf = image_grey[:,85:]

        image_left = cv2.resize(image_lf,(400,400),interpolation=cv2.INTER_AREA)

        image_left_final = cv2.threshold(image_left,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

        image_rt = image_grey[:,:225]

        image_right = cv2.resize(image_rt,(400,400),interpolation=cv2.INTER_AREA)

        image_right_final = cv2.threshold(image_right,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

        #cv2.imshow('image',image3)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        save_path = os.path.join(preprocessed_folder,folder_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #print(save_path)
        grey_full_path = os.path.join(save_path,"{}001.jpg".format(i))

        rotate1_full_path = os.path.join(save_path,"{}002.jpg".format(i))

        rotate2_full_path = os.path.join(save_path,"{}003.jpg".format(i))

        left_full_path = os.path.join(save_path,"{}004.jpg".format(i))

        right_full_path = os.path.join(save_path,"{}005.jpg".format(i))
        #print(full_path)
        cv2.imwrite(grey_full_path,image_grey_final)

        cv2.imwrite(rotate1_full_path,image_rotate1_final)

        cv2.imwrite(rotate2_full_path,image_rotate2_final)

        cv2.imwrite(left_full_path,image_left_final)

        cv2.imwrite(right_full_path,image_right_final)










import numpy as np                                                                                   

IMAGE_SIZE = 512 #256                                                                                
                                                         
def normalize_x(image):                                                                              
    image = image/127.5 - 1                                                                          
    return image                                                                                     

                                                            
def normalize_y(image):                                                                              
    image = image/255                                                                                
    return image                                                                                     

                                                                   
def denormalize_y(image):                                                                            
    image = image*255                                                                                
    return image                                                                                     

                                                    
def load_X(folder_path):                                                                             
    import os, cv2                                                                                   
                                                                                                     
    image_files = []                                                                                 
                                                         
                                                                                                     
    for file in os.listdir(folder_path):                                                             
        base, ext = os.path.splitext(file)                                                           
        if ext == '.png':                                                                            
            image_files.append(file)                                                                 
        else :                                                                                       
            pass                                                                                     
                                                                                                     
    image_files.sort()                                                                               
    print (image_files)                                                                              
    #image_files = image_files[1:]                                                                   
    print(image_files)                                                                               
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)                     
    for i, image_file in enumerate(image_files):                                                     
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)                  
        print (image.shape)                                                                          
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))                                          
        image = image[:, :, np.newaxis]                                                              
        images[i] = normalize_x(image)                                                               
    return images, image_files                                                                       


def load_Y(folder_path):                                                                             
    import os, cv2                                                                                   

    image_files = []                                                                                 
    #image_files = os.listdir(folder_path)                                                           
    for file in os.listdir(folder_path):                                                             
        base, ext = os.path.splitext(file)                                                           
        if ext == '.png':                                                                            
            image_files.append(file)                                                                 
        else :                                                                                       
            pass                                                                                     
                                                                                                     
    image_files.sort()                                                                               
    print (image_files)                                                                              
    image_files = image_files                                                                        
    #image_files = image_files[1:]                                                                   
    print (image_files)                                                                              
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)                     
    for i, image_file in enumerate(image_files):                                                     
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)                  
        print (image)                                                                                
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))                                          
        image = image[:, :, np.newaxis]                                                              
        images[i] = normalize_y(image)                                                               
    print(images.shape)                                                                              
    return images, image_files                                                                       

def load_X_for_Y(Y_file_names, folder_path):                                                         
    import os, cv2                                                                                   
                                                                                                     
    X_files = Y_file_names                                                                           
    image_files = []                                                                                 
    for file in os.listdir(folder_path):                                                             
        base, ext = os.path.splitext(file)                                                           
        if ext == '.png':                                                                            
            image_files.append(file)                                                                 
        else :                                                                                       
            pass                                                                                     

    #image_files = os.listdir(folder_path)                                                           
    image_files.sort()                                                                               
    images = np.zeros((len(X_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)                         
    for i, image_file in enumerate(X_files):                                                         
        ind = image_files.index(image_file)                                                          
        temp_img = cv2.imread(folder_path + os.sep + image_files[ind], cv2.IMREAD_GRAYSCALE)         
        temp_img = cv2.resize(temp_img, (IMAGE_SIZE, IMAGE_SIZE))                                    
        temp_img = temp_img[:, :, np.newaxis]                                                        
        images[i] = normalize_x(temp_img)                                                            
                                                                                                     
    print(X_files)                                                                                   
    print(images.shape)                                                                              
                                                                                                     
    return images, image_files                 
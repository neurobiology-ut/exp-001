import os                                                                                            
import numpy as np                                                                                   
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger                                
from keras.preprocessing.image import ImageDataGenerator                                             

from load import *                                                                                   
from model import *                                                                                  
from loss import *                                                                                   

def train_unet():                                                                                                       
    #X_train, file_names = load_X('512/train/cropped_512')                                                           
    Y_train1, Y_file_names = load_Y('../train/mito')                                                 
    Y_train2, Y_file_names = load_Y('../train/er')                                                   
    Y_train = np.concatenate([Y_train1, Y_train2], axis=3)                                           
    X_train, X_file_names = load_X_for_Y(Y_file_names, '../train/ori')                               
                                                                                                     
    print(X_train.shape)                                                                             
    print(Y_train.shape)                                                                             
                                                                                                     
                                                                                                     
    # we create two instances with the same arguments                                                
    data_gen_args = dict(                                                                            
        #rotation_range=90.,                                                                         
        #width_shift_range=0.1,                                                                      
        #height_shift_range=0.1,                                                                     
        #shear_range=0.2,                                                                            
        #zoom_range=0.2,                                                                             
        horizontal_flip=True,                                                                        
        vertical_flip=True                                                                           
    )                                                                                                
    image_datagen = ImageDataGenerator(**data_gen_args)                                              
    mask_datagen = ImageDataGenerator(**data_gen_args)                                               

    # Provide the same seed and keyword arguments to the fit and flow methods                        
    seed = 1                                                                                         
    image_datagen.fit(X_train, augment=True, seed=seed)                                              
    mask_datagen.fit(Y_train, augment=True, seed=seed)                                               

    image_generator = image_datagen.flow(X_train, seed=seed, batch_size=8)                           
    mask_generator = mask_datagen.flow(Y_train, seed=seed, batch_size=8)                             

    # combine generators into one which yields image and masks                                       
    train_generator = zip(image_generator, mask_generator)                                           
                                                                                                     
    model = get_unet_512()                                                                           

    BATCH_SIZE = 16                                                                                                                                                 
    NUM_EPOCH = 400                                                                                  
                                                                                                     
    callbacks = []                                                                                   
    callbacks.append(CSVLogger("../train/history.csv"))                                              
    history = model.fit_generator(train_generator,steps_per_epoch=32, epochs=NUM_EPOCH, verbose=1, ca
llbacks=callbacks)                                                                                   
    #history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, callba
cks=callbacks)                                                                                       
    model.save_weights('../train/unet_weights.hdf5')                                                 

if __name__ == "__main__":                                                                           
    train_unet() 
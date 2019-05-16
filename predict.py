from load import *                                                                                   
from model import *                                                                                  
from loss import *                                                                                   

def predict():                                                                                       
    import cv2                                                                                       
                      
    X_test, file_names = load_X('../test')                                                           
    print (file_names)                                                                               

    #input_channel_count = 1                                                                         
    #output_channel_count = 1                                                                        
    #first_layer_filter_count = 64                                                                   
                                                                                                     
    #network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)             
    #model = network.get_model()                                                                     
                                                                                                     
    model = get_unet_512()                                                                           
                                                                                                     
    model.load_weights('../train/unet_weights.hdf5')                                                 
    BATCH_SIZE = 1                                                                                   
    Y_pred = model.predict(X_test, BATCH_SIZE)                                                       

    for i, y in enumerate(Y_pred):                                                                                      
        img = cv2.imread('../test/'+ file_names[i], cv2.IMREAD_GRAYSCALE)                            
        #y = cv2.resize(y, (img.shape[1], img.shape[0]))                                             
        y1 = y[:,:,0]                                                                                
        y2 = y[:,:,1]                                                                                
        cv2.imwrite('../pred/mito/' + file_names[i], denormalize_y(y1))                              
        cv2.imwrite('.../pred/er/' + file_names[i], denormalize_y(y2))                               

if __name__ == "__main__":                                                                           
    predict()   
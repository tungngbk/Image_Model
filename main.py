
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import urllib.request

# #Load the trained model. (Pickle file)
# my_model = load_model('./Models/VGG16_batchsize=32_epo=10.h5')

# def getPrediction(img_path):
#     SIZE = 150 
#     # img_path="https://lptech.asia/uploads/files/2020/07/10/seo-hinh-anh-la-gi-lptech.png"
#     urllib.request.urlretrieve(img_path, "PredictImage")
#     img = np.asarray(Image.open("PredictImage").resize((SIZE,SIZE)))
    
#     img = img/255.      #Scale pixel values
    
#     img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
#     pred = my_model.predict(img) #Predict    
#     return pred


model = load_model("./Models/VGG16_batchsize=32_epo=10.h5")
def getPrediction(img_path):
    urllib.request.urlretrieve(img_path, "PredictImage")
    
    SIZE=120
    a=[]
    img = np.asarray(Image.open("PredictImage").convert('L').resize((SIZE,SIZE)))
    a.append(img)
    a=np.array(a)
    a=a.reshape(-1, SIZE,SIZE, 1)
    a=a/255    
    
    pred = model.predict(a)[0][0] #Predict    
    return pred





# test_prediction =getPrediction('https://phuongnamcons.vn/wp-content/uploads/2021/06/vet-nut-tuong-nho.jpg')
# print(test_prediction)


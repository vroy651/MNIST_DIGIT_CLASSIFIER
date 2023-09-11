import streamlit as st
# minist digit recoginition
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
import torchvision
import cv2
from PIL import Image
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# define LENET model
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    #layer 1
    self.conv1=nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2)
    self.act1=nn.Tanh()
    self.pooling1=nn.AvgPool2d(kernel_size=2,stride=2)

    #layer2
    self.conv2=nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0)
    self.act2=nn.Tanh()
    self.pooling2=nn.AvgPool2d(kernel_size=2,stride=2)

    #leyer3
    self.conv3=nn.Conv2d(16,120,kernel_size=5,stride=1,padding=0)
    self.act3=nn.Tanh()

    #flatten the feature map

    self.flat=nn.Flatten()
    #linear layer
    self.linear1=nn.Linear(1*1*120,84)
    self.act4=nn.Tanh()
    self.linear2=nn.Linear(84,10)

  # forward pass
  def forward(self,input):

    #input->1*128*128,output->6*128*128
    input=self.act1(self.conv1(input))

    # averagepooling, input->6*128*128,output->6*14*14
    input=self.pooling1(input)

    #input->6*14*14,output->16*10*10
    input=self.act2(self.conv2(input))

    #averagepooling, input->16*10*10,output->16*5*5
    input=self.pooling2(input)

    #input->16*5*5,output->120*1*1
    input=self.act3(self.conv3(input))

    # linear layer1 ,input ->120*1*1,output->85
    input=self.act4(self.linear1(self.flat(input)))

    #linear layer2 ,input->85,output->10
    input =self.linear2(input)

    return input

# define loaded_model
model_loaded=CNN()

# load the model
model_loaded.load_state_dict(torch.load('/Users/vishalroy/MNIST_DIGIT_CLASSIFIER/model.pth'))
model_loaded.eval()

# preproccsing image before evaluating
# def process(img_path):
#   img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
  
#   #resize the image to 28*28
#   img=cv2.resize(img,(28,28))

#   return img

def preprocessing(img):
    try:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        return img
    except Exception as e:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        return img

# test the model
# img_file=process('/Users/vishalroy/MNIST_DIGIT_CLASSIFIER/images/img_41790.jpg')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,))
])


# #tranform new images into required format
# up_img=Image.open('/Users/vishalroy/MNIST_DIGIT_CLASSIFIER/images/img_41790.jpg')
# st.image(up_img)
# img = np.asarray(up_img)
# img = cv2.resize(img, (28,28))
# img = preprocessing(img)
# # img=transform(up_img)
# img=transform(img)

# # predict the number 
# model_loaded.eval()

# with torch.no_grad():
#   prediction=model_loaded(img.unsqueeze(0))
#   print(prediction.shape)

# predicted_class = torch.argmax(prediction, dim=1).item()

# print(f"predicted number is :{predicted_class}",type(prediction))
# print(np.amax(prediction.numpy()))

def main():
    st.title("Handwritten Digit Classification Web App")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    activities = ["Program", "Credits"]
    choices = st.sidebar.selectbox("Select Option", activities)

    if choices == "Program":
        st.subheader("Kindly upload file below")
        img_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            up_img=Image.open(img_file)
            st.image(up_img)
        if st.button("Predict Now"):
            try:
                # tranform new images into required format
                img = np.asarray(up_img)
                img = cv2.resize(img, (28, 28))
                img = preprocessing(img)
                img=transform(img)
                with torch.inference_mode():
                    prediction=model_loaded(img.unsqueeze(0))
                classIndex = torch.argmax(prediction,dim=1).item()
        
                if np.argmax(prediction.numpy()) > 0.90:
                    if classIndex == 0:
                        st.success("0")
                        # speak("Predicted Number is Zero")
                    elif classIndex == 1:
                        st.success("1")
                        # speak("Predicted Number is One")
                    elif classIndex == 2:
                        st.success("2")
                        #speak("Predicted Number is Two")
                    elif classIndex == 3:
                        st.success("3")
                        # speak("Predicted Number is Three")
                    elif classIndex == 4:
                        st.success("4")
                        # speak("Predicted Number is Four")
                    elif classIndex == 5:
                        st.success("5")
                        # speak("Predicted Number is Five")
                    elif classIndex == 6:
                        st.success("6")
                        # speak("Predicted Number is Six")
                    elif classIndex == 7:
                        st.success("7")
                        # speak("Predicted Number is Seven")
                    elif classIndex == 8:
                        st.success("8")
                        # speak("Predicted Number is Eight")
                    elif classIndex == 9:
                        st.success("9")
                        # speak("Predicted Number is Nine")
                   
                else:
                    st.success("Invalid input image or Image too large")
            except Exception as e:
                st.error("Connection Error")

    elif choices == 'Credits':
        st.write(
            "Application Developed by Vishal.")


if __name__ == '__main__':
    main()
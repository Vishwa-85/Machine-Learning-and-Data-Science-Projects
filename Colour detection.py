'''
The Sparks Foundation Graduate Rotational Internship Program (GRIP) - August 2021
Computer Vision and Internet of Things

TASK #2: Color Identification in Images

Algorithm used: K-Nearest Neighbours (KNN) Algorithm
Step 1: Convert the image into points
Step 2: Calculate the color distance
Try to find nearest color using RGB values. Nearest color that matches these three values will be extracted

'''

#Importing necessary libraries
import pandas as pd
import cv2

#Reading the image
img = cv2.imread("flowers.jpg")

#Read the csv file and add the relevant names to columns
index=["color", "color_name", "hex", "R", "G", "B"]
color_file = pd.read_csv('colors.csv', names=index, header=None)

clicked = False
r = g = b = xpos = ypos = 0

#Get color function: To find nearest color from the dataset and extract the color
def get_color(R,G,B):
    minimum = 10000
    for i in range(len(color_file)):
        dist = abs(R- int(color_file.loc[i,"R"])) + abs(G- int(color_file.loc[i,"G"]))+ abs(B- int(color_file.loc[i,"B"]))
        if(dist<=minimum):
            minimum = dist
            colorName = color_file.loc[i,"color_name"]
    return colorName

#Click function: To get the image colors in the form of integer values. These integer values will be used later
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)

#Opens the window 'Color Identification' and displays image in the window.
cv2.namedWindow('Color Identification')
cv2.setMouseCallback('Color Identification', click_event)

#Working of the application window 
while(1):
    cv2.imshow("Color Identification",img)
    if (clicked):   
        #Create a rectangular text box in which the text is supposed to be written
        cv2.rectangle(img,(20,20), (350,60), (b,g,r), -1)    
        #Function call to extract color
        text = get_color(r,g,b)        
        #To write the text indicating the color 
        cv2.putText(img, text,(50,50),3,1,(255,255,255),2,cv2.LINE_AA)
        if(r+g+b>=600):
            cv2.putText(img, text,(50,50),3,1,(0,0,0),2,cv2.LINE_AA)
            
        clicked=False
        
    #Break the loop when user hits 'esc' key    
    if cv2.waitKey(20) & 0xFF ==27:
        break
    
#When loop is broken, this will close the application window
cv2.destroyAllWindows()


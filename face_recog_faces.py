from PIL import Image, ImageDraw
from IPython.display import display

# to get frames & iterate through them
import frames
import os
from os import listdir

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

def main():
    #call frames - command line tho??
    #for video in os.listdir(r'C:\Users\lilyy\Documents\Social IQ\facial detection\raw'): #C:\Users\lilyy\Downloads\Social-IQ.zip\raw\vision\raw'): #iterates through all videos in a folder w all 1250 videos
        #command = "python frames.py " + video #command-line prompt 
        #os.system('cmd /k' + command) #running prompt above, splits video into frames
        #filename, _ = os.path.splitext(video) #this is the name of the folder with all frames of ONE video
        #filename += "-opencv" 

        folder_dir = r'C:\Users\lilyy\Documents\Social IQ\facial detection\cra-opencv' + os.path(video) 
        newpath = r'C:\Users\lilyy\Documents\Social IQ\photos' + video #make new folder for photos 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        tupleList = []
        unique_faces = []
        unique_face_locations = []

        for frame in os.listdir(folder_dir): 
            image = mpimg.imread(folder_dir+"\\"+frame) # for mpl, photo we want to find faces in?
            #plt.imshow(image)
            #plt.show()

            # Find all the faces and face encodings in the unknown image   
            face_locations = face_recognition.face_locations(image)

            face_encodings = face_recognition.face_encodings(image, face_locations)  #********** THIS IS WHAT WE WANT 
            if len(face_encodings) == 0:
                continue

            for i in range(len(face_encodings)):
                #tuple of (face_encoding, face_location)
                tupleList.append((face_encodings[i], face_locations[i]))

            #face_encodings = np.array(face_encodings) 
            #face_encodings = face_encodings.flatten() #big 1D list of all the faces in vid
            #print(face_encodings)
            # Loop through each face found in the unknown image
            
            for (enc, loc) in tupleList: #use encodings
                compFaces = face_recognition.compare_faces(unique_faces, enc)  #face_distances = face_recognition.face_distance(face_encodings, j) #compares list of faces in one frame (i) to list of unique faces (j)
                if True in compFaces: #len(face_distances)
                    pass
                else:
                    unique_faces.append(enc)
                    unique_face_locations.append((image, loc))

            # The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
            # be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
            # positive matches at the risk of more false negatives.
            # smaller distance are more similar to each other than ones with a larger distance.
        

        #print(unique_faces)
        for (image, loc) in unique_face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = loc
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image, 'RGB')
            pil_image.show()
            newpath.write(pil_image)
    


if(__name__ == "__main__"):
    main()




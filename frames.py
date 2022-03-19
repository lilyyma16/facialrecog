#given video, extracts a frame every second and returns the frames as a list
#uses opencv

#video is passed through command-line arguments
#python frames.py cra.mp4

import cv2
import numpy as np
import os

# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
# just save one frame per second
SAVING_FRAMES_PER_SECOND = 1

# format timedelta helper function not included (need package import as well)


# helper function 

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    if(cap.get(cv2.CAP_PROP_FPS)!=0):
        clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    else:
        clip_duration = 1
    
    # use np.arange() to make floating-point steps
    if(saving_fps==0): saving_fps = 1
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


# main function
def main(video_file):
    filename, _ = os.path.splitext(video_file)
    filename += "-opencv"
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.mkdir(filename)
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            cv2.imwrite(os.path.join(filename, f"frame{frame_duration}.jpg"), frame) 
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1

if __name__ == "__main__":
    import sys
    for video_file in os.listdir(r"C:\Users\lilyy\Documents\Social IQ\facial detection\test"):
    #video_file = sys.argv[1]
        main(video_file)





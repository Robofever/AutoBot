import cv2
import os
import numpy as np
from tqdm import tqdm 

path = os.getcwd()

frames = os.path.join(path,"frames")
os.mkdir(frames)

l = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))]
# print(l)
pbar = tqdm(total=len(l))
for videos in l:
    cap = cv2.VideoCapture(os.path.join(path,videos))
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            name = videos 
            name += f"{count}.jpg"
            frame_path = os.path.join(frames,name)
            # print(frame_path)
            # frame = np.array(frame,dtype=np.float32)
            # cv2.imshow("video",frame)
            cv2.imwrite(filename=frame_path,img=frame)
            count += 1
        else :
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pbar.update()
    cap.release()
    cv2.destroyAllWindows()
pbar.close()
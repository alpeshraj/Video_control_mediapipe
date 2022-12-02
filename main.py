import vlc
from facemesh import FaceMeshDetector
import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    pTime=0
    media_player = vlc.MediaPlayer()
    media = vlc.Media("/home/mongoose/Downloads/Projects/face_control_final/sample_video_-_3_minutemp4.mp4 (720p).mp4")
    media_player.set_media(media)
    detector = FaceMeshDetector()
    media_player.play()
    while True:
        success , img = cap.read()
        img,faces,res,lf,rf,le,re,fdy,lpy = detector.findFaceMesh(img,draw=False)
        if len(faces)!=0:
            if fdy-lpy > 65:
                media_player.set_rate(2)
                print('face up')
            elif fdy-lpy < 40:
                media_player.set_rate(0.5)
                print('face down')    
            elif re-lf < 10:
                value = media_player.get_time()
                media_player.set_time(value+1000)
                print('video forward')
            elif rf-le < 10:
                value = media_player.get_time()
                media_player.set_time(value-1000)
                print('video backward')
            else:
                media_player.set_rate(1)
                media_player.play()
        else:
            media_player.set_pause(1)
        cTime = time.time()
        fps = 1 / (cTime -  pTime)
        pTime=cTime
        cv2.putText(img,f'FPS {str(int(fps))}',(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            media_player.stop()
            break
    

if __name__=="__main__":
    main()
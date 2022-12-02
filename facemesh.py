import itertools
import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    # def getunique(c):
    #     temp_list = list(c)
    #     temp_set = set()
    #     for t in temp_list:
    #         temp_set.add(t[0])
    #         temp_set.add(t[1])
    #     return list(temp_set)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []

        LEFT_EYE_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_LEFT_EYE)))
        RIGHT_EYE_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_RIGHT_EYE)))
        FACE_OVAL_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_FACE_OVAL)))
        LIPS_INDEXES = list(set(itertools.chain(*self.mpFaceMesh.FACEMESH_LIPS)))
        # FACE_OVAL_INDEXES = self.getunique(CONNECTION_FACE_OVAL)
        # print(len(LEFT_EYE_INDEXES))
        # print(len(RIGHT_EYE_INDEXES))
        # print(len(FACE_OVAL_INDEXES))

        lsum = 0
        rsum = 0
        le = 0
        re = 0
        lf = 0
        rf = 0

        lex = 0
        rex = 0
        lfx = 0
        rfx = 0

        ley = 0
        rey = 0
        lfy = 0
        rfy = 0

        lpx = 0
        fdx = 0
        lpy = 0
        fdy = 0

        if self.results.multi_face_landmarks:
            for faceid, faceLms in enumerate(self.results.multi_face_landmarks):

                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                               self.drawSpec, self.drawSpec)

                # print(f'LEFT EYE LANDMARKS:n')
                lsum = 0
                rsum = 0
                l = {}
                for index in LEFT_EYE_INDEXES:
                    x = int(faceLms.landmark[index].x * img.shape[1])
                    y = int(faceLms.landmark[index].y * img.shape[0])
                    if index == 263:
                        l[index] = (x, y)
                        le = x
                        lex = x
                        ley = y

                # print(f'RIGHT EYE LANDMARKS:n')
                r = {}
                for index in RIGHT_EYE_INDEXES:
                    x = int(faceLms.landmark[index].x * img.shape[1])
                    y = int(faceLms.landmark[index].y * img.shape[0])
                    if index == 33:
                        r[index] = (x, y)
                        re = x
                        rex = x
                        rey = y
                d = {}
                # print(f'FACE OVAL LANDMARKS:n')
                for index in FACE_OVAL_INDEXES:
                    x = int(faceLms.landmark[index].x * img.shape[1])
                    y = int(faceLms.landmark[index].y * img.shape[0])
                    d[index] = (x, y)
                    if (index == 127):
                        lf = x
                        lfx = x
                        lfy = y
                    if (index == 356):
                        rf = x
                        rfx = x
                        rfy = y
                    if (index == 152):
                        fdy = y
                    # print(x,y)
                lp = {}

                for index in LIPS_INDEXES:
                    x = int(faceLms.landmark[index].x * img.shape[1])
                    y = int(faceLms.landmark[index].y * img.shape[0])
                    lp[index] = (x, y)
                    if index == 14:
                        lpy = y
                # print('lips:')
                for index in LIPS_INDEXES:
                    if index == 14:
                        cv2.circle(img, (lp[index][0], lp[index][1]),
                                   2, (0, 255, 0), 2)
                    if index == 127:
                        cv2.circle(img, (d[index][0], d[index][1]),
                                   2, (0, 255, 0), 2)
                for index in FACE_OVAL_INDEXES:
                    # cv2.circle(img,(d[index][0],d[index][1]),
                    #             2,(0,255,0),2)
                    if index == 152:
                        cv2.circle(img, (d[index][0], d[index][1]),
                                   2, (0, 255, 0), 2)
                    if index == 127:
                        cv2.circle(img, (d[index][0], d[index][1]),
                                   2, (0, 255, 0), 2)
                    if index == 356:
                        cv2.circle(img, (d[index][0], d[index][1]),
                                   2, (0, 255, 0), 2)
                # print('right\n')       
                for index in RIGHT_EYE_INDEXES:
                    if index == 33:
                        cv2.circle(img, (r[index][0], r[index][1]),
                                   2, (0, 255, 0), 2)

                # print('left\n')    
                for index in LEFT_EYE_INDEXES:
                    if index == 263:
                        cv2.circle(img, (l[index][0], l[index][1]),
                                   2, (0, 255, 0), -1)

                # cv2.line(img,(rex,lfy),(lfx,rey),(255,0,255),5)

                # lsum+=lx+ly
                # print(lsum)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x + iw), int(lm.y * ih)

                    # print(id,x,y)
                    face.append([x, y])
                faces.append(face)

        return img, faces, lsum - rsum, lf, rf, le, re, fdy, lpy


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces, res, lf, rf, le, re, fdy, lpy = detector.findFaceMesh(img, draw=False)
        if len(faces) != 0:
            if fdy - lpy > 65:
                print('volume up')
            if fdy - lpy < 40:
                print('volume down')
            if re - lf < 10:
                print('video forward')
            if rf - le < 10:
                print('video backward')
            # print(res)
            # if res<5:
            #     print("Video Forward")
            # elif res>18:
            #     print("Video Backward")
            # print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

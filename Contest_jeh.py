import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import *
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import datetime

form_class = uic.loadUiType("Contest_jeh.ui")[0]





# 윈도우 클래스 선언
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(QSize(428, 388))
        self.stackedWidget.setCurrentIndex(0)

        self.timerLabel.setText("00:00:00")
        self.btn_start.clicked.connect(self.start_timer)
        self.time = QTime(0, 0, 0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer_label)

        self.btn_reset.clicked.connect(self.timeReset)

        # 레이블 정의
        self.actions = ['not_stretching', 'stretching']
        # sequence 길이 정의
        self.seq_length = 30
        # stretching 카운트 정의
        self.count_time = 0
        self.counter = 0

        # 학습된 모델 로드
        model = load_model('models_TEST/model_1.h5')

        self.model_list = [model]

    def start_timer(self):
        self.stackedWidget.setCurrentIndex(1)
        self.timer.start(1000)

    def timeReset(self):
        self.stackedWidget.setCurrentIndex(0)
        self.timer.stop()
        self.timerLabel.setText("00:00:00")
        self.time = QTime(0, 0, 0)

    def update_timer_label(self):
        self.time = self.time.addSecs(1)
        self.timerLabel.setText(self.time.toString("hh:mm:ss"))
        # if self.time.minute() % 1 == 0 and self.time.second() == 0:
        if self.time.second() % 1 == 0:
                    # 웹캠 창 열기
                    # pass  # TODO: 웹캠 창 열기 코드 추가
            print("바보")
            self.webcamopen()

    def webcamopen(self):

        # MediaPipe pose 모델 정의
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(0)

        # sequence와 action sequence 초기화
        seq = []
        action_seq = []

        # 비디오 스트림으로부터 프레임을 읽어오며 무한 루프 수행
        while cap.isOpened():
            # 프레임 읽어오기
            ret, img = cap.read()
            img0 = img.copy()
            # 프레임 좌우반전 및 컬러 채널 변환
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Mediapipe pose 모델을 이용하여 포즈 예측 수행
            result = pose.process(img)
            # 이미지 색상채널 변환
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # 예측 결과가 존재할 경우, joint 위치 추출
            if result.pose_landmarks is not None:
                joint = np.zeros((33, 4))
                for j, lm in enumerate(result.pose_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                #         # Compute angles between joints
                #         v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,0,21,22,23,0,25,26,27,0,29,30,31], :2]  # Parent joint
                #         v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32], :2]  # Child joint
                #         v = v2 - v1  # [19, 3]

                #         # Normalize v
                #         v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                #         # Get angle using arcos of dot product
                #         angle = np.arccos(np.einsum('nt,nt->n',
                #                                     v[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], :],
                #                                     v[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], :]))  # [15,]

                # 관절 사이의 각도 계산
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19, 0, 21, 22, 23],
                     :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     :3]  # Child joint
                v = v2 - v1  # [19, 3]

                # v (벡터) 정규화
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 내적을 이용해 각도 계산
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                               21, 22], :],
                                            v[
                                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                             22, 23], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 입력 시퀀스가 seq_length보다 작으면 다음 프레임으로 넘어감
                if len(seq) < self.seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-self.seq_length:], dtype=np.float32), axis=0)

                # 모델에 입력 후 예측
                y_pred = self.model_list[0].predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                # 예측된 결과가 일정한 신뢰도 이상이 아니면 다음 프레임으로 넘어감
                if conf < 0.9:
                    continue

                action = self.actions[i_pred]
                action_seq.append(action)

                # action_seq가 5개 이하이면 다음 프레임으로 넘어감
                if len(action_seq) < 3:
                    continue

                # 이전 3개의 action이 모두 같으면 현재 action을 this_action으로 설정
                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                # 프레임에 this_action 텍스트 추가
                print(f'Text to add to image: {this_action.upper()}')
                # 프레임에 this_action 텍스트 추가
                cv2.putText(img, f'{this_action.upper()}', org=(int(result.pose_landmarks.landmark[0].x * img.shape[1])
                                                                , int(
                    result.pose_landmarks.landmark[0].y * img.shape[0] + 20))
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                #             #영상에 카운트타임 표시
                time = datetime.datetime.now()
                # cv2.putText(img, f'{time}', org=(10, 50)
                #             , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 155), thickness=2)
                # 영상에 횟수 표시
                cv2.putText(img, "'count : 1' is Success" , org=(10, 50)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 155), thickness=2)
                cv2.putText(img, f'count : {self.counter}', org=(10, 80)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 155),
                            thickness=2)

                if this_action.upper() == "STRETCHING":
                    count_time += 1
                    #          경과시간 약 5초일 때 카운트 올리기
                    if count_time == 30:
                        self.counter += 1


                else:
                    count_time = 0






            # 화면에 보여주기
            cv2.imshow('img', img)
            # 'q' 버튼을 누르면 반복문 종료
            if cv2.waitKey(1) == ord('q') or self.counter == 1:
                cv2.putText(img, 'success!', org=(100, 1000)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                            thickness=2)
                time.sleep(1)

                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()

    sys.exit(app.exec_())
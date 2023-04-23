import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import *
from dialog import dialog #다이얼로그 폼에서 다이얼로그 클래스 가져오기
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import random
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

form_class = uic.loadUiType("Contest_jeh.ui")[0]
# 윈도우 클래스 선언
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(QSize(412, 326))
        self.stackedWidget.setCurrentIndex(0)
        self.comboBox.setCurrentIndex(2)
        self.dialog = dialog()

        # QMediaPlayer 객체 생성 및 설정
        self.media_player = QMediaPlayer()
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile("sound.wav")))

        self.timerLabel.setText('00:00:00')
        self.btn_start.clicked.connect(self.start_timer)
        self.time = QTime(0, 0, 0)
        self.timer = QTimer()

        self.timer.timeout.connect(self.update_timer_label)
        self.btn_reset.clicked.connect(self.timeReset)
        self.btn_menual.clicked.connect(self.show_menual)
        self.dialog.btn_ok.clicked.connect(self.close_menual)

        # sequence 길이 정의
        self.seq_length = 30
        # stretching 카운트 정의
        self.count_time1 = 0
        self.count_time2 = 0
        self.counter = 0
        # 학습된 모델 로드
        self.model_1 = load_model('models_TEST/model_1.h5')
        self.model_2 = load_model('models_TEST/model_2.h5')

        self.timeResetSig = False
        self.successSig = False
        self.successSig2 = False

        self.functions = [self.arms_up,self.cross_arm]
    def show_menual(self): # 다이얼로그 폼열기
        self.dialog.show()

    def close_menual(self): # 다이얼로그 폼닫기
        self.dialog.close()
    def start_timer(self): # 시작 버튼
        self.stackedWidget.setCurrentIndex(1)
        self.timer.start(1000)
        self.comboBox.setEnabled(False)
    def timeReset(self): # 초기화 버튼
        self.comboBox.setEnabled(True)
        self.timeResetSig = True
        self.stackedWidget.setCurrentIndex(0)
        self.timer.stop()
        self.timerLabel.setText("00:00:00")
        self.time = QTime(0, 0, 0)
        self.successSig = False
        self.successSig2 = False
        self.count_time1 = 0
        self.count_time2 = 0
        self.counter = 0
    def update_timer_label(self):
        self.time = self.time.addSecs(1)
        self.timerLabel.setText(self.time.toString("hh:mm:ss"))
        if self.time.minute() % int(self.comboBox.currentText()) == 0 and self.time.second() == 0:
                    # 선택한 시간마다 웹캠 창 열기
                    # TODO: arms_up 모델 웹캠 열기
            # 소리 재생
            self.media_player.play()
            random_func = random.choice(self.functions)
            random_func()
        if self.time.hour() % 1 == 0 and self.time.minute() == 0 and self.time.second() == 0:
                    # 1시간이 되면 시간초기화
            self.timeReset()
            self.media_player.stop()
        else :
            self.media_player.stop()
    def arms_up(self):
        self.timeResetSig = False
        self.successSig = False
        self.successSig2 = False
        self.count_time1 = 0

        # 레이블 정의
        actions = ['not_stretching', 'stretching']

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
                y_pred = self.model_1.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                # 예측된 결과가 일정한 신뢰도 이상이 아니면 다음 프레임으로 넘어감
                if conf < 0.9:
                    continue

                action = actions[i_pred]
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
                cv2.putText(img, "Please stretch one's arms up!" , org=(5, 30)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                cv2.putText(img, f"Run until 'Status: 5'", org=(5, 80)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                            thickness=2)


                if this_action.upper() == "STRETCHING":
                    cv2.putText(img, f"Status: {int(self.count_time1)}", org=(350, 80)
                                , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(64, 255, 0),
                                thickness=2)
                    self.count_time1 += 0.1
                    #          경과시간 약 5초일 때 완료하기
                    # if self.count_time1 >= 5:
                    #     self.successSig = True
                else:
                    self.count_time1 = 0
            # 화면에 보여주기
            cv2.imshow('cam', img)
            # 'q' 버튼을 누르면 반복문 종료
            if cv2.waitKey(1) == ord('q') or self.count_time1 >= 5 or self.timeResetSig == True:
               break
        cap.release()
        cv2.destroyAllWindows()

    def cross_arm(self):
        print("팔교차스트레칭")
        self.timeResetSig = False
        self.successSig = False
        self.successSig2 = False
        self.count_time1 = 0
        self.count_time2 = 0
        self.counter = 0
        # 레이블 정의
        actions = [
            'not_stretching'
            ,'left stretching'
            ,'right stretching'
        ]
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
                y_pred = self.model_2.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                # 예측된 결과가 일정한 신뢰도 이상이 아니면 다음 프레임으로 넘어감
                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                # action_seq가 5개 이하이면 다음 프레임으로 넘어감
                if len(action_seq) < 3:
                    continue

                # 이전 3개의 action이 모두 같으면 현재 action을 this_action으로 설정
                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3] :
                    this_action = action

                # 프레임에 this_action 텍스트 추가
                print(f'Text to add to image: {this_action.upper()}')
                # 프레임에 this_action 텍스트 추가
                cv2.putText(img, f'{this_action.upper()}', org=(int(result.pose_landmarks.landmark[0].x * img.shape[1])
                                                                , int(
                    result.pose_landmarks.landmark[0].y * img.shape[0] + 20))
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                # 영상에 횟수 표시
                cv2.putText(img, "Please cross-arm left&right stretches!" , org=(5, 30)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                cv2.putText(img, f"Run until 'Status: 5'", org=(5, 80)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                            thickness=2)
                if self.successSig == False and self.successSig2 == False:
                    cv2.putText(img, f"One arm Status:", org=(5, 450)
                            , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(64, 255, 0),
                            thickness=2)
                elif self.successSig == True:
                    cv2.putText(img, f"One arm Status: Success ", org=(5, 450)
                                , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(64, 255, 0),
                                thickness=2)
                elif self.successSig2 == True:
                    cv2.putText(img, f"One arm Status: Success ", org=(5, 450)
                                , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(64, 255, 0),
                                thickness=2)

                if this_action.upper() == "LEFT STRETCHING":
                    self.count_time1 += 0.1
                    cv2.putText(img, f"Status: {int(self.count_time1)}", org=(350, 80)
                                , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(64, 255, 0),
                                thickness=2)
                    #   경과시간 약 5초일 때 카운트 올리기
                    if self.count_time1 >= 5:
                        self.successSig = True
                elif this_action.upper() == "RIGHT STRETCHING":
                    self.count_time2 += 0.1
                    cv2.putText(img, f"Status: {int(self.count_time2)}", org=(350, 80)
                                , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(64, 255, 0),
                                thickness=2)
                    print(self.successSig2)
                    #   경과시간 약 5초일 때 카운트 올리기
                    if self.count_time2 >= 5:
                        self.successSig2 = True
                else:
                    self.count_time1 = 0
                    self.count_time2 = 0
            # 화면에 보여주기
            cv2.imshow('cam', img)
            # 'q' 버튼을 누르면 반복문 종료
            if cv2.waitKey(1) == ord('q') or (self.successSig == True and self.successSig2 == True)\
            or self.timeResetSig == True:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
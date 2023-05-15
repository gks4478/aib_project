# kivy 라이브러리
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# 기타 라이브러리
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

fontName= '_rjz8SJzx9_rdfA7bCcae2U6c_A.ttf'

class CamApp(App):

    # 1. 앱 구성
    def build(self):
        # 카메라 화면 크기
        self.web_cam = Image(size_hint=(1,.8))
        # 상단 텍스트
        self.new_Label= Label(text= '얼굴인식으로 문을 여세요', font_name= fontName, size_hint= (1, .1), font_size= 20)
        self.new_Label.color = (1, 1, 1, 1)
        # 현재 상태 텍스트
        self.verification_label = Label(text="안녕하세요", font_name= fontName, size_hint=(1,.1), font_size= 20)
        self.verification_label.color = (1, 1, 1, 1)
        #버튼
        yellow_color = (1, 1, 0, 1)
        self.button = Button(text="검증 시작", font_name= fontName, on_press=self.verify, size_hint=(1,.1), background_color= yellow_color, font_size= 20) 
        self.button.color = (0.5, 0.5, 0.5, 1)

        # 카메라, 버튼, 텍스트 수직 배치
        layout = BoxLayout(orientation='vertical', padding= 10)
        # 상단 텍스트
        layout.add_widget(self.new_Label)
        # 카메라 화면
        layout.add_widget(self.web_cam)
        # 버튼 
        layout.add_widget(self.button)
        # 텍스트
        layout.add_widget(self.verification_label) 

        # siamesemodel.h5 불러오고 L1Dist 지정
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        # 카메라 연결/노트북 내장이므로 0
        self.capture = cv2.VideoCapture(0) 
        # 버튼을 누르고 사진이 업데이트 되는 시간
        Clock.schedule_interval(self.update, 1.0/33.0) 
        
        return layout

    # 2. 카메라 실시간 업데이트
    def update(self, *args):

        # 카메라 실행
        ret, frame = self.capture.read() 
        # 카메라 화면크기 설정(250*250)
        frame = frame[30:30+250, 250:250+250, :] 
        # frame 수직으로 뒤집고 바이트문자열로 변환: 프레임 데이터
        buf = cv2.flip(frame, 0).tostring()
        # brg(blue,red,green) 형식으로 텍스처(texture) 생성
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # img_texture에 buf를 복사해서 사진 재구성(각 픽셀 8비트)
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # 카메라에 할당(시실간으로 표시)
        self.web_cam.texture = img_texture

    # 3. 사진 정규화
    def preprocess(self, file_path):

        # file_path의 사진을 바이트로 읽어온다.
        byte_img = tf.io.read_file(file_path)
        # btye_img를 jpeg로 디코딩
        img = tf.io.decode_jpeg(byte_img)
        
        # 105*105*3 크기로 변경(3은 RGB)
        img = tf.image.resize(img, (105,105))
        # img를 0~1 값으로 정규화
        img = img / 255.0
        
        return img

    # 4. 일치 불일치 확인
    def verify(self, *args):
        # 임계값
        detection_threshold = 0.5
        verification_threshold = 0.48

        # 경로 지정: application_data/input_image/(파일 이름).jpg
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')

        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            # 해당 경로 이미지 정규화(입력)
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            # 해당 경로 이미지 정규화(검증)
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # [입력:검증] 모델(siamesemodel.h5)로 예측
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            # results에 result 저장
            results.append(result)

        # results(리스트)의 각 값이 detection_threshold(0.99)를 초과허면 True 아니면 Fasle로
        # 표시되는데 np.sum으로 True의 개수를 계산한다.
        detection = np.sum(np.array(results) > detection_threshold)
        
        # 현재 'verification_images'파일에는 50개의 사진이 있다.
        # 비율계산해서 True(일치)인지 False(불일치)인지 verified에 저장
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        # verified에 따른 텍스트 할당
        self.verification_label.text = '[열림] 환영합니다' if verified == True else '[안열림] 등록이 안 된 사용자입니다'

        # 표시
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        # results와 verified만 터미널창에 표시
        return results, verified

if __name__ == '__main__':
    CamApp().run()
# AgileXpiper_rl
agileX piper를 활용한 강화학습 기초 환경 세팅(no viewer)

chat gpt 를 활용하여 확장가능한 기초적인 강화학습 환경 구축

현재 코드 추가 및 버그 수정중

학습 기록용 코드입니다.

<개발 환경>

ubuntu 20.04 ros noetic

conda 24.5.0

Cuda version 12.6

isaac gym preview4

python 3.8.10

urdf -> https://github.com/agilexrobotics/piper_ros/tree/noetic/src/piper_description 참고

<Framework>
  
![framework](https://github.com/user-attachments/assets/02022e95-2a92-4112-816e-1b007736a9d8)

<Policy>
  
![policy](https://github.com/user-attachments/assets/8616f610-8b98-475e-940d-0391ea975e86)

---------------------------------------------------------------------------------------------------------------------------------------------------------

학습 정책 결과 저장 및 실행, 보상 점수

![studypolicy](https://github.com/user-attachments/assets/bbfee059-92fd-47f7-a9a7-f5ae8fbe9989)

![policysave](https://github.com/user-attachments/assets/2559d4c2-25ba-4515-80a5-f613853279c5)

![reward(play)](https://github.com/user-attachments/assets/7c9c022b-3118-4afe-acd1-b68fdb52cc9d)

아직 학습 action 부분은 미구현이며 이에따라 움직이지는 않는다.

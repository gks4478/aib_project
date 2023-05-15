# section2_project_preview

데이터 : https://www.kaggle.com/datasets/ruchibhadauria/hotel-reviews-from-tripadvisor

### 1. 관광지 검색, 숙소 예약, 일정 세우기를 한번에 할 수 있는 앱 구상
### 2. 앱 구상 안

> ![스크린샷 2023-04-12 174448](https://user-images.githubusercontent.com/77867734/231403544-9c78d796-38f7-4a00-9d22-65bc05563947.png)

### 3. 앱의 발전을 위해 사용자들의 피드백으로 수정해나간다.
>* 머신러닝 도입
>* 목표 : 리뷰로 타겟(positive와 negative)을 구분해낸다.
>* 전처리 → 인코딩 → 불균형 해소 → 기준 모델 선정 → xgb, svc 모델 훈련 → 최종 모델 svc

### 4. 보안 사항
>* 데이터들을 더 추가하여 모델의 성능을 높인다.
>* 현재는 모델이 분류한 리뷰들을 개발자들이 직접 읽어야 하는 번거로움이 있었다.
>* 리뷰에서 중요한 핵심만 간추려 개발자들의 번거로움과 시간을 줄일 수 있도록 한다.
>* 앱의 지역 범위를 국내에서 국외로 확장 시키다.

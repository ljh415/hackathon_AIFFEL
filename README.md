## AIFFEL_CV_TEAM3B



## EDA & Visualizing ([변호윤님](https://github.com/hybyun0121))

### 1.1 Data 구성

**ID** : test set에서 (Shop, Item) 묶음을 인덱싱

**shop_id** : 가게 번호
**item_id** : 제품 번호

**item_category_id** : 제품 카테고리 번호

**item_cnt_day** : 판매량

**item_price** : 제품 가격

**date** : date dd/mm/yyy

**date_block_num** : 시간별 월 인덱싱 Jan.2013 = 0 ~ Oct.2015 = 33

**item_name** : 제품 이름

**shop_name** : 가게 이름

**item_category_name** : 제품 카테고리 이름

**Target** : Nov. 2015 (34번째 달)에 특정 매장 특정 제품의 예상 판매량 예측

### EDA - 변호윤

어떤 영역에서 어떤 데이터를 가지고 어떤 문제를 해결해야 할때

먼저 목표(타겟)를 설정하고 타겟에 영향을 미치는 feature들은 어떤 것들이 있을지 고민하는 과정이 우선되어야 할 것이다. 뿐만아니라 주어진 feature들의 상관관계를 파악하고 새로운 feature를 뽑아 내는 것 역시 좋은 결과를 얻기위해 필요할 것이다.

데이터의 특징을 분석하고 feature를 뽑아내는 일은 해당 영역의 도메인 지식이 매우 중요할 것으로 보인다.

도메인 지식이 부족한 단순한 초보개발자 입장에서 EDA가 중요한 이유이기도 하다.

여러 사람들이 이번 캐글 문제를 고민하고 공유한 노트북들을 참고하면서 feature들의 관계를 어떻게 뽑아내고 특히 이번 데이터 만의 문제(누군가에겐 아닐 수도 있는)인 언어적인 부분이라던지, 또는 데이터의 량이 방대하고 타겟에 데이터가 주어지지 않고 우리가 가지고 있는 데이터로 가공(일별 판매량을 한달 단위로 합쳐야한다)해서 만들어야하는 등 기존에 접했던 데이터와 다른 점들이 많아 힘든 부분이 있었다.

구체적인 예로 들자면 타겟 데이터를 만들기 위해서 shop_id와 item_id를 묶은 다음 매달 총 판매량을 계산해야한다. 그리고 나서 시간순으로 정렬하여 최종적으로 34번째 달의 예상 판매량을 예측해야하는 것이다.

데이터를 잘 정제하고 적절한 모델을 선택하여 학습하는 노하우를 더 많이 쌓기 위해 많은 경험이 필요해 보인다.

---

---





## 전처리 및 Feature Engineering, 모델 학습

- 데이터 전처리

  - 이상치 탐색
    - ```item_cnt_day``` : 1,000 이상의 값 삭제
    - ```item_price``` : 100,000 이상의 값 삭제
  - ```shop_name``` 전처리
    - 러시아어로 작성되어 있다.
    - ```shop_name```을 살펴보면 같은 이름의 가게가 존재 => id값을 수정
    - ```shop_name```으로 부터 두 가지 정보 추출
      1. ```city```
      2. ```category```
    - ```LabelEncoder``` 활용 : categorical 정보를 numeric으로 변환한다.
  - ```item_categories``` 전처리
    - ```item_category_name```에서 물품의 ```type_code``` 추출
    - ```LabelEncoder```활용
  - ```items``` 전처리
    - 정규표현식을 활용
    - ```LabelEncoder```사용

  

- Feature Engineering

  - ```revenue```  생성
  - ```item_cnt_month``` 생성
  -  ```date_avg_item_cnt``` 생성
  - ```date_item_avg_item_cnt``` 생성
  - ```date_shop_avg_item_cnt``` 생성
  - ```date_shop_item_avg_item_cnt``` 생성
  - ```date_shop_subtype_avg_item_cnt``` 생성
  - ```date_city_avg_item_cnt``` 생성
  - ```date_item_city_avg_item_cnt``` 생성
  - ```delta_price_lag``` 생성
  - ```delta_revenue_lag_1``` 생성
    - ```date_block_num```, shop_id``` -> ```revenue```의 합
        - ```date_shop_revenue``` 생성
    - ```shop_id``` -> ```date_block_num```의 평균
        - ```shop_avg_revenue``` 생성
    - ```date_shop_revenue```, ```shop_avg_revenue``` 활용
        - ```delta_revenue``` 생성
    - ```delta_revenue``` + ```lag_feature``` (lag=1)
        - ```delta_revenue_lag_1``` 생성
  - ```month``` 생성
  - ```item_shop_first_sale```, ```item_first_sale``` 생성

- 모델 생성 및 학습

  - 사용모델

    1. XGBoost
    2. LGBM
    3. LGBM + ensemble

  - 사용 파라미터

    - XGBoost : 0.89775

      ```python
      model = XGBRegressor(
          max_depth=13,
          n_estimators=900,
          min_child_weight=0.5, 
          colsample_bytree=0.7, 
          subsample=0.8, 
          eta=0.1,
      #     tree_method='gpu_hist',
          seed=42)
      
      ```

    - LGBM : 0.88153

      ```python
      model = LGBMRegressor(
          learning_rate = 0.08,
          num_leaves= 4000,
          n_estimators=300,        # no
          max_depth=-1,
          min_child_weight= 3,     # no
          subsample= 0.8,
          colsample_bytree= 0.4,
          n_jobs= -1,
          reg_lambda=0.8
          # lambda = 0.7
      )
      ```

    - LGBM + ensemble (10회) : 0.87036

      ```python
      param = {
          'learning_rate' : 0.08,
          'num_leaves': 4000,
          'n_estimators':300,        # no
          'max_depth':-1,
          'min_child_weight': 3,     # no
          'subsample': 0.8,
          'colsample_bytree': 0.4,
          'n_jobs': -1,
      #     reg_lambda=0.8
          'lambda' : 0.8
      }
      ```

---


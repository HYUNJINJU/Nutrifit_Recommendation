import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import json

# 데이터 로드
file_path = '전국통합식품영양성분정보_음식_표준데이터.csv'
data = pd.read_csv(file_path, encoding='cp949')

# 필요한 열 선택
selected_columns = [
    "대표식품명", 
    "에너지(kcal)", 
    "단백질(g)", 
    "지방(g)", 
    "탄수화물(g)", 
    "당류(g)", 
    "식이섬유(g)", 
    "칼슘(mg)", 
    "철(mg)", 
    "나트륨(mg)", 
    "비타민 A(μg RAE)", 
    "비타민 C(mg)",
    "식품대분류명"
]
nutrition_data = data[selected_columns].dropna()

# 대표식품명 기준으로 평균값 처리
nutrition_data_grouped = nutrition_data.groupby(["대표식품명", "식품대분류명"], as_index=False).mean()

# 사용자 정보 로드 및 일일 섭취 권장량 계산
def calculate_daily_intake(user_data):
    user = user_data['user'][0]
    weight = user['weight']
    height = user['height']
    age = user['age']
    gender = user['gender']
    activity = int(user['activity'])

    print(f"height={height}, weight={weight}, age={age}, gender={gender}, activity={activity}")

    # 기초대사량 계산
    if gender == "1":  # 남성
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # 여성
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # 활동 수준에 따른 총 에너지 요구량 계산
    if activity == 1: # 활동량 매우낮음
        tdee = bmr * 1.0
    elif activity == 2:  # 활동량 낮음
        tdee = bmr * 1.2
    elif activity == 3:  # 활동량 보통
        tdee = bmr * 1.4
    elif activity == 4:  # 활동량 높음
        tdee = bmr * 1.6
    elif activity == 5:  # 활동량 매우높음
        tdee = bmr * 1.8
    else:
        print(f"Error: Invalid activity level: {activity}")
        raise ValueError(f"Invalid activity level: {activity}.")
    
    # 권장 섭취량 계산
    daily_intake = {
        "에너지(kcal)": tdee,
        "단백질(g)": 0.8 * weight, 
        "지방(g)": tdee * 0.3 / 9,
        "탄수화물(g)": tdee * 0.5 / 4,
        "당류(g)": tdee * 0.1 / 4,
        "식이섬유(g)": 0,
        "칼슘(mg)": 850,
        "철(mg)": 0,
        "나트륨(mg)": 2000,
        "비타민 A(μg RAE)": 0,
        "비타민 C(mg)": 100,
    }
        
    if gender == "1":
        daily_intake["식이섬유(g)"] = 38
    else:
        daily_intake["식이섬유(g)"] = 25
            
    if gender == "1":
        daily_intake["철(mg)"] = 10
    else:
        daily_intake["철(mg)"] = 18
            
    if gender == "1":
        daily_intake["비타민 A(μg RAE)"] = 900
    else:
        daily_intake["비타민 A(μg RAE)"] = 700

    return daily_intake

# 사용자 데이터 로드
with open('user_data.json', 'r', encoding='utf-8') as file:
    user_data = json.load(file)

daily_intake_recommendations = calculate_daily_intake(user_data)

# 사용자 일일 섭취 권장량 출력
print("\n사용자 일일 섭취 권장량:")
for nutrient, value in daily_intake_recommendations.items():
    print(f"{nutrient}: {value:.2f}")

# json 파일에서 섭취 데이터 로드
with open('food_data.json', 'r', encoding='utf-8') as file:
    consumed_data_json = json.load(file)

# 섭취한 음식 데이터 추출
consumed_foods = consumed_data_json['foods']

# 섭취한 음식의 영양소 계산 및 사용자 권장량 업데이트
for food in consumed_foods:
    nutrition = food['nutrition']
    eatAmount = food['eatAmount']
    daily_intake_recommendations['에너지(kcal)'] -= (nutrition['calories'] * eatAmount)
    daily_intake_recommendations['단백질(g)'] -= (nutrition['protein'] * eatAmount)
    daily_intake_recommendations['지방(g)'] -= (nutrition['fat'] * eatAmount)
    daily_intake_recommendations['탄수화물(g)'] -= (nutrition['carbonhydrate'] * eatAmount)
    daily_intake_recommendations['당류(g)'] -= (nutrition['sugar'] * eatAmount)
    daily_intake_recommendations['식이섬유(g)'] -= (nutrition['dietrayfiber'] * eatAmount)
    daily_intake_recommendations['칼슘(mg)'] -= (nutrition['calcium'] * eatAmount)
    daily_intake_recommendations['철(mg)'] -= (nutrition.get('iron', 0) * eatAmount)
    daily_intake_recommendations['나트륨(mg)'] -= (nutrition['sodium'] * eatAmount)
    daily_intake_recommendations['비타민 A(μg RAE)'] -= (nutrition['vitamina'] * eatAmount)
    daily_intake_recommendations['비타민 C(mg)'] -= (nutrition['vitaminc'] * eatAmount)

# 부족한 영양소 계산 함수
def calculate_deficiencies_and_update_recommendations(recommendations):
    deficiencies = {nutrient: max(0, value) for nutrient, value in recommendations.items()}
    return deficiencies, recommendations

nutrition_data_grouped = nutrition_data_grouped[~nutrition_data_grouped["대표식품명"].str.contains("개고기|미꾸라지")]

# 음식 추천 함수
def recommend_foods(deficiencies, nutrition_data, category1, category2, category3, taste1, taste2):
    recommendations = pd.DataFrame()

    # category1에서 하나 추천
    primary_data = nutrition_data[nutrition_data["식품대분류명"] == category1]
    specific_foods = [
        "쌀밥", "연잎밥", "오곡밥", "자장밥", "잡탕밥", "기장밥", "보리밥", "비빔밥", "수수밥", "차조밥", 
        "현미밥", "흑미밥", "율무밥", "볶음밥", "돌솥밥", "곤드레", "주먹밥", "김밥", "귀리밥"
    ]

    is_rice = False

    # 가중치 정의
    nutrient_importance = {
        "에너지(kcal)": 1.0,
        "단백질(g)": 1.5,
        "지방(g)": 1.0,
        "탄수화물(g)": 1.2,
        "당류(g)": 0.8,
        "식이섬유(g)": 1.5,
        "칼슘(mg)": 1.3,
        "철(mg)": 1.4,
        "나트륨(mg)": 0.7,
        "비타민 A(μg RAE)": 1.0,
        "비타민 C(mg)": 1.2
    }

    # 가중치 반영 결핍 벡터 계산 (음수는 추가 가중치)
    weighted_deficiency_vector = np.array([
        (deficiencies.get(col, 0) * nutrient_importance.get(col, 1.0)) * (2 if deficiencies.get(col, 0) < 0 else 1)
        for col in selected_columns[1:-1]
    ])

    if not primary_data.empty:
        nutrition_features = primary_data.drop(columns=["대표식품명", "식품대분류명"]).values
        scores = cosine_similarity(nutrition_features, weighted_deficiency_vector.reshape(1, -1)).flatten()
        primary_data = primary_data.copy()
        primary_data["추천점수"] = scores
        top_primary_recommendation = primary_data.sort_values(by="추천점수", ascending=False).head(1)

        # category1에서 추천된 음식 이름에 specific_foods 중 하나라도 포함되어 있는지 확인
        recommended_food_name = top_primary_recommendation.iloc[0]["대표식품명"]
        if any(food in recommended_food_name for food in specific_foods):
            is_rice = True

        recommendations = pd.concat([recommendations, top_primary_recommendation], ignore_index=True)

    # category2 에서 하나 추천 (taste1 반영)
    if is_rice:
        for category in category2:
            second_data = nutrition_data[nutrition_data["식품대분류명"] == category]
            if not second_data.empty:
                nutrition_features = second_data.drop(columns=["대표식품명", "식품대분류명"]).values
                scores = cosine_similarity(nutrition_features, weighted_deficiency_vector.reshape(1, -1)).flatten()

                # taste1 가중치 반영
                if taste1 == 1 and category == "찌개 및 전골류":
                    scores *= 1.1
                elif taste1 == 2 and category == "국 및 탕류":
                    scores *= 1.1

                second_data = second_data.copy()
                second_data["추천점수"] = scores
                top_second_recommendation = second_data.sort_values(by="추천점수", ascending=False).head(1)
                recommendations = pd.concat([recommendations, top_second_recommendation], ignore_index=True)
                break

    # category3, category4, category5에서 각각 하나씩 추천
    third_recommendations = []
    for category_list in [category3, category4, category5]:
        for category in category_list:
            if len(third_recommendations) >= len([category3, category4, category5]):  # 각 그룹에서 하나씩 추천되면 종료
                break
            third_data = nutrition_data[nutrition_data["식품대분류명"] == category]
            if not third_data.empty:
                nutrition_features = third_data.drop(columns=["대표식품명", "식품대분류명"]).values
                scores = cosine_similarity(nutrition_features, weighted_deficiency_vector.reshape(1, -1)).flatten()

                # taste2 가중치 반영
                if taste2 == 11 and category == "튀김류":
                    scores *= 1.1
                elif taste2 == 12 and category == "구이류":
                    scores *= 1.1
                elif taste2 == 13 and category == "볶음류":
                    scores *= 1.1
                elif taste2 == 14 and category == "찜류":
                    scores *= 1.1
                elif taste2 == 15 and category == "전·적 및 부침류":
                    scores *= 1.1
                elif taste2 == 16 and category == "조림류":
                    scores *= 1.1

                third_data = third_data.copy()
                third_data["추천점수"] = scores
                # 각 카테고리에서 상위 하나만 추천
                top_third_recommendation = third_data.sort_values(by="추천점수", ascending=False).head(1)
                third_recommendations.append(top_third_recommendation)
                break  # 각 그룹에서 하나만 선택하므로 반복 종료

    # 추천 결과를 DataFrame에 추가
    if third_recommendations:
        third_recommendations_df = pd.concat(third_recommendations, ignore_index=True)
        recommendations = pd.concat([recommendations, third_recommendations_df], ignore_index=True)

    return recommendations

# 사용자 음식 분류 선호도 데이터 로드
taste1 = user_data['user'][0]['taste1']
taste2 = user_data['user'][0]['taste2']

# 부족한 영양소와 업데이트된 권장량 계산
deficiencies_example, updated_recommendations = calculate_deficiencies_and_update_recommendations(daily_intake_recommendations)

# 음식 분류 정의
category1 = "밥류"
category2 = ["찌개 및 전골류", "국 및 탕류"]
category3 = ["튀김류", "구이류", "볶음류", "찜류", "전·적 및 부침류", "조림류"]
category4 = ["생채·무침류", "나물·숙채류"]
category5 = ["김치류", "장아찌·절임류", "젓갈류"]

# 음식 추천
deficiencies_example, updated_recommendations = calculate_deficiencies_and_update_recommendations(daily_intake_recommendations)
recommended_foods_specific = recommend_foods(
    deficiencies_example, nutrition_data_grouped, category1, category2, category3, taste1, taste2
)
# 추천된 음식의 영양소 소수점 세 자리 반올림
recommended_foods_specific[selected_columns[1:]] = recommended_foods_specific[selected_columns[1:]].round(2)

# 결과 출력
print("\n업데이트된 사용자 일일 섭취 권장량:")
for nutrient, value in updated_recommendations.items():
    print(f"{nutrient}: {value:.2f}")

print("\n추천된 음식과 영양성분:")
print(tabulate(
    recommended_foods_specific[["대표식품명", "추천점수"] + selected_columns[1:]], 
    headers="keys", 
    tablefmt="pretty"))

# 추천 결과를 json 형식으로 저장
output_file = 'recommended_foods.json'
recommended_foods_json = recommended_foods_specific[["대표식품명"] + selected_columns[1:]].to_dict(orient='records')
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(recommended_foods_json, json_file, ensure_ascii=False, indent=4)

print(f"\n추천 결과가 '{output_file}' 파일에 저장되었습니다.")
from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# CSV 데이터 로드
file_path = '전국통합식품영양성분정보_음식_표준데이터.csv'
data = pd.read_csv(file_path, encoding='cp949')

selected_columns = [
    "대표식품명", "에너지(kcal)", "단백질(g)", "지방(g)", "탄수화물(g)", "당류(g)", "식이섬유(g)",
    "칼슘(mg)", "철(mg)", "나트륨(mg)", "비타민 A(\u03bcg RAE)", "비타민 C(mg)", "식품대분류명"
]
nutrition_data = data[selected_columns].dropna()
nutrition_data_grouped = nutrition_data.groupby(["대표식품명", "식품대분류명"], as_index=False).mean()
nutrition_data_grouped = nutrition_data_grouped[~nutrition_data_grouped["대표식품명"].str.contains("개고기|미꾸라지")]

# 권장 섭취량 계산
def calculate_daily_intake(user):
    weight = user['weight']
    height = user['height']
    age = user['age']
    gender = user['gender']
    activity = int(user['activity'])

    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "1" else -161)
    tdee = bmr * (1.0 + 0.2 * (activity - 1))

    return {
        "에너지(kcal)": tdee,
        "단백질(g)": 0.8 * weight,
        "지방(g)": tdee * 0.3 / 9,
        "탄수화물(g)": tdee * 0.5 / 4,
        "당류(g)": tdee * 0.1 / 4,
        "식이섬유(g)": 38 if gender == "1" else 25,
        "칼슘(mg)": 850,
        "철(mg)": 10 if gender == "1" else 18,
        "나트륨(mg)": 2000,
        "비타민 A(\u03bcg RAE)": 900 if gender == "1" else 700,
        "비타민 C(mg)": 100
    }

# 추천 함수
def recommend_foods(deficiencies, taste1, taste2):
    recommendations = pd.DataFrame()
    categories = {
        "category1": "밥류",
        "category2": ["찌개 및 전골류", "국 및 탕류"],
        "category3": ["튀김류", "구이류", "볶음류", "찜류", "전·적 및 부침류", "조림류"],
        "category4": ["생채·무침류", "나물·숙채류"],
        "category5": ["김치류", "장아찌·절임류", "젓갈류"]
    }

    nutrient_importance = {
        "에너지(kcal)": 1.0, "단백질(g)": 1.5, "지방(g)": 1.0, "탄수화물(g)": 1.2,
        "당류(g)": 0.8, "식이섬유(g)": 1.5, "칼슘(mg)": 1.3, "철(mg)": 1.4,
        "나트륨(mg)": 0.7, "비타민 A(\u03bcg RAE)": 1.0, "비타민 C(mg)": 1.2
    }

    weighted_vector = np.array([
        (deficiencies.get(col, 0) * nutrient_importance.get(col, 1.0)) * (2 if deficiencies.get(col, 0) < 0 else 1)
        for col in selected_columns[1:-1]
    ])

    def select_top(df, boost=1.0):
        features = df.drop(columns=["대표식품명", "식품대분류명"]).values
        scores = cosine_similarity(features, weighted_vector.reshape(1, -1)).flatten() * boost
        df = df.copy()
        df["추천점수"] = scores
        return df.sort_values(by="추천점수", ascending=False).head(1)

    # category1: 밥류 1개 추천
    df1 = nutrition_data_grouped[nutrition_data_grouped["식품대분류명"] == categories["category1"]]
    if not df1.empty:
        top1 = select_top(df1)
        recommendations = pd.concat([recommendations, top1], ignore_index=True)
        is_rice = any(x in top1.iloc[0]["대표식품명"] for x in ["밥", "비빔밥", "김밥", "주먹밥"])
    else:
        is_rice = False

    # category2: 찌개/국 중 taste1에 따라 1개 추천
    if is_rice:
        for cat in categories["category2"]:
            df2 = nutrition_data_grouped[nutrition_data_grouped["식품대분류명"] == cat]
            if not df2.empty:
                boost = 1.1 if (taste1 == 1 and cat == "찌개 및 전골류") or (taste1 == 2 and cat == "국 및 탕류") else 1.0
                top2 = select_top(df2, boost)
                recommendations = pd.concat([recommendations, top2], ignore_index=True)
                break

    # category3~5: 각각 1개씩 taste2 반영하여 추천
    for cat_list in ["category3", "category4", "category5"]:
        for cat in categories[cat_list]:
            df = nutrition_data_grouped[nutrition_data_grouped["식품대분류명"] == cat]
            if not df.empty:
                boost = 1.1 if (
                    (taste2 == 11 and cat == "튀김류") or (taste2 == 12 and cat == "구이류") or
                    (taste2 == 13 and cat == "볶음류") or (taste2 == 14 and cat == "찜류") or
                    (taste2 == 15 and cat == "전·적 및 부침류") or (taste2 == 16 and cat == "조림류")
                ) else 1.0
                top = select_top(df, boost)
                recommendations = pd.concat([recommendations, top], ignore_index=True)
                break

    return recommendations[["대표식품명"] + selected_columns[1:] + ["추천점수"]].round(2)

# 추천 API
@app.route('/recommend', methods=['GET', 'POST'])
def recommend_html():
    if request.method == 'GET':
        return '''
        <h2>추천 시스템 서버</h2>
        <p>이 서버는 POST 방식으로 사용자 데이터를 받아 음식 추천 결과를 HTML로 반환합니다.</p>
        <p>Postman 또는 앱에서 JSON 데이터를 전송해 주세요.</p>
        '''
        
    try:
        input_data = request.get_json()
        user = input_data['user'][0]
        foods = input_data['foods']

        daily = calculate_daily_intake(user)

        for food in foods:
            n = food['nutrition']
            ea = food['eatAmount']
            daily['에너지(kcal)'] -= n['calories'] * ea
            daily['단백질(g)'] -= n['protein'] * ea
            daily['지방(g)'] -= n['fat'] * ea
            daily['탄수화물(g)'] -= n['carbonhydrate'] * ea
            daily['당류(g)'] -= n['sugar'] * ea
            daily['식이섬유(g)'] -= n['dietrayfiber'] * ea
            daily['칼슘(mg)'] -= n['calcium'] * ea
            daily['철(mg)'] -= n.get('iron', 0) * ea
            daily['나트륨(mg)'] -= n['sodium'] * ea
            daily['비타민 A(\u03bcg RAE)'] -= n['vitamina'] * ea
            daily['비타민 C(mg)'] -= n['vitaminc'] * ea

        deficiencies = {k: max(0, v) for k, v in daily.items()}
        result_df = recommend_foods(deficiencies, user['taste1'], user['taste2'])
        result = result_df.to_dict(orient='records')

        html = '<h2>추천 음식 결과</h2><table border="1" cellpadding="5" cellspacing="0">'
        html += '<tr>' + ''.join(f'<th>{col}</th>' for col in result[0].keys()) + '</tr>'
        for row in result:
            html += '<tr>' + ''.join(f'<td>{val}</td>' for val in row.values()) + '</tr>'
        html += '</table>'

        return html

    except Exception as e:
        return f"<h3>에러 발생: {str(e)}</h3>", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

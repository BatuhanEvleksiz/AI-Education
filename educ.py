import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats


np.random.seed(42)
teacher_experience = np.random.randint(1, 30, 100)  
ai_tool_usage = np.random.randint(1, 10, 100)       
student_success = ai_tool_usage * 5 + teacher_experience + np.random.normal(0, 5, 100) 

data = pd.DataFrame({
    'Teacher_Experience_Years': teacher_experience,
    'AI_Tool_Usage_Level': ai_tool_usage,
    'Student_Success_Score': student_success
})


print("Veri Setinin İlk Satırları:")
print(data.head())

print("\nVerinin Genel Bilgisi:")
print(data.info())

print("\nEksik Veri Kontrolü:")
print(data.isnull().sum())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='AI_Tool_Usage_Level', y='Student_Success_Score', data=data)
plt.title('AI Araç Kullanım Seviyesi ile Öğrenci Başarısı Arasındaki İlişki')
plt.xlabel('AI Kullanım Seviyesi')
plt.ylabel('Öğrenci Başarısı')
plt.show()

X = data[['AI_Tool_Usage_Level']]
y = data['Student_Success_Score']

model = LinearRegression()
model.fit(X, y)

print(f"Model Katsayısı: {model.coef_[0]}")
print(f"Model Kesişimi: {model.intercept_}")

plt.figure(figsize=(10, 6))
sns.regplot(x='AI_Tool_Usage_Level', y='Student_Success_Score', data=data, line_kws={'color': 'red'})
plt.title('AI Kullanım Seviyesine Göre Regresyon Çizgisi')
plt.show()

correlation, p_value = stats.pearsonr(data['AI_Tool_Usage_Level'], data['Student_Success_Score'])
print(f"Korelasyon Katsayısı: {correlation}")
print(f"P-değeri: {p_value}")

if p_value < 0.05:
    print("Sonuç: AI araçlarının kullanımı ile öğrenci başarısı arasında anlamlı bir ilişki var.")
else:
    print("Sonuç: AI araçlarının kullanımı ile öğrenci başarısı arasında anlamlı bir ilişki yok.")

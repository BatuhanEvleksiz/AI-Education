import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

# 1. Örnek Veri Seti Oluşturma
np.random.seed(42)
teacher_experience = np.random.randint(1, 30, 100)  # 1-30 yıl arası öğretmenlik deneyimi
ai_tool_usage = np.random.randint(1, 10, 100)       # 1-10 arasında AI araç kullanım seviyesi
student_success = ai_tool_usage * 5 + teacher_experience + np.random.normal(0, 5, 100)  # Basit bir formül

# Veri setini bir DataFrame'e dönüştürme
data = pd.DataFrame({
    'Teacher_Experience_Years': teacher_experience,
    'AI_Tool_Usage_Level': ai_tool_usage,
    'Student_Success_Score': student_success
})

# 2. Veri Keşfi ve Ön İşleme
print("Veri Setinin İlk Satırları:")
print(data.head())

print("\nVerinin Genel Bilgisi:")
print(data.info())

print("\nEksik Veri Kontrolü:")
print(data.isnull().sum())

# 3. Veriyi Görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AI_Tool_Usage_Level', y='Student_Success_Score', data=data)
plt.title('AI Araç Kullanım Seviyesi ile Öğrenci Başarısı Arasındaki İlişki')
plt.xlabel('AI Kullanım Seviyesi')
plt.ylabel('Öğrenci Başarısı')
plt.show()

# 4. Modelleme ve Regresyon Analizi
# Bağımsız değişken (X) ve bağımlı değişken (y) tanımlama
X = data[['AI_Tool_Usage_Level']]
y = data['Student_Success_Score']

# Lineer regresyon modeli oluşturma
model = LinearRegression()
model.fit(X, y)

# Modelin katsayısını ve kesişimini görüntüleme
print(f"Model Katsayısı: {model.coef_[0]}")
print(f"Model Kesişimi: {model.intercept_}")

# Regresyon doğrusunu çizme
plt.figure(figsize=(10, 6))
sns.regplot(x='AI_Tool_Usage_Level', y='Student_Success_Score', data=data, line_kws={'color': 'red'})
plt.title('AI Kullanım Seviyesine Göre Regresyon Çizgisi')
plt.show()

# 5. Hipotez Testi
correlation, p_value = stats.pearsonr(data['AI_Tool_Usage_Level'], data['Student_Success_Score'])
print(f"Korelasyon Katsayısı: {correlation}")
print(f"P-değeri: {p_value}")

if p_value < 0.05:
    print("Sonuç: AI araçlarının kullanımı ile öğrenci başarısı arasında anlamlı bir ilişki var.")
else:
    print("Sonuç: AI araçlarının kullanımı ile öğrenci başarısı arasında anlamlı bir ilişki yok.")

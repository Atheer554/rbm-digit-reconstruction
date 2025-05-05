
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler


# تحميل بيانات الأرقام من 0 الى 9
digits = load_digits()
data = digits.data
labels = digits.target


data_filtered=data
labels_filtered=labels

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_filtered)


# إنشاء نموذج RBM
# ننشئ نموذج RBM بعدد 300 مكون خفي، بمعدل تعلم 0.01، ونتدرب عليه لمدة 500 تكرار
rbm = BernoulliRBM(n_components=300, learning_rate=0.01, n_iter=500, random_state=42)
#n_components=300: يعطي تمثيل داخلي أعمق
##n_iter=500: يزيد عدد مرات التعلم
#learning_rate=0.01: يجعل التحديثات أهدأ ادق


#عشان يعلمني انه بداء 
print("training started....")

rbm.fit(data_scaled)
#عشان اعرف انه خلص
print("trainig finished.")

# نحول البيانات إلى الطبقة المخفية
hidden = rbm.transform(data_scaled)

# نعيد بناء البيانات الأصلية (من الطبقة المخفية)
reconstructed = 1 / (1 + np.exp(-(np.dot(hidden, rbm.components_) + rbm.intercept_visible_)))


# نحدد كم صورة نبي نعرض
n_images = 10
shown_digits = []

for digit in range(n_images):
    # نختار أول صورة تمثل الرقم المطلوب
    index = np.where(labels_filtered == digit)[0][0]
    shown_digits.append(index)

for i, idx in enumerate(shown_digits):
    plt.subplot(2, n_images, i + 1)
    plt.imshow(data_scaled[idx].reshape(8, 8), cmap='gray')
    plt.axis('off')
    plt.title(f"Original: {labels_filtered[idx]}")

    plt.subplot(2, n_images, i + 1 + n_images)
    plt.imshow(reconstructed[idx].reshape(8, 8), cmap='gray')
    plt.axis('off')
    plt.title("Reconstructed")

plt.tight_layout()
plt.show()




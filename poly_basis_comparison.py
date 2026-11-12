import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.hermite import Hermite

# ============================================================
# ۱. تولید دادهٔ ساختگی (می‌توانید دادهٔ واقعی خود را جایگزین کنید)
# ============================================================
np.random.seed(42)
x = np.linspace(-2, 2, 40)
# تابع واقعی (مثلاً یک تابع غیرخطی + نویز)
y_true = 0.8*x**3 - 1.5*x**2 + 0.3*x + 2 + 0.4*np.sin(3*x)
noise = 0.25 * np.random.randn(len(x))
y = y_true + noise

# ============================================================
# ۲. برازش چندجمله‌ای معمولی با poly1d (درجه ۳)
# ============================================================
deg = 3
coeffs = np.polyfit(x, y, deg)          # ضرایب از درجه بالا به پایین
p = np.poly1d(coeffs)                   # شیء poly1d
y_poly = p(x)

# ============================================================
# ۳. برازش با پایه‌های Chebyshev و Hermite
# ============================================================
# Chebyshev (دامنه را به [-1,1] نگاشت می‌کنیم)
x_cheb = (x - x.min()) / (x.max() - x.min()) * 2 - 1   # نگاشت به [-1,1]
cheb = Chebyshev.fit(x_cheb, y, deg)
y_cheb = cheb(x_cheb)

# Hermite
herm = Hermite.fit(x, y, deg)
y_herm = herm(x)

# ============================================================
# ۴. حل سیستم خطی با ماتریس ون‌درموند (روش مستقیم)
# ============================================================
# ساخت ماتریس ون‌درموند (درجه deg)
V = np.vander(x, deg+1)                 # ستون‌ها: x^deg ... x^0
# حل سیستم V @ c = y
c_lin, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
y_lin = V @ c_lin

# نمایش شکل‌ها با reshape / flatten (فقط برای تمرین)
print("شکل ماتریس ون‌درموند:", V.shape)
print("flatten شده (۵ عنصر اول):", V.flatten()[:5])
print("reshape به (۸,۲۰):", V.reshape(8, 20).shape)

# ============================================================
# ۵. محاسبهٔ خطاها
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

errors = {
    "poly1d": rmse(y, y_poly),
    "Chebyshev": rmse(y, y_cheb),
    "Hermite": rmse(y, y_herm),
    "Linear (Vandermonde)": rmse(y, y_lin)
}

print("\n===== خطای RMSE =====")
for name, err in errors.items():
    print(f"{name:20s}: {np.round(err, 4)}")

# ============================================================
# ۶. نمودار مقایسه‌ای
# ============================================================
plt.figure(figsize=(12, 7))

# دادهٔ اصلی
plt.scatter(x, y, c='k', s=40, alpha=0.7, label='دادهٔ نویزی', zorder=5)
plt.plot(x, y_true, 'k--', lw=1.5, alpha=0.6, label='تابع واقعی')

# تقریب‌ها
plt.plot(x, y_poly, 'r-', lw=2, label=f'poly1d (درجه {deg})')
plt.plot(x, y_cheb, 'b-', lw=2, label=f'Chebyshev (درجه {deg})')
plt.plot(x, y_herm, 'g-', lw=2, label=f'Hermite (درجه {deg})')
plt.plot(x, y_lin, 'm--', lw=2, label='حل سیستم خطی (Vandermonde)')

plt.title('مقایسهٔ روش‌های تقریب چندجمله‌ای', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ۷. نمودار خطا (اختیاری)
# ============================================================
plt.figure(figsize=(10, 4))
methods = list(errors.keys())
vals = list(errors.values())
bars = plt.bar(methods, vals, color=['red','blue','green','magenta'], alpha=0.7)
plt.ylabel('RMSE')
plt.title('مقایسهٔ خطای تقریب‌ها')
plt.xticks(rotation=15)
for bar, val in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{np.round(val,4)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
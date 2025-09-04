# app.py
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Utils ---
def to_dt(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

def rmse_safe(y_true, y_pred):
    # sklearn >=1.4 рекомендует root_mean_squared_error
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        diff = np.asarray(y_true) - np.asarray(y_pred)
        return float(np.sqrt(np.mean(diff**2)))

def mae_safe(y_true, y_pred):
    try:
        from sklearn.metrics import mean_absolute_error
        return float(mean_absolute_error(y_true, y_pred))
    except Exception:
        diff = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        return float(np.mean(diff))

def seasonal_strength(y, seasonal, resid):
    # Rob Hyndman style seasonal strength metric ∈ [0,1]
    var = np.nanvar(y - np.nanmean(y))
    if var <= 0 or np.isnan(var):
        return 0.0
    return float(1 - np.nanvar(resid) / np.nanvar(resid + seasonal))

# --- App ---
st.set_page_config(page_title="📈 Time Series Forecast App", page_icon="📈", layout="wide")
st.title("📈 Time Series Forecast")
st.caption("Загрузите свой CSV со временным рядом и получите прогноз, метрики и краткую интерпретацию.")

with st.sidebar:
    st.header("1) Загрузка данных")
    up = st.file_uploader("CSV-файл", type=["csv"])
    st.markdown("**Формат:** одна серия на листе. Можно с заголовками.\n"
                "Если колонок несколько — выберете нужные ниже.")
    st.header("2) Настройки модели")
    freq = st.selectbox("Частота данных", ["D", "W", "MS", "M", "QS", "Q"], index=1,
                        help="D=дни, W=недели, MS=мес (начало), M=мес (конец), QS=квартал (начало), Q=квартал (конец)")
    season_map = {"D": 7, "W": 52, "MS": 12, "M": 12, "QS": 4, "Q": 4}
    s_default = season_map.get(freq, 12)
    s = st.number_input("Сезонный период (s)", min_value=0, max_value=366, value=int(s_default), step=1,
                        help="0 = без сезонности. Для недель берите ~52, для месяцев ~12, для кварталов ~4.")
    st.markdown("---")
    st.subheader("SARIMA параметры")
    p = st.number_input("p (AR)", 0, 5, 1)
    d = st.number_input("d (diff)", 0, 2, 1)
    q = st.number_input("q (MA)", 0, 5, 1)
    P = st.number_input("P (seasonal AR)", 0, 5, 1 if s else 0)
    D = st.number_input("D (seasonal diff)", 0, 2, 0 if s else 0)
    Q = st.number_input("Q (seasonal MA)", 0, 5, 1 if s else 0)
    use_const = st.checkbox("Тренд (константа)", value=True)
    h = st.number_input("Горизонт прогноза (периодов)", 1, 52, 12)

if up is None:
    st.info("Загрузите CSV, чтобы продолжить.")
    st.stop()

# --- Read CSV ---
raw = pd.read_csv(up)
st.subheader("Предпросмотр данных")
st.dataframe(raw.head(10), use_container_width=True)

# Guess columns
cols = list(raw.columns)
with st.expander("Выбор столбцов"):
    date_col = st.selectbox("Колонка даты/времени", cols, index=0)
    value_col = st.selectbox("Колонка значений (y)", cols, index=min(1, len(cols)-1))
    keep_only_positive = st.checkbox("Заменять отрицательные значения на 0 (для продаж/заявок)", value=False)

# Build series
df = raw[[date_col, value_col]].copy()
df[date_col] = to_dt(df[date_col])
df = df.dropna(subset=[date_col]).sort_values(date_col)
df = df.rename(columns={date_col: "ds", value_col: "y"})

if keep_only_positive:
    df["y"] = df["y"].clip(lower=0)

# Resample to selected freq
ser = (
    df.set_index("ds")["y"]
      .sort_index()
      .resample(freq)
      .sum()       # если нужно среднее — поменяй на .mean()
      .astype(float)
)

st.subheader("Сформированный ряд")
col1, col2 = st.columns([2,1])
with col1:
    fig, ax = plt.subplots()
    ser.plot(ax=ax)
    ax.set_title(f"Временной ряд ({freq})")
    ax.set_xlabel("Date"); ax.set_ylabel("Value")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.write("Период:", f"{ser.index.min().date()} → {ser.index.max().date()}")
    st.write("Точек:", len(ser))
    st.write("Пропусков:", int(ser.isna().sum()))
    st.write("Среднее:", f"{ser.mean():.2f}")

ser = ser.sort_index().dropna()

# Сплит 
# если пользователь задал слишком большой h — ужмём его
h = int(min(h, max(1, len(ser) // 8)))
if h <= 0 or len(ser) < (p + d + q + max(2, s and (P + D + Q) * 2)):
    st.error("Мало точек для SARIMA при текущих параметрах/горизонте. Уменьшите (p,d,q), (P,D,Q) или h.")
    st.stop()

y_train = ser.iloc[:-h]
y_test  = ser.iloc[-h:]

# Бейзлайны (на случай падения SARIMA)
naive = pd.Series(y_train.iloc[-1], index=y_test.index)
w = min(4, max(1, len(y_train) // 12))
ma_val = y_train.rolling(w).mean().iloc[-1] if len(y_train) >= w else y_train.mean()
moving_avg = pd.Series(ma_val, index=y_test.index)

# SARIMA fit 
order = (int(p), int(d), int(q))
seasonal_order = (int(P), int(D), int(Q), int(s)) if s else (0, 0, 0, 0)
trend = "c" if use_const else None

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# лог-преобразование (стабилизирует масштаб и дисперсию)
eps = 1e-6
y_train_pos = y_train.clip(lower=eps)
y_test_pos  = y_test.clip(lower=eps)

y_train_t = np.log1p(y_train_pos)

# # более стабильные параметры + ограничения
# order = (1, 1, 1)                              
# seasonal_order = (1, 0, 1, int(s)) if int(s) else (0, 0, 0, 0)
# trend = "c" if use_const else None

model = SARIMAX(
    y_train_t,
    order=order,
    seasonal_order=seasonal_order,
    trend=trend,
    enforce_stationarity=True,       # включаем проверки
    enforce_invertibility=True
)
fit = model.fit(disp=False, maxiter=500)

# прогноз в лог-пространстве → обратно к исходной шкале
fc = fit.get_forecast(steps=h)
pred_t = pd.Series(fc.predicted_mean, index=y_test.index)
ci_t   = fc.conf_int(alpha=0.05)

y_pred = np.expm1(pred_t)                       # обратно из логов
lower  = pd.Series(np.expm1(ci_t.iloc[:, 0]).values, index=y_test.index)
upper  = pd.Series(np.expm1(ci_t.iloc[:, 1]).values, index=y_test.index)

# Метрики (если есть тестовые точки)
def mae_safe(y_true, y_hat):
    try:
        from sklearn.metrics import mean_absolute_error
        return float(mean_absolute_error(y_true, y_hat))
    except Exception:
        import numpy as np
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        return float(np.mean(np.abs(y_true - y_hat)))

def rmse_safe(y_true, y_hat):
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_hat))
    except Exception:
        import numpy as np
        y_true, y_hat = np.asarray(y_true), np.asarray(y_hat)
        return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))

res_rows = []
res_rows.append({"model": "Naive", "MAE": mae_safe(y_test, naive), "RMSE": rmse_safe(y_test, naive)})
res_rows.append({"model": f"MovingAvg (w={w})", "MAE": mae_safe(y_test, moving_avg), "RMSE": rmse_safe(y_test, moving_avg)})
res_rows.append({"model": f"SARIMA{order}x{seasonal_order}", "MAE": mae_safe(y_test, y_pred), "RMSE": rmse_safe(y_test, y_pred)})
st.subheader("Метрики (на последних h периодах)")
st.dataframe(pd.DataFrame(res_rows), use_container_width=True)

# Прогноз — график + таблица + скачивание
st.subheader("Прогноз")
fig2, ax2 = plt.subplots()
y_train.plot(ax=ax2, label="Train")
y_test.plot(ax=ax2, label="Test")
y_pred.plot(ax=ax2, label="Forecast")
ax2.fill_between(y_pred.index, lower.values, upper.values, alpha=0.2, label="95% CI")
ax2.set_title(f"SARIMA{order} x {seasonal_order}")
ax2.set_xlabel("Date"); ax2.set_ylabel("Value"); ax2.legend()
st.pyplot(fig2, use_container_width=True)

out = pd.DataFrame({
    "ds": y_pred.index,
    "yhat": y_pred.values,
    "yhat_lower": lower.values,
    "yhat_upper": upper.values,
})
st.dataframe(out.tail(10), use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Скачать forecast.csv", data=csv_bytes, file_name="forecast.csv", mime="text/csv")

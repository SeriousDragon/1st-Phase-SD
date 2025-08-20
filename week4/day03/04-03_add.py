# --- берём исходные транзакции и удаляем выбросы по Тьюки отдельно в каждой группе ---
# границы: upper = Q3 + 1.5*IQR
q = (
    transactions.groupby("group")["revenue"].quantile([0.25, 0.75]).unstack()
)  # считаем Q1 и Q3
q.columns = ["q1", "q3"]  # переименовываем
q["iqr"] = q["q3"] - q["q1"]  # межквартильный размах
q["upper"] = q["q3"] + 1.5 * q["iqr"]  # верхняя граница выбросов

transactions_no = transactions.merge(
    q[["upper"]], left_on="group", right_index=True, how="left"
)  # подливаем порог для своей группы
transactions_no = transactions_no[
    transactions_no["revenue"] <= transactions_no["upper"]
]  # фильтруем выбросы
transactions_no = transactions_no.drop(columns=["upper"])  # чистим временный столбец

# --- считаем только то, чего нет: cum_orders и cum_revenue на очищенных данных ---
# дневные заказы и их кумулятив (на очищенных)
daily_ord_no = (
    transactions_no.groupby(["date", "group"])
    .size()
    .reset_index(name="daily_orders")  # число транзакций в день
    .sort_values(["group", "date"])  # порядок важен для cumsum
)
daily_ord_no["cum_orders"] = daily_ord_no.groupby("group")[
    "daily_orders"
].cumsum()  # кумулятив заказов

# дневная выручка и её кумулятив (на очищенных)
daily_rev_no = (
    transactions_no.groupby(["date", "group"], as_index=False)["revenue"]
    .sum()
    .rename(columns={"revenue": "daily_revenue"})
    .sort_values(["group", "date"])
)
daily_rev_no["cum_revenue"] = daily_rev_no.groupby("group")[
    "daily_revenue"
].cumsum()  # кумулятив выручки

# --- собираем очищенные кумулятивы в один df и считаем кумулятивный средний чек ---
daily_no = (
    daily_rev_no[["date", "group", "cum_revenue"]]
    .merge(
        daily_ord_no[["date", "group", "cum_orders"]], on=["date", "group"], how="inner"
    )
    .sort_values(["group", "date"])
)
daily_no["cum_avg_check"] = (
    daily_no["cum_revenue"] / daily_no["cum_orders"]
)  # формула среднего чека

# --- делаем отношение B к A (без выбросов) и рисуем (и опционально — поверх сырую линию) ---
pivot_no = daily_no.pivot(
    index="date", columns="group", values="cum_avg_check"
).dropna()
rel_pct_no = (pivot_no["B"] / pivot_no["A"] - 1.0) * 100

plt.figure(figsize=(10, 5))
plt.plot(
    rel_pct_no.index, rel_pct_no.values, label="B к A, % (без выбросов)"
)  # линия без выбросов
if "rel_pct" in locals():
    plt.plot(
        rel_pct.index, rel_pct.values, linestyle="--", label="B к A, % (сырые)"
    )  # пунктиром — сырые, если есть
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("B к A: кумулятивный средний чек (сравнение без выбросов vs сырые)")
plt.xlabel("Дата")
plt.ylabel("Изменение, %")
plt.legend()
plt.tight_layout()
plt.show()

print(
    f"[Без выбросов] Финал: {rel_pct_no.iloc[-1]:.2f}% | Пик: {rel_pct_no.max():.2f}% | Мин: {rel_pct_no.min():.2f}%"
)
if "rel_pct" in locals():
    print(
        f"[Сырые]       Финал: {rel_pct.iloc[-1]:.2f}% | Пик: {rel_pct.max():.2f}% | Мин: {rel_pct.min():.2f}%"
    )

from __future__ import annotations

import pandas as pd
import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    """Базовый датафрейм для существующих тестов"""
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# ============================================================
# ТЕСТЫ НОВЫХ ЭВРИСТИК КАЧЕСТВА ДАННЫХ
# ============================================================

def test_has_constant_columns():
    """Тест для эвристики has_constant_columns"""
    # Создаем датафрейм с константной колонкой
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "constant_col": ["A", "A", "A", "A", "A"],  # Все значения одинаковые
        "normal_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "all_nulls": [None, None, None, None, None],  # Все значения пропущены
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_constant_columns установлен в True
    assert flags["has_constant_columns"] == True
    
    # Проверяем детальную информацию
    assert "constant_columns_list" in flags
    assert "constant_col" in flags["constant_columns_list"]
    assert "all_nulls" in flags["constant_columns_list"]
    assert flags["constant_columns_count"] == 2
    
    # Проверяем, что quality_score снижен
    assert flags["quality_score"] < 1.0


def test_has_high_cardinality_categoricals():
    """Тест для эвристики has_high_cardinality_categoricals"""
    # Создаем датафрейм с категориальной колонкой высокой кардинальности
    n_rows = 50
    df = pd.DataFrame({
        "id": range(n_rows),
        "low_card_col": ["A", "B"] * (n_rows // 2),  # Низкая кардинальность
        "high_card_col": [f"item_{i}" for i in range(n_rows)],  # Высокая кардинальность
        "numeric_col": np.random.randn(n_rows),
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_high_cardinality_categoricals установлен в True
    assert flags["has_high_cardinality_categoricals"] == True
    
    # Проверяем детальную информацию
    assert "high_cardinality_columns" in flags
    assert len(flags["high_cardinality_columns"]) >= 1
    
    # Проверяем, что high_card_col попал в список проблемных
    high_card_columns = [col["column"] for col in flags["high_cardinality_columns"]]
    assert "high_card_col" in high_card_columns
    
    # Проверяем, что low_card_col НЕ попал в список проблемных
    assert "low_card_col" not in high_card_columns
    
    # Проверяем количество уникальных значений для high_card_col
    for col_info in flags["high_cardinality_columns"]:
        if col_info["column"] == "high_card_col":
            assert col_info["unique_count"] == n_rows
            break


def test_has_suspicious_id_duplicates():
    """Тест для эвристики has_suspicious_id_duplicates"""
    # Создаем датафрейм с ID-колонкой, содержащей дубликаты
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 2, 3, 4, 5],  # Дубликаты: 1, 2, 3
        "order_id": [f"ORD_{i:03d}" for i in range(8)],  # Уникальный ID
        "email": ["a@test.com", "b@test.com", "c@test.com", "a@test.com", 
                  "b@test.com", "c@test.com", "d@test.com", "e@test.com"],
        "value": [100, 200, 300, 400, 500, 600, 700, 800],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_suspicious_id_duplicates установлен в True
    assert flags["has_suspicious_id_duplicates"] == True
    
    # Проверяем детальную информацию
    assert "suspicious_id_columns" in flags
    assert len(flags["suspicious_id_columns"]) >= 1
    
    # Проверяем, что user_id попал в список подозрительных
    suspicious_cols = [col["column"] for col in flags["suspicious_id_columns"]]
    assert "user_id" in suspicious_cols
    
    # Проверяем, что order_id НЕ попал в список подозрительных (он уникальный)
    assert "order_id" not in suspicious_cols
    
    # Проверяем уникальность для user_id
    for col_info in flags["suspicious_id_columns"]:
        if col_info["column"] == "user_id":
            uniqueness_ratio = col_info["uniqueness_ratio"]
            # В нашем случае: 5 уникальных из 8 = 0.625
            expected_uniqueness = 5 / 8  # 5 уникальных значений из 8 строк
            assert abs(uniqueness_ratio - expected_uniqueness) < 0.01
            assert uniqueness_ratio < 0.999  # Должно быть меньше порога 99.9%
            break


def test_has_many_zero_values():
    """Тест для эвристики has_many_zero_values"""
    # Создаем датафрейм с колонками, содержащими много нулей
    df = pd.DataFrame({
        "id": range(10),
        "all_zeros": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 100% нулей
        "mostly_zeros": [0, 0, 0, 0, 0, 1, 2, 3, 0, 0],  # 70% нулей
        "normal_col": np.random.randn(10),
        "mixed_values": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],  # 60% нулей
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_many_zero_values установлен в True
    assert flags["has_many_zero_values"] == True
    
    # Проверяем детальную информацию
    assert "high_zero_columns" in flags
    assert len(flags["high_zero_columns"]) >= 1
    
    # Проверяем, что all_zeros попал в список проблемных
    zero_columns = [col["column"] for col in flags["high_zero_columns"]]
    assert "all_zeros" in zero_columns
    
    # Проверяем соотношение нулей для all_zeros
    for col_info in flags["high_zero_columns"]:
        if col_info["column"] == "all_zeros":
            assert col_info["zero_ratio"] == 1.0  # 100% нулей
            assert col_info["zero_count"] == 10  # 10 нулевых значений
            break


def test_quality_score_with_new_heuristics():
    """Тест, что новые эвристики влияют на итоговый quality_score"""
    # Создаем "плохой" датафрейм с несколькими проблемами
    n_rows = 80
    df = pd.DataFrame({
        "duplicate_id": [1, 2, 3] * (n_rows // 3),  # Много дубликатов
        "constant_col": ["X"] * n_rows,  # Константная колонка
        "high_card_col": [f"value_{i}" for i in range(n_rows)],  # Высокая кардинальность
        "zero_col": [0] * n_rows,  # Все нули
        "normal_col": np.random.randn(n_rows),
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что все новые флаги установлены
    assert flags["has_constant_columns"] == True
    assert flags["has_high_cardinality_categoricals"] == True
    assert flags["has_suspicious_id_duplicates"] == True
    assert flags["has_many_zero_values"] == True
    
    # Проверяем, что quality_score существенно снижен
    # Базовый скор 1.0, минус штрафы за каждую эвристику
    assert flags["quality_score"] < 0.8  # Должен быть значительно ниже 1.0
    
    # Проверяем, что скор находится в допустимом диапазоне
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_quality_flags_with_good_data():
    """Тест, что на хороших данных новые флаги не срабатывают"""
    # Создаем "хороший" датафрейм без проблем
    df = pd.DataFrame({
        "unique_id": [f"ID_{i}" for i in range(100)],  # Уникальный ID
        "category": ["A", "B", "C"] * 33 + ["A"],  # Низкая кардинальность
        "value": np.random.randn(100),
        "status": [0, 1] * 50,  # Булевы значения, не все нули
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что новые флаги НЕ установлены
    assert flags["has_constant_columns"] == False
    assert flags["has_high_cardinality_categoricals"] == False
    assert flags["has_suspicious_id_duplicates"] == False
    assert flags["has_many_zero_values"] == False
    
    # Проверяем, что quality_score достаточно высок
    assert flags["quality_score"] >= 0.7


def test_edge_cases_for_new_heuristics():
    """Тест пограничных случаев для новых эвристик"""
    # Тест 1: ID-колонка с пропусками (не должна считаться проблемной)
    df1 = pd.DataFrame({
        "user_id_with_nulls": [1, 2, None, 4, None, 6],
        "value": [10, 20, 30, 40, 50, 60],
    })
    
    summary1 = summarize_dataset(df1)
    missing_df1 = missing_table(df1)
    flags1 = compute_quality_flags(summary1, missing_df1)
    
    # Если пропусков много, эвристика не должна срабатывать
    if missing_df1.loc["user_id_with_nulls", "missing_share"] >= 0.5:
        assert flags1["has_suspicious_id_duplicates"] == False
    
    # Тест 2: Колонка с одним уникальным значением, но с пропусками
    df2 = pd.DataFrame({
        "col_with_nulls": ["A", "A", None, "A", None],
    })
    
    summary2 = summarize_dataset(df2)
    missing_df2 = missing_table(df2)
    flags2 = compute_quality_flags(summary2, missing_df2)
    
    # Проверяем, что константная колонка с пропусками не учитывается как константная
    # (в нашей реализации учитывается только если пропусков нет)
    assert flags2["has_constant_columns"] == False
    
    # Тест 3: Маленький датасет с высокой кардинальностью
    df3 = pd.DataFrame({
        "tiny_dataset_high_card": [f"val_{i}" for i in range(10)],
    })
    
    summary3 = summarize_dataset(df3)
    missing_df3 = missing_table(df3)
    flags3 = compute_quality_flags(summary3, missing_df3)
    
    # Для маленького датасета порог кардинальности снижается
    # (в нашей реализации: min(1000, n_rows * 0.5) = min(1000, 5) = 5
    # И уникальных значений должно быть > 50 (по нашей логике)
    # Так как у нас 10 уникальных, флаг не должен сработать
    assert flags3["has_high_cardinality_categoricals"] == False


def test_integration_of_new_flags():
    """Интеграционный тест: проверка, что новые флаги корректно работают вместе"""
    df = pd.DataFrame({
        "product_id": ["P001", "P002", "P001", "P003", "P002"],  # Дубликаты ID
        "category": ["Electronics"] * 5,  # Константная
        "sku": [f"SKU_{i:06d}" for i in range(1000, 1005)],  # Высокая кардинальность (в реальном датасете)
        "price": [0, 0, 0, 0, 0],  # Все нули
        "rating": [4.5, 4.7, 4.3, 4.6, 4.4],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем все флаги
    assert flags["has_constant_columns"] == True
    assert flags["has_suspicious_id_duplicates"] == True
    assert flags["has_many_zero_values"] == True
    
    # Проверяем, что информация сохраняется в словаре
    assert "constant_columns_list" in flags
    assert "suspicious_id_columns" in flags
    assert "high_zero_columns" in flags
    
    # Проверяем, что каждая проблемная колонка есть в соответствующих списках
    constant_cols = flags.get("constant_columns_list", [])
    suspicious_cols = [col["column"] for col in flags.get("suspicious_id_columns", [])]
    zero_cols = [col["column"] for col in flags.get("high_zero_columns", [])]
    
    assert "category" in constant_cols
    assert "product_id" in suspicious_cols
    assert "price" in zero_cols

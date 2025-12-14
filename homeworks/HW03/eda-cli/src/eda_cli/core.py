from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(summary: DatasetSummary, missing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Простейшие эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    и т.п.
    """
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # НОВЫЕ ЭВРИСТИКИ (добавлены)
    flags["has_constant_columns"] = False
    flags["has_high_cardinality_categoricals"] = False
    flags["has_suspicious_id_duplicates"] = False
    flags["has_many_zero_values"] = False
    
    # Эвристика 1: Проверка на константные колонки
    constant_columns = []
    for col in summary.columns:
        # Если уникальных значений всего 1 и пропусков нет, либо уникальных 0 (все пропуски)
        if col.unique <= 1 and col.missing == 0:
            constant_columns.append(col.name)
        # Если все значения пропущены
        elif col.missing == summary.n_rows:
            constant_columns.append(col.name)
    
    if constant_columns:
        flags["has_constant_columns"] = True
        flags["constant_columns_list"] = constant_columns
        flags["constant_columns_count"] = len(constant_columns)

    # Эвристика 2: Проверка на высокую кардинальность категориальных признаков
    high_cardinality_cols = []
    for col in summary.columns:
        # Проверяем строковые или категориальные колонки
        if (col.dtype == 'object' or 'category' in col.dtype.lower()) and col.is_numeric == False:
            # Порог: если уникальных значений больше 50% от общего числа строк
            # или больше 1000 уникальных значений (для больших датасетов)
            card_threshold = min(1000, summary.n_rows * 0.5)
            if col.unique > card_threshold and col.unique > 50:
                high_cardinality_cols.append({
                    "column": col.name,
                    "unique_count": col.unique,
                    "threshold": card_threshold
                })
    
    if high_cardinality_cols:
        flags["has_high_cardinality_categoricals"] = True
        flags["high_cardinality_columns"] = high_cardinality_cols
        flags["high_cardinality_count"] = len(high_cardinality_cols)

    # Эвристика 3: Проверка на дубликаты ID-колонок
    suspicious_id_cols = []
    # Ищем колонки, которые могут быть идентификаторами
    id_like_patterns = ['id', '_id', 'uuid', 'guid', 'key', 'code', 'token', 'num', 'nbr', 'no', 'nr']
    
    for col in summary.columns:
        col_lower = col.name.lower()
        
        # Проверяем, похожа ли колонка на идентификатор
        is_id_like = any(pattern in col_lower for pattern in id_like_patterns)
        
        if is_id_like:
            # Для ID-колонок ожидаем высокую уникальность (близкую к 100%)
            uniqueness_ratio = col.unique / summary.n_rows if summary.n_rows > 0 else 0
            
            # Если уникальность меньше 99.9% - подозрительно
            if uniqueness_ratio < 0.999 and col.missing_share < 0.5:
                suspicious_id_cols.append({
                    "column": col.name,
                    "uniqueness_ratio": round(uniqueness_ratio, 4),
                    "expected_min": 0.999
                })
    
    if suspicious_id_cols:
        flags["has_suspicious_id_duplicates"] = True
        flags["suspicious_id_columns"] = suspicious_id_cols
        flags["suspicious_id_count"] = len(suspicious_id_cols)

    # Эвристика 4: Проверка на много нулевых значений в числовых колонках
    high_zero_cols = []
    for col in summary.columns:
        if col.is_numeric and col.min == 0 and col.max == 0 and col.non_null > 0:
            # Вся колонка состоит из нулей
            zero_ratio = 1.0
            if zero_ratio > 0.8:  # Порог: 80%
                high_zero_cols.append({
                    "column": col.name,
                    "zero_ratio": zero_ratio,
                    "zero_count": col.non_null
                })
    
    if high_zero_cols:
        flags["has_many_zero_values"] = True
        flags["high_zero_columns"] = high_zero_cols
        flags["high_zero_count"] = len(high_zero_cols)

    # Простейший «скор» качества с учетом новых эвристик
    score = 1.0
    score -= max_missing_share  # чем больше пропусков, тем хуже
    
    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    
    # Штрафы за новые эвристики
    if flags["has_constant_columns"]:
        score -= 0.1
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.05 * min(flags.get("high_cardinality_count", 0), 4)  # до 0.2
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.15
    if flags["has_many_zero_values"]:
        score -= 0.1

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = round(score, 3)

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)


# Дополнительные функции, которые могли быть в исходном коде
def get_numeric_columns_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Получить сводку по числовым колонкам.
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return pd.DataFrame()
    
    summary = numeric_df.describe().transpose()
    summary['missing'] = numeric_df.isna().sum()
    summary['missing_share'] = summary['missing'] / len(numeric_df)
    summary['dtype'] = [str(df[col].dtype) for col in numeric_df.columns]
    
    return summary[['dtype', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing', 'missing_share']]


def get_categorical_columns_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Получить сводку по категориальным колонкам.
    """
    categorical_cols = []
    for col in df.columns:
        if ptypes.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            categorical_cols.append(col)
    
    result = {}
    for col in categorical_cols:
        s = df[col]
        result[col] = {
            'dtype': str(s.dtype),
            'count': s.count(),
            'missing': s.isna().sum(),
            'missing_share': s.isna().sum() / len(s),
            'unique': s.nunique(dropna=True),
            'most_common': s.mode().iloc[0] if not s.mode().empty else None,
            'most_common_count': s.value_counts().iloc[0] if not s.value_counts().empty else 0,
            'most_common_share': s.value_counts().iloc[0] / s.count() if s.count() > 0 else 0,
        }
    
    return result


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Обнаружение выбросов с помощью метода IQR.
    """
    if column not in df.columns or not ptypes.is_numeric_dtype(df[column]):
        return {}
    
    data = df[column].dropna()
    if len(data) == 0:
        return {}
    
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return {
        'column': column,
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outliers_count': int(len(outliers)),
        'outliers_share': float(len(outliers) / len(data)),
        'min_outlier': float(outliers.min()) if len(outliers) > 0 else None,
        'max_outlier': float(outliers.max()) if len(outliers) > 0 else None,
    }


def get_column_distribution(df: pd.DataFrame, column: str, bins: int = 10) -> Dict[str, Any]:
    """
    Получить распределение значений колонки.
    """
    if column not in df.columns:
        return {}
    
    s = df[column]
    result = {
        'column': column,
        'dtype': str(s.dtype),
        'count': int(s.count()),
        'missing': int(s.isna().sum()),
        'unique': int(s.nunique(dropna=True)),
    }
    
    if ptypes.is_numeric_dtype(s):
        data = s.dropna()
        if len(data) > 0:
            hist, bin_edges = pd.cut(data, bins=bins, retbins=True)
            bin_counts = hist.value_counts().sort_index()
            
            result['histogram'] = {
                'bins': bin_edges.tolist(),
                'counts': bin_counts.values.tolist(),
                'bin_labels': [str(interval) for interval in bin_counts.index]
            }
            result['skewness'] = float(data.skew())
            result['kurtosis'] = float(data.kurtosis())
    
    return result


def validate_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Проверка типов данных в колонках на соответствие ожидаемым.
    """
    issues = {
        'numeric_in_object': [],
        'datetime_in_object': [],
        'boolean_in_numeric': [],
    }
    
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        
        # Проверка числовых данных в object
        if dtype_str == 'object':
            try:
                # Пробуем преобразовать к numeric
                numeric_test = pd.to_numeric(s.dropna(), errors='coerce')
                if numeric_test.notna().sum() > len(s) * 0.5:  # Более 50% успешно конвертируется
                    issues['numeric_in_object'].append(col)
            except:
                pass
        
        # Проверка дат в object
        if dtype_str == 'object':
            try:
                datetime_test = pd.to_datetime(s.dropna(), errors='coerce')
                if datetime_test.notna().sum() > len(s) * 0.5:  # Более 50% успешно конвертируется
                    issues['datetime_in_object'].append(col)
            except:
                pass
        
        # Проверка булевых значений в numeric
        if ptypes.is_numeric_dtype(s):
            unique_vals = s.dropna().unique()
            if len(unique_vals) <= 3 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                issues['boolean_in_numeric'].append(col)
    
    return issues


def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str = "df1", df2_name: str = "df2") -> Dict[str, Any]:
    """
    Сравнение двух датасетов.
    """
    comparison = {
        'shapes': {
            df1_name: df1.shape,
            df2_name: df2.shape,
        },
        'common_columns': list(set(df1.columns) & set(df2.columns)),
        'unique_to_df1': list(set(df1.columns) - set(df2.columns)),
        'unique_to_df2': list(set(df2.columns) - set(df1.columns)),
    }
    
    # Сравнение типов для общих колонок
    type_comparison = {}
    for col in comparison['common_columns']:
        type1 = str(df1[col].dtype)
        type2 = str(df2[col].dtype)
        if type1 != type2:
            type_comparison[col] = {df1_name: type1, df2_name: type2}
    
    comparison['type_differences'] = type_comparison
    
    return comparison
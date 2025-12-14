from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличку по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Сколько top-значений выводить для категориальных признаков."),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта в Markdown."),
    min_missing_share: float = typer.Option(0.1, help="Порог доли пропусков для проблемных колонок (0.0-1.0)."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    # Проверка валидности параметров
    if min_missing_share < 0 or min_missing_share > 1:
        typer.echo(f"Ошибка: min_missing_share должен быть между 0.0 и 1.0, получено {min_missing_share}")
        raise typer.Exit(code=1)
    
    if top_k_categories <= 0:
        typer.echo(f"Ошибка: top_k_categories должен быть положительным, получено {top_k_categories}")
        raise typer.Exit(code=1)
    
    if max_hist_columns <= 0:
        typer.echo(f"Ошибка: max_hist_columns должен быть положительным, получено {max_hist_columns}")
        raise typer.Exit(code=1)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    
    # Используем новый параметр top_k_categories
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)
    
    # 3. Определяем проблемные колонки на основе min_missing_share
    problematic_columns = []
    if not missing_df.empty:
        problematic_columns = missing_df[missing_df['missing_share'] >= min_missing_share].index.tolist()

    # 4. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 5. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        # Используем пользовательский заголовок
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")
        
        # Добавляем информацию о параметрах отчёта
        f.write("## Параметры генерации отчёта\n\n")
        f.write(f"- Макс. гистограмм: **{max_hist_columns}**\n")
        f.write(f"- Top-K категорий: **{top_k_categories}**\n")
        f.write(f"- Порог проблемных пропусков: **{min_missing_share:.1%}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        
        # Добавляем новые эвристики
        if 'has_constant_columns' in quality_flags and quality_flags['has_constant_columns']:
            f.write(f"- Есть константные колонки: **ДА** ({quality_flags.get('constant_columns_count', 0)} шт.)\n")
        
        if 'has_high_cardinality_categoricals' in quality_flags and quality_flags['has_high_cardinality_categoricals']:
            f.write(f"- Есть высокая кардинальность: **ДА** ({quality_flags.get('high_cardinality_count', 0)} шт.)\n")
        
        if 'has_suspicious_id_duplicates' in quality_flags and quality_flags['has_suspicious_id_duplicates']:
            f.write(f"- Есть подозрительные дубликаты ID: **ДА** ({quality_flags.get('suspicious_id_count', 0)} шт.)\n")
        
        if 'has_many_zero_values' in quality_flags and quality_flags['has_many_zero_values']:
            f.write(f"- Есть много нулевых значений: **ДА** ({quality_flags.get('high_zero_count', 0)} шт.)\n\n")
        else:
            f.write("\n")

        # Добавляем раздел о проблемных колонках на основе min_missing_share
        if problematic_columns:
            f.write("## Проблемные колонки (высокие пропуски)\n\n")
            f.write(f"Колонки с долей пропусков ≥ {min_missing_share:.1%}:\n\n")
            for col in problematic_columns:
                missing_share = missing_df.loc[col, 'missing_share']
                missing_count = missing_df.loc[col, 'missing_count']
                f.write(f"- **{col}**: {missing_count} пропусков ({missing_share:.1%})\n")
            f.write("\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")
            # Добавляем сводку по пропускам
            total_missing = missing_df['missing_count'].sum()
            total_cells = summary.n_rows * summary.n_cols
            overall_missing_share = total_missing / total_cells if total_cells > 0 else 0
            f.write(f"Общая статистика пропусков:\n")
            f.write(f"- Всего пропущенных значений: **{total_missing}**\n")
            f.write(f"- Общая доля пропусков: **{overall_missing_share:.2%}**\n")
            f.write(f"- Колонок с пропусками: **{(missing_df['missing_count'] > 0).sum()}** из **{summary.n_cols}**\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")
            # Добавляем информацию о сильных корреляциях
            strong_correlations = []
            if not corr_df.empty:
                corr_values = corr_df.unstack()
                corr_values = corr_values[corr_values.index.get_level_values(0) != corr_values.index.get_level_values(1)]
                strong_corr = corr_values[(corr_values.abs() >= 0.7) & (corr_values.abs() < 1.0)]
                if not strong_corr.empty:
                    f.write("Сильные корреляции (≥ |0.7|):\n")
                    for (col1, col2), value in strong_corr.items():
                        f.write(f"- **{col1}** ↔ **{col2}**: {value:.3f}\n")
                    f.write("\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"См. файлы в папке `top_categories/`. Для каждой категориальной колонки показаны топ-{top_k_categories} значений.\n\n")
            f.write(f"Всего категориальных колонок: **{len(top_cats)}**\n\n")
            # Добавляем примеры категориальных колонок
            f.write("Примеры категориальных колонок:\n")
            for i, (col_name, table) in enumerate(list(top_cats.items())[:3]):
                unique_count = summary_df[summary_df['name'] == col_name]['unique'].iloc[0] if not summary_df.empty else "N/A"
                f.write(f"- **{col_name}**: {unique_count} уникальных значений\n")
            if len(top_cats) > 3:
                f.write(f"- ... и еще {len(top_cats) - 3} колонок\n")
            f.write("\n")

        f.write("## Гистограммы числовых колонок\n\n")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        f.write(f"Всего числовых колонок: **{len(numeric_cols)}**\n")
        f.write(f"Сгенерировано гистограмм: **{min(max_hist_columns, len(numeric_cols))}** (лимит: {max_hist_columns})\n")
        f.write("См. файлы `hist_*.png`.\n\n")
        
        # Добавляем список числовых колонок
        if numeric_cols:
            f.write("Числовые колонки в датасете:\n")
            for col in numeric_cols[:10]:
                f.write(f"- {col}\n")
            if len(numeric_cols) > 10:
                f.write(f"- ... и еще {len(numeric_cols) - 10} колонок\n")
            f.write("\n")

    # 6. Картинки
    # Используем новый параметр max_hist_columns
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    # 7. Вывод информации в консоль
    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    typer.echo(f"\nИспользованные параметры:")
    typer.echo(f"  • Макс. гистограмм: {max_hist_columns}")
    typer.echo(f"  • Top-K категорий: {top_k_categories}")
    typer.echo(f"  • Заголовок: {title}")
    typer.echo(f"  • Порог проблемных пропусков: {min_missing_share:.1%}")
    
    if problematic_columns:
        typer.echo(f"\n⚠️  Обнаружены проблемные колонки (пропуски ≥ {min_missing_share:.1%}):")
        for col in problematic_columns[:5]:
            missing_share = missing_df.loc[col, 'missing_share']
            typer.echo(f"  • {col}: {missing_share:.1%}")
        if len(problematic_columns) > 5:
            typer.echo(f"  • ... и еще {len(problematic_columns) - 5} колонок")


if __name__ == "__main__":
    app()
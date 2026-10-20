# Titanic Survival Prediction (Kaggle)

## О проекте
End-to-end Machine Learning проект: бинарная классификация (предсказание выживаемости пассажиров Титаника). 
Цель — построить надежный пайплайн с защитой от утечек данных и сравнить различные алгоритмы.

## Стек
* Python (pandas, numpy)
* Scikit-learn (Pipeline, ColumnTransformer, модели)
* Matplotlib, Seaborn (EDA)
* Jupyter Notebook

##  Что сделано
1. **EDA:** анализ выживаемости по полу, классу, возрасту. Построены графики (barplot, heatmap).
2. **Feature Engineering:** 
   - Извлечен признак `Title` (Mr, Mrs, Miss, Master, Other) из колонки `Name`.
   - Созданы признаки `FamilySize` и `IsAlone`.
3. **Эксперимент с пропусками (`Age`):** Сравнены 4 стратегии заполнения (медиана, медиана+флаг, sentinel -999, биннинг). Выбран биннинг (child, teen, young, adult, senior) как наиболее стабильный.
4. **Моделирование:** Обучены 5 моделей (LogReg, KNN, RandomForest, HistGradientBoosting, SVC). Использована 5-fold кросс-валидация (`StratifiedKFold`) с защитой от утечек (через `Pipeline`).
5. **Интерпретация:** Проведен анализ важности признаков (Permutation Importance) и анализ коэффициентов логистической регрессии.
6. **Submission:** Финальная модель — SVC (RBF). Сгенерирован `submission.csv` (Public LB: 0.777).

## Результаты Кросс-валидации
| Модель | CV Accuracy | Train Accuracy | Разрыв |
|---|---|---|---|
| SVC | 0.8316 | 0.8454 | 0.0138 |
| KNN (k=5) | 0.8260 | 0.8650 | 0.0390 |
| HistGradientBoosting | 0.8249 | 0.8917 | 0.0668 |
| LogisticRegression | 0.8204 | 0.8280 | 0.0076 |
| RandomForest | 0.8114 | 0.9834 | 0.1720 |

##  Главные выводы
* Извлечение `Title` — самый мощный способ feature engineering для этой задачи. 
* Permutation Importance показал, что `Title_Mr` "забрал" важность у `Sex_male`, так как дублирует информацию о поле, но добавляет контекст возраста/статуса.
* SVC показал лучший результат при минимальном переобучении (разрыв ~1.3%).

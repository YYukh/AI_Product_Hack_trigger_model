# Yura production-shaped signal pipeline

Альтернативный полноценный pipeline для прямого сравнения с
`notebooks/prod_pipline.ipynb`. Он использует тот же источник ЦБ, causal
features, targets `G0/W1` и тот же контракт финального backtest, но сокращает
число обучаемых сущностей и явно разделяет ответственность компонентов.

## Архитектура

```text
ЦБ РФ -> daily panel -> causal features -> G0/W1 labels
                         |-> standalone rule engines
                         |-> pooled probability models
                         |-> pooled expected-BPS model
                                  |
                    deterministic evidence aggregation
                                  |
                        replaceable selector
                                  |
                  cooldown + rolling frequency policy
                                  |
                 immutable holdout events and reports
```

Главное архитектурное правило: labels, evidence и клиентский push — разные
сущности. `G0/W1` нужны для обучения и проверки источников evidence. Rule и ML
движки отдают стандартизированные кандидаты. Evidence-слой ничего не обучает и
не вводит новую скрытую оптимизацию. Selector решает, какие возможности
допустить, а детерминированная policy отдельно гарантирует cooldown и максимум
два сигнала за любые семь дней по каждой валюте.

Используются две версии label:

- `G0`: текущий курс является точным минимумом в окне `±h`;
- `W1`: в момент `t` курс находится в нижних 15% своего 90-дневного диапазона,
  а через `h` дней становится хуже минимум на 75 BPS. Это исходное определение
  `W1` из общего pipeline.

### Base engines

- Семь одиночных rule-семейств представляют разные экономические идеи:
  relative cheapness, negative momentum, down streak, normalized negative
  surprise, trend down, causal Kalman downtrend и reversal from low.
- Четыре заранее объявленных `AND`-архетипа соединяют разные виды evidence:
  cheapness + downward pressure, cheapness + surprise, persistent Kalman
  downtrend и cheapness + reversal. Общего pairwise-перебора в pipeline нет.
- `percentile`, `z-score` и distance from low являются вариантами одного
  `relative_cheapness`, а не тремя независимыми подтверждениями.
- Параметры каждого семейства переоцениваются на rolling train отдельно для
  `target_family × horizon`, но pooled по валютам. Используется 159 заранее
  ограниченных вариантов вместо полного discovery-grid.
- Два небольших pooled ML-классификатора — по одному для `G0` и `W1`.
  Валюта и горизонт входят как категориальные признаки, поэтому число моделей
  не растёт как `currency × target × horizon`. Это основной режим
  `ml_scope='pooled'`.
- ML использует тот же компактный feature space плюс четыре causal-признака,
  прошедшие смысловой отбор: Kalman level gap, Kalman trend, Kalman reversal и
  normalized return surprise. Остальные experimental features не перенесены.
- Для проверки межвалютной неоднородности доступны два совместимых режима без
  изменения candidate-контракта, selector, policy и отчётов:
  `hybrid` (общая модель, равный вес `currency × horizon`, контекстная temporal
  calibration) и `per_currency` (по модели на валюту и family, но горизонты
  остаются pooled; общая модель служит fallback при нехватке данных).
- Для всех режимов `confidence_lift` ML сравнивает вероятность не с общей
  частотой класса, а со сглаженной train-частотой той же валюты для конкретных
  `family × horizon`. Поэтому различия базовой частоты KGS и других валют не
  маскируются общей pooled baseline.
- Одна pooled-регрессия оценивает ожидаемый BPS-uplift относительно случайного
  входа той же валюты и горизонта.
- Все движки переобучаются каждые шесть месяцев на последних 36 месяцах.
  В train попадают только labels, полностью созревшие до момента refit.
- Rule confidence — иерархическая precision на trailing 24-месячной истории
  уже созревших OOS-сигналов со shrinkage к pooled оценке. ML confidence —
  temporally calibrated `predict_proba` текущей версии модели.

### Evidence, selector и policy

- Разнородные вероятности и экономические оценки переводятся в causal
  относительные scores внутри сопоставимых конфигураций. Rank считается по
  текущему evidence и предыдущим 36 календарным месяцам; более старая рыночная
  история выпадает из шкалы, а будущие наблюдения не читаются.
- Длинный горизонт удаляется лишь при строгом Pareto-доминировании более
  быстрым кандидатом одновременно по статистическому и экономическому evidence.
- `ThresholdSelector` — простой прозрачный default. `LearnedOpportunitySelector`
  поддерживает `logistic_regression` и `extra_trees`, не меняя engines, policy,
  holdout или формат отчётов. У learned-варианта pre-holdout история строго
  разделена на fit модели, отдельную калибровку вероятности и validation порога;
  после этого и модель, и calibrator, и порог заморожены.
- `SignalPolicyConfig` не обучается: он применяет cooldown и rolling hard cap.
  Это продуктовые ограничения, а не способ улучшить качество на holdout.

## Динамическая временная схема

`TemporalPlan` строится по фактически доступной истории. По умолчанию:

- первые 36 месяцев — только initial train;
- затем начинается causal base OOS;
- первые 12 месяцев OOS служат warm-up для confidence;
- следующие 36 месяцев доступны selector validation;
- всё, что идёт после этой заранее определённой границы, является единственным
  holdout. Его начало не сдвигается при поступлении новых данных.

Если история начинается в январе 2018 года и актуальна на сентябрь 2026 года,
это автоматически даёт initial train 2018–2020, OOS с 2021 года, validation
2022–2024 и holdout с 01.01.2025. Границы можно явно заморозить в конфигурации,
но алгоритм не привязан к этим календарным годам.

Base engines продолжают планово переобучаться внутри holdout только на прошлых
созревших labels. Selector и policy после validation заморожены.

## Отчёты

- `backtest_summary` — совместимый с основным pipeline 80-строчный отчёт:
  5 `currency`, 25 `currency+horizon` и 50
  `currency+horizon+target_family` строк.
- `action_summary` — частота и BPS по уникальным клиентским push. В нём нет
  общей precision по двум несовместимым labels.
- `holdout_coverage` и `holdout_quarterly_stability` — диагностика покрытия и
  временной устойчивости, а не дополнительные критерии подгонки holdout.
- `base_replay.audit` фиксирует реальную границу train каждого refit и позволяет
  проверить отсутствие незрелых labels.

## Запуск и расширение

Откройте `notebooks/yura_pipeline.ipynb` и выполните все ячейки. Ноутбук является
тонким runner: подготовка данных, один вызов pipeline, audit, отчёты и графики.
Тяжёлая логика находится в `research/yura/src`.

Для отбора новых сигналов отдельно предназначен
`notebooks/indicator_discovery.ipynb`. Это исследовательский стенд, который не
меняет продуктовый pipeline: на единой walk-forward-схеме он сравнивает четыре
исходные target-концепции (`G0`, `G1`, `W0`, `W1`), одиночные правила,
train-only отобранные `AND`/`OR`-комбинации и компактные ML-модели-индикаторы.
Период после discovery OOS зарезервирован и не участвует в выборе. Основные
результаты — `indicator_leaderboard`, `best_indicators`,
`indicator_family_summary` и fold-level таблица `all_discovery_folds`.
Исторический выполненный снимок, ещё содержащий сравнение с удалённым
`W1_YURA`, сохранён в `reports/indicator_discovery_results.html`, а его
интерпретация — в `reports/indicator_discovery_findings.md`; для просмотра
архивного результата notebook повторно запускать не нужно.

Новый простой индикатор добавляется как набор вариантов в `RULE_LIBRARY`.
Новая вероятностная модель добавляется именованным `MLModelSpec` в
`EngineRegistry`; она автоматически проходит тот же causal WF и выдаёт тот же
candidate-контракт.
Альтернативный selector реализует три метода протокола `OpportunitySelector`:
`fit`, `select` и `policy_config`, после чего передаётся в
`run_yura_pipeline(..., selector=my_selector)`. Формат итоговой таблицы при этом
не меняется.

В notebook selector выбирается одной строкой:
`SELECTOR_TYPE = 'threshold'`, `'logistic_regression'` или `'extra_trees'`.
Для ML-вариантов `confidence` — temporally calibrated вероятность целевого
события; `ExtraTrees.predict_proba` напрямую как confidence не используется.

Архитектура base ML выбирается независимо строкой `ML_SCOPE = 'pooled'`,
`'hybrid'` или `'per_currency'`. Менять валютные threshold вручную не нужно:
различия валют учитываются моделью/калибровкой и локальной causal baseline, а
финальный лимит остаётся единым и применяется отдельно по каждой валюте.

Для прямого сравнения рассчитанных таблиц используйте
`compare_backtest_summaries(current_summary, yura_summary)`.

Код альтернативы находится только внутри `research/yura`; существующие `src`,
production notebook и production-конфигурация не изменяются.

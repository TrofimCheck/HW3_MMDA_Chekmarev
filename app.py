import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test_df = load_test()

features = list(model.feature_names_in_)
X_test = test_df[features]
y_test = test_df['target'].to_numpy()
proba = test_proba(model, X_test)

st.sidebar.header('Настройки')
threshold = st.sidebar.slider('Порог решения', 0.0, 1.0, float(config['threshold']), 0.01, key='threshold')
st.sidebar.caption(f'Порог итоговой модели: {config["threshold"]:.2f}. '
                   'Вино попадает в подборку, если вероятность того, что оно хорошее, не меньше порога.')

st.title('🍷 Отбор хороших вин для подборки')

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab_one:
    st.subheader('Результаты лабораторного анализа')
    values = {}
    numeric = [f for f in features if f != 'is_red']
    cols = st.columns(3)
    for i, f in enumerate(numeric):
        lo, hi, med = float(X_test[f].min()), float(X_test[f].max()), float(X_test[f].median())
        values[f] = cols[i % 3].number_input(f, min_value=lo, max_value=hi, value=med,
                                             step=(hi - lo) / 100, format='%.4f')
    default_red = int(X_test['is_red'].median()) == 1
    wine_type = st.radio('Тип вина', ['красное', 'белое'], index=0 if default_red else 1, horizontal=True)
    values['is_red'] = 1 if wine_type == 'красное' else 0

    x = pd.DataFrame([values])[features]
    p = model.predict_proba(x)[0, 1]

    st.metric('Вероятность, что вино хорошее', f'{p:.1%}')
    if p >= threshold:
        st.success(f'Решение модели: хорошее вино, рекомендуем в подборку (порог {threshold:.2f}).')
    else:
        st.warning(f'Решение модели: в подборку не берём (порог {threshold:.2f}).')

with tab_metrics:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)

    c1, c2, c3 = st.columns(3)
    c1.metric('Precision', f'{precision:.3f}')
    c2.metric('Recall', f'{recall:.3f}')
    c3.metric('F1', f'{f1:.3f}')

    col_cm, col_roc, col_pr = st.columns(3)

    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]),
                                   display_labels=['не хорошее', 'хорошее']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_xlabel('Предсказание модели')
    ax.set_ylabel('Истинный класс')
    ax.set_title('Матрица ошибок')
    col_cm.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, proba):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.scatter(fp / (fp + tn), recall, color='red', zorder=3, label=f'порог {threshold:.2f}')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC-кривая')
    ax.legend()
    col_roc.pyplot(fig)
    plt.close(fig)

    prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(rec_curve, prec_curve, label=f'PR-AUC = {metrics.average_precision_score(y_test, proba):.3f}')
    ax.axhline(y_test.mean(), color='k', linestyle='--', linewidth=1)
    if tp + fp > 0:
        ax.scatter(recall, precision, color='red', zorder=3, label=f'порог {threshold:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR-кривая')
    ax.legend()
    col_pr.pyplot(fig)
    plt.close(fig)

    st.info(
        f'При текущем пороге в подборку из {len(y_test)} вин тестовой выборки попадает {tp + fp}: '
        f'из них {tp} действительно хорошие и {fp} посредственные, а {fn} хороших вин остаются за бортом.\n\n'
        '**Если порог увеличить**, модель будет брать в подборку только вина, в которых она уверена: '
        'подборка станет меньше, посредственных вин в ней будет меньше (precision растёт), '
        'но больше хороших вин окажется пропущено (recall падает).\n\n'
        '**Если порог уменьшить**, подборка станет шире и в неё попадёт больше хороших вин (recall растёт), '
        'но вместе с ними и больше посредственных (precision падает).')

with tab_batch:
    st.subheader('Оценка нескольких вин из файла')
    batch_threshold = st.session_state['threshold']
    st.caption(f'Используется порог из боковой панели: {batch_threshold:.2f}. '
               f'Нужные столбцы: {", ".join(features)}.')
    uploaded = st.file_uploader('CSV-файл с признаками вин', type='csv')
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception:
            data = None
            st.error('Не получилось прочитать файл. Загрузите CSV-файл с разделителем-запятой.')

        if data is not None:
            missing = [f for f in features if f not in data.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            else:
                try:
                    batch_proba = model.predict_proba(data[features])[:, 1]
                except Exception:
                    batch_proba = None
                    st.error('Не получилось посчитать предсказания. Проверьте, что во всех нужных столбцах '
                             'только числа и нет пустых значений.')

                if batch_proba is not None:
                    result = data.copy()
                    result.insert(0, 'decision', np.where(batch_proba >= batch_threshold, 'хорошее', 'не хорошее'))
                    result.insert(0, 'proba_good', batch_proba)
                    st.dataframe(result)
                    st.download_button('Скачать результат', result.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv')

with tab_coef:
    st.subheader('Коэффициенты логистической регрессии')
    coefs = pd.Series(model[-1].coef_[0], index=features).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coefs.index, coefs.values, color=np.where(coefs.values > 0, 'tab:green', 'tab:red'))
    ax.axvline(0, color='k', linewidth=1)
    ax.set_xlabel('Коэффициент (признаки масштабированы)')
    st.pyplot(fig)
    plt.close(fig)

    top_for = coefs.sort_values(ascending=False).head(3)
    top_against = coefs.head(3)
    st.markdown(
        '**Сильнее всего голосуют за хорошее вино:** ' + ', '.join(f'`{f}` ({v:+.2f})' for f, v in top_for.items())
        + '.\n\n**Сильнее всего голосуют против:** '
        + ', '.join(f'`{f}` ({v:+.2f})' for f, v in top_against.items()) + '.\n\n'
        'Признаки масштабированы, поэтому коэффициент показывает, насколько изменяется логарифм шансов '
        'того, что вино хорошее, при увеличении признака на одно стандартное отклонение.')

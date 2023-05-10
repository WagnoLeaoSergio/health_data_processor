import os
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

INPUT_FOLDER = "data"
OUTPUT_FOLDER = "results"
SENSOR_DATA_FILE = 'sensor_data.csv'
HEALTH_RATE_FILE = "heart_rate.csv"


def preprocess_data():
    dados = pd.read_csv(os.path.join(INPUT_FOLDER, SENSOR_DATA_FILE))
    dados = dados.drop(columns=['Unnamed: 0'])

    dados_bpm = dados[['date', 'heart']]
    dados_bpm = dados_bpm.rename(columns={'heart': 'heart_rate'})
    dados_bpm = dados_bpm.rename(
        columns={'date': 'heart_rate_start_time'})
    dados_bpm = dados_bpm[dados_bpm['heart_rate'] != -1]

    dados_bpm['heart_rate_update_time'] = dados_bpm['heart_rate_start_time']
    dados_bpm['heart_rate_create_time'] = dados_bpm['heart_rate_start_time']
    dados_bpm['heart_rate_end_time'] = dados_bpm['heart_rate_start_time']

    dados_bpm['heart_rate_max'] = dados_bpm['heart_rate']
    dados_bpm['heart_rate_min'] = dados_bpm['heart_rate']

    dados_freq = dados_bpm.reset_index().drop(columns=['index'])
    dias_experimentos = dados_freq['heart_rate_start_time'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date()).unique()

    for dia in dias_experimentos:

        horarios_frequencias = []

        for row in dados_freq.iterrows():

            date_row = datetime.strptime(
                row[1]['heart_rate_start_time'],
                '%Y-%m-%d %H:%M:%S'
            ).date()

            if dia == date_row:
                horarios_frequencias.append(
                    (
                        row[0],
                        datetime.strptime(
                            row[1]['heart_rate_start_time'],
                            '%Y-%m-%d %H:%M:%S'
                        ).strftime('%H:%M:%S'),
                        row[1]['heart_rate']
                    )
                )

        horarios_frequencias.sort(key=lambda x: x[0])

        if len(horarios_frequencias) > 1:
            for previous, current in zip(
                horarios_frequencias,
                horarios_frequencias[1:]
            ):
                max_freq = max(previous[2], current[2])
                min_freq = min(previous[2], current[2])

                dados_freq['heart_rate'][previous[0]] = (
                    max_freq + min_freq) / 2

                dados_freq['heart_rate_max'][previous[0]] = max_freq
                dados_freq['heart_rate_min'][previous[0]] = min_freq

                curr_date = datetime.strptime(
                    dados_freq['heart_rate_start_time'][current[0]],
                    '%Y-%m-%d %H:%M:%S'
                )

                end_date = curr_date - timedelta(minutes=1)
                dados_freq['heart_rate_end_time'][previous[0]] = end_date

    dados_freq.to_csv(
        os.path.join(OUTPUT_FOLDER, HEALTH_RATE_FILE),
        index=False
    )

    return dados_freq


def train_model():
    dados = preprocess_data()
    # dados = pd.read_csv(os.path.join(INPUT_FOLDER, SENSOR_DATA_FILE))
    dados.rename(
        columns={
            "heart_rate_start_time": "inicio",
            "heart_rate": "frequencia",
            "heart_rate_max": "maximo",
            "heart_rate_min": "minimo"
        },
        inplace=True
    )

    dados.drop(["heart_rate_update_time", "heart_rate_create_time",
               "heart_rate_end_time"], axis=1, inplace=True)

    dados["intervalo_min_max"] = dados.maximo - dados.minimo

    dados['aumento_frequencia'] = \
        dados['frequencia'] - dados['frequencia'].shift(-1)

    dados['aceleracao_frequencia'] = \
        dados['aumento_frequencia'] - dados['aumento_frequencia'].shift(-1)

    dados.inicio = pd.to_datetime(dados.inicio)

    dados.set_index("inicio", inplace=True)

    heart_dist = plt.figure(figsize=(15, 6))
    sns.set_style("darkgrid")
    ax = sns.distplot(dados.frequencia)
    ax.set_title("Heart rate distribution", fontdict={"fontsize": 20})
    ax.set_xlabel("Heart Rate", fontdict={"fontsize": 15})
    ax.set_ylabel("Density", fontdict={"fontsize": 15})

    heart_dist.savefig(OUTPUT_FOLDER + "/" + "heart_rate_distribution.png")

    heart_boxplot = plt.figure(figsize=(15, 4))
    sns.set_style("darkgrid")
    sns.boxplot(x=dados.frequencia, data=dados)

    heart_boxplot.savefig(OUTPUT_FOLDER + "/" + "heart_rate_boxplot.png")

    # validation_size = len(dados.index[
    #     dados.index > '2022-11-05 13:02:00'
    # ].to_list())
    validation_size = len(dados) // 2

    labels = ["minimo", "maximo", "aumento_frequencia", "intervalo_min_max"]

    x = dados[labels][:-validation_size]
    y = dados["frequencia"][:-validation_size]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        shuffle=True,
        random_state=7
    )

    x_test.fillna(method='ffill', inplace=True)

    knn = KNeighborsRegressor(
        algorithm="brute",
        n_neighbors=10,
        p=1,
        weights="uniform"
    )

    knn.fit(x_train, y_train)

    dump(
        value=knn,
        filename=os.path.join(OUTPUT_FOLDER, "knn_classifier.joblib"),
        compress=5
    )

    y_previsto_knn = knn.predict(x_test)

    print("A acurácia das predições R² igual a: {}".format(
        metrics.r2_score(y_test, y_previsto_knn).round(5)))

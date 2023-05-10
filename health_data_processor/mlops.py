import os
import time
import mlflow


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


def run_dummy_model():
    db = load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        db.data,
        db.target
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        max_features=3
    )

    rf.fit(X_train, y_train)

    # y_predicted = rf.predict(X_test)
    # acc = accuracy_score(y_test, y_predicted).round(5)
    # mse = mean_squared_error(y_test, y_predicted).round(5)
    # r2 = r2_score(y_test, y_predicted).round(5)

    return


def init():

    now = int(time.time() * 1000)
    user = 'USER1'
    run_name = f"{user}_{now}"
    experiment_name = "prototype_v0"
    experiment_id = "0"

    mlflow.set_tracking_uri('http://127.0.0.1:5000')

    current_experiment = mlflow.search_experiments(
        filter_string=f"name = '{experiment_name}'"
    )

    if len(current_experiment) == 0:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = current_experiment[0].experiment_id

    latest_user_run = mlflow.search_runs(
        filter_string=f"attributes.status = 'FINISHED' and tags.`mlflow.user` = '{user}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    print(latest_user_run)

    with mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
    ) as run:
        mlflow.set_tag("mlflow.user", user)
        mlflow.autolog()
        run_dummy_model()

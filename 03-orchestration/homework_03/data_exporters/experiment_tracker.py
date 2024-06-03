if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import pickle
@data_exporter
def log_model(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    dv = data[3]
    lr = data[4]
    #mlflow.set_tracking_uri("http://mlflow:5000")
    # Serialize the DictVectorizer
    with open("dict_vectorizer.pkl", "wb") as f:
        pickle.dump(dv, f)
  
    mlflow.set_experiment("LinearRegressionMageAI")
    with mlflow.start_run():
        #Log DV
        mlflow.log_artifact("dict_vectorizer.pkl", artifact_path="artifacts")
        print("artifact_loggged")
        #Log ML model Linera Regression
        mlflow.sklearn.log_model(lr, "model")
        print("model logged")

   


"""
    Simple Kubeflow Pipeline.
"""

# --------------
# --- IMPORT ---
# --------------

import sys
from kfp import dsl, compiler, components
from typing import NamedTuple

# -----------------
# --- FUNCTIONS ---
# -----------------

def load_dataset(m: float = 0.5,
                 q: float = 1.0,
                 noise_mean: float = 0,
                 noise_scale: float = 0.1,
                 test_size: float = 0.3,
                 random_state: int = 123,
                 sep: str = ',',
                 minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local",
                 bucket_name: str = "datasets",
                 update_dataset: bool = False):
    """
    Function for data loading.
    """
    # >>> IMPORT
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from minio import Minio

    print("--- Loading dataset ---")

    # >>> CONNECT TO MINIO
    minio_client = Minio(endpoint=minio_endpoint+":9000",
                         access_key="minio",
                         secret_key="minio123",
                         secure=False)
    
    if not minio_client.bucket_exists(bucket_name=bucket_name):
        print("--- Creating bucket {} ---".format(bucket_name))
        minio_client.make_bucket(bucket_name=bucket_name)

    obj_names = list(map(lambda obj: obj.object_name, minio_client.list_objects(bucket_name=bucket_name)))
    if not update_dataset and "simple_pipeline_dataset/" in obj_names:   
        # >>> DOWNLOAD DATASET FROM MINIO
        minio_client.fget_object(bucket_name=bucket_name,
                                    object_name="simple_pipeline_dataset/train_data.csv",
                                    file_path="data/train_data.csv")
        minio_client.fget_object(bucket_name=bucket_name,
                                    object_name="simple_pipeline_dataset/test_data.csv",
                                    file_path="data/test_data.csv")
        print("--- Dataset found ---")
        return
    else:
        print("--- Creating Dataset ---")
        # >>> CREATE DATA
        X = np.linspace(-10, 10, 100)
        Y = m * X + q
        rs = np.random.RandomState(np.random.MT19937(seed = random_state))
        Y += rs.normal(loc = noise_mean, scale = noise_scale, size = Y.shape)

        # >>> TRAIN-TEST SPLIT
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=random_state)
        train_df = pd.DataFrame({"X": X_train, "Y": Y_train})
        test_df = pd.DataFrame({"X": X_test, "Y": Y_test})

        # >>> SAVE DATA LOCALLY
        train_df.to_csv(f"data/train_data.csv", sep=sep, index=False)
        test_df.to_csv(f"data/test_data.csv", sep=sep, index=False)

        # >>> UPLOAD TO MINIO
        minio_client.fput_object(bucket_name = bucket_name, 
                                 object_name = "simple_pipeline_dataset/train_data.csv",
                                 file_path = "data/train_data.csv")
        minio_client.fput_object(bucket_name = bucket_name, 
                                 object_name = "simple_pipeline_dataset/test_data.csv",
                                 file_path = "data/test_data.csv")

        print("--- Data csv saved to MinIO location simple_pipeline_dataset/ ---")

def print_data_desc(sep: str = ',', 
                    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local", 
                    bucket_name: str = "datasets") -> NamedTuple('PrintDataOutput', [('mlpipeline_ui_metadata', 'UI_metadata')]):
    """
    Print Dataset description.
    """
    # >>> IMPORT
    import pandas as pd
    from minio import Minio
    from collections import namedtuple
    import json

    # >>> CONNECT TO MINIO
    minio_client = Minio(endpoint=minio_endpoint+":9000",
                         access_key="minio",
                         secret_key="minio123",
                         secure=False)
    if not minio_client.bucket_exists(bucket_name=bucket_name):
        raise ValueError("{} bucket does not exists.".format(bucket_name))

    # >>> DOWNLOAD DATASET FROM MINIO
    try:
        minio_client.fget_object(bucket_name=bucket_name,
                                    object_name="simple_pipeline_dataset/train_data.csv",
                                    file_path="data/train_data.csv")
        minio_client.fget_object(bucket_name=bucket_name,
                                    object_name="simple_pipeline_dataset/test_data.csv",
                                    file_path="data/test_data.csv")
    except:
        raise RuntimeError("Error while loading dataset.")
    
    # >>> READ DATASET
    train_df = pd.read_csv(f"data/train_data.csv", sep=sep)
    test_df = pd.read_csv(f"data/test_data.csv", sep=sep)
    
    # >>> PRINT DESCRIPTION
    print("--- TRAIN SET ---")
    print("> Info :")
    print(train_df.info())
    print("> Description :")
    print(train_df.describe())

    print("--- TEST SET ---")
    print("> Info :")
    print(test_df.info())
    print("> Description :")
    print(test_df.describe())

    train_head = train_df.head().to_csv(index = False, header = False)
    test_head = test_df.head().to_csv(index = False, header = False)

    metadata = {
        'outputs' : [{
            'type': 'table',
            'format': 'csv',
            'storage': 'inline',
            'header': train_df.columns.to_list(),
            'source': train_head
        },
        {
            'type': 'table',
            'format': 'csv',
            'storage': 'inline',
            'header': test_df.columns.to_list(),
            'source': test_head
        }]
    }
    output = namedtuple('PrintDataOutput', ['mlpipeline_ui_metadata'])
    return output(json.dumps(metadata))

def train_model(sep: str = ',',
                save_model: bool = True,
                model_name: str = "simple_pipeline_model",
                minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local",
                data_bucket: str = "datasets",
                model_bucket: str = "models") -> NamedTuple('TrainModelOutput', [('mlpipeline_ui_metadata', 'UI_metadata')]):
    """
    Train a Linear Regression model.
    """
    # >>> IMPORT
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from minio import Minio
    import joblib
    import plotly.graph_objects as go
    from collections import namedtuple
    import json

    # >>> CONNECT TO MINIO
    minio_client = Minio(endpoint=minio_endpoint+":9000",
                         access_key="minio",
                         secret_key="minio123",
                         secure=False)
    if not minio_client.bucket_exists(bucket_name=data_bucket):
        raise ValueError("{} bucket does not exists.".format(data_bucket))

    # >>> DOWNLOAD DATASET FROM MINIO
    try:
        minio_client.fget_object(bucket_name=data_bucket,
                                    object_name="simple_pipeline_dataset/train_data.csv",
                                    file_path="data/train_data.csv")
    except:
        raise RuntimeError("Error while loading dataset.")

    # >>> SPLIT DATA
    train_df = pd.read_csv(f"data/train_data.csv", sep=sep)
    X_train = train_df.loc[:,"X"].values.reshape(-1, 1)
    Y_train = train_df.loc[:,"Y"].values.reshape(-1, 1)

    # >>> TRAIN MODEL
    model = LinearRegression()
    model.fit(X=X_train, y=Y_train)

    # >>> SAVE MODEL
    if save_model:
        # Dump model
        model_filename = "data/" + model_name + ".sav"
        joblib.dump(model, filename=model_filename)
        # Upload to MinIO
        if not minio_client.bucket_exists(bucket_name=model_bucket):
            print("--- Creating bucket {} ---".format(model_bucket))
            minio_client.make_bucket(bucket_name=model_bucket)
        try:
            minio_client.fput_object(bucket_name = model_bucket, 
                                    object_name = "simple_pipeline/"+ model_name + ".sav",
                                    file_path = model_filename)
        except:
            raise RuntimeError("Error while uploading model.") 
        print("--- Model saved to MinIO location simple_pipeline_dataset/ ---")

    # >>> VISUALIZE MODEL
    score = model.score(X_train, Y_train)
    Y_pred = model.predict(X_train)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train[:,0], y=Y_train[:,0], mode="markers", name="train data"))
    fig.add_trace(go.Scatter(x=X_train[:,0], y=Y_pred[:,0], mode="lines", name="model"))
    fig.update_layout({
        'title': "Model on train set (R2: {:.2f})".format(score),
        'xaxis_title': "X",
        'yaxis_title': "Y"
    })
    fig.write_html("data/train_plot.html")
    minio_client.fput_object(bucket_name = model_bucket, 
                             object_name = "simple_pipeline/train_plot.html",
                             file_path = "data/train_plot.html")
    metadata = {
        'outputs' : [{
            'source': "minio://{}/simple_pipeline/train_plot.html".format(model_bucket),
            'type': 'web-app',
    }]}
    output = namedtuple('TrainModelOutput', ['mlpipeline_ui_metadata'])
    return output(json.dumps(metadata))

def eval_model(minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local",
               data_bucket: str = "datasets",
               sep: str = ',',
               model_bucket: str = "models",
               model_name: str = "simple_pipeline_model") -> NamedTuple('EvalModelOutput', [('mlpipeline_ui_metadata', 'UI_metadata'), ('mlpipeline_metrics', 'Metrics')]):
    """
    Evaluate model on test set.
    """
    # >>> IMPORT
    from minio import Minio
    import pandas as pd
    import joblib
    import plotly.graph_objects as go
    from collections import namedtuple
    import json

    # >>> CONNECT TO MINIO
    minio_client = Minio(endpoint=minio_endpoint+":9000",
                         access_key="minio",
                         secret_key="minio123",
                         secure=False)
    if not minio_client.bucket_exists(bucket_name=data_bucket):
        raise ValueError("{} bucket does not exists.".format(data_bucket))
    if not minio_client.bucket_exists(bucket_name=model_bucket):
        raise ValueError("{} bucket does not exists.".format(model_bucket))

    # >>> DOWNLOAD DATASET AND MODEL FROM MINIO
    try:
        minio_client.fget_object(bucket_name=data_bucket,
                                 object_name="simple_pipeline_dataset/test_data.csv",
                                 file_path="data/test_data.csv")
        minio_client.fget_object(bucket_name=model_bucket,
                                 object_name="simple_pipeline/"+ model_name + ".sav",
                                 file_path="data/" + model_name + ".sav")
    except:
        raise RuntimeError("Error while loading dataset and model.")

    # >>> SPLIT DATA
    test_df = pd.read_csv(f"data/test_data.csv", sep=sep)
    X_test = test_df.loc[:,"X"].values.reshape(-1, 1)
    Y_test = test_df.loc[:,"Y"].values.reshape(-1, 1)

    # >>> LOAD MODEL
    model = joblib.load(filename="data/" + model_name + ".sav")

    # >>> EVAL MODEL
    score = model.score(X=X_test, y=Y_test)
    Y_pred = model.predict(X_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test[:,0], y=Y_test[:,0], mode="markers", name="test data"))
    fig.add_trace(go.Scatter(x=X_test[:,0], y=Y_pred[:,0], mode="lines", name="model"))
    fig.update_layout({
        'title': "Model on test set (R2: {:.2f})".format(score),
        'xaxis_title': "X",
        'yaxis_title': "Y"
    })
    fig.write_html("data/test_plot.html")
    minio_client.fput_object(bucket_name = model_bucket, 
                             object_name = "simple_pipeline/test_plot.html",
                             file_path = "data/test_plot.html")
    metadata = {
        'outputs' : [{
            'source': "minio://{}/simple_pipeline/test_plot.html".format(model_bucket),
            'type': 'web-app',
    }]}

    # >>> LOG METRICS
    print("Score on test set: {:.2f} %".format(score * 100))
    metrics = {
      'metrics': [{
          'name': 'test-r2-score',
          'numberValue':  float(score),
          'format' : "PERCENTAGE"
        }]}
    
    output = namedtuple('EvalModelOutput', ['mlpipeline_ui_metadat', 'mlpipeline_metrics'])
    return output(json.dumps(metadata), json.dumps(metrics))

# ----------------
# --- PIPELINE ---
# ----------------

# > COMPONENTS
load_dataset_component = components.create_component_from_func(
    func = load_dataset,
    base_image = "python:3.8",
    packages_to_install = ["pandas==2.0.0", 
                            "numpy==1.24.2", 
                            "scikit-learn==1.2.2",
                            "minio==7.1.14"]
)
print_data_desc_component = components.create_component_from_func(
    func = print_data_desc,
    base_image = "python:3.8",
    packages_to_install = ["pandas==2.0.0", 
                           "minio==7.1.14"]
)
train_model_component = components.create_component_from_func(
    func = train_model,
    base_image = "python:3.8",
    packages_to_install = ["pandas==2.0.0", 
                           "numpy==1.24.2", 
                           "scikit-learn==1.2.2", 
                           "plotly==5.14.1",
                           "minio==7.1.14",
                           "joblib==1.2.0"] 
)
eval_model_component = components.create_component_from_func(
    func = eval_model,
    base_image = "python:3.8",
    packages_to_install = ["pandas==2.0.0",
                           "plotly==5.14.1",
                           "minio==7.1.14",
                           "scikit-learn==1.2.2",
                           "joblib==1.2.0"]
)

# > PIPELINE
@dsl.pipeline(
        name="simple-pipeline",
        description="Simple pipeline example."
)
def simple_pipeline(attach_volume: bool = True,
                    data_path: str = "/data",
                    m: float = 0.5,
                    q: float = 1.0,
                    noise_mean: float = 0,
                    noise_scale: float = 0.1,
                    test_size: float = 0.3,
                    random_state: int = 123,
                    update_dataset: bool = False,
                    sep: str = ',',
                    save_model: bool = True,
                    model_name: str = "simple_pipeline_model",
                    minio_endpoint: str = "minio-service.kubeflow.svc.cluster.local",
                    data_bucket: str = "datasets",
                    model_bucket: str = "models"):
    """
    Pipeline function
    """
    # >>> Create a volume for this pipeline
    if attach_volume:
        vop = dsl.VolumeOp(name = "simple_pipeline_volume", 
                            resource_name = "simple_pipeline_volume",
                            size = "1Gi",
                            modes = dsl.VOLUME_MODE_RWO)
        # >>> Run tasks
        load_data_task = load_dataset_component(m = m, 
                                                q = q,
                                                noise_mean = noise_mean,
                                                noise_scale = noise_scale,
                                                test_size = test_size,
                                                random_state = random_state,
                                                sep = sep,
                                                minio_endpoint = minio_endpoint,
                                                bucket_name = data_bucket,
                                                update_dataset = update_dataset).add_pvolumes({data_path: vop.volume})
        print_desc_task = print_data_desc_component(sep = sep,
                                                    minio_endpoint = minio_endpoint,
                                                    bucket_name = data_bucket).add_pvolumes({data_path: vop.volume}).after(load_data_task)
        train_model_task = train_model_component(sep = sep,
                                                save_model = save_model,
                                                model_name = model_name,
                                                minio_endpoint = minio_endpoint,
                                                data_bucket = data_bucket,
                                                model_bucket = model_bucket).add_pvolumes({data_path: vop.volume}).after(load_data_task)
        eval_model_task = eval_model_component(sep = sep,
                                                model_name = model_name,
                                                minio_endpoint = minio_endpoint,
                                                data_bucket = data_bucket,
                                                model_bucket = model_bucket).add_pvolumes({data_path: vop.volume}).after(train_model_task)
    else:
        # >>> Run tasks
        load_data_task = load_dataset_component(m = m, 
                                                q = q,
                                                noise_mean = noise_mean,
                                                noise_scale = noise_scale,
                                                test_size = test_size,
                                                random_state = random_state,
                                                sep = sep,
                                                minio_endpoint = minio_endpoint,
                                                bucket_name = data_bucket,
                                                update_dataset = update_dataset)
        print_desc_task = print_data_desc_component(sep = sep,
                                                    minio_endpoint = minio_endpoint,
                                                    bucket_name = data_bucket).after(load_data_task)
        train_model_task = train_model_component(sep = sep,
                                                save_model = save_model,
                                                model_name = model_name,
                                                minio_endpoint = minio_endpoint,
                                                data_bucket = data_bucket,
                                                model_bucket = model_bucket).after(load_data_task)
        eval_model_task = eval_model_component(sep = sep,
                                                model_name = model_name,
                                                minio_endpoint = minio_endpoint,
                                                data_bucket = data_bucket,
                                                model_bucket = model_bucket).after(train_model_task)
    
    # >>> Disable caching
    load_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    print_desc_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    eval_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    

if __name__ == "__main__":
    a_count = len(sys.argv)
    if a_count > 1:
        mode = sys.argv[1]
    else:
        mode = "test"
    if mode == "test":
        # >>> Run test
        minio_endpoint = "127.0.0.1"
        load_dataset(minio_endpoint = minio_endpoint)
        print_data_desc(minio_endpoint = minio_endpoint)
        train_model(minio_endpoint = minio_endpoint)
        eval_model(minio_endpoint = minio_endpoint)
    elif mode == "compile":
        # >>> Compile pipeline
        compiler.Compiler().compile(
            pipeline_func=simple_pipeline,
            package_path='pipelines/SimplePipeline.yaml'
        )
    else:
        print("USAGE: {} [OPTION]".format(sys.argv[0]))
        print("OPTIONS:\n \ttest: Run test.\n \tcompile Compile pipeline")

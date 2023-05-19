"""
    End-to-End Kubeflow Pipeline, from data loading to model serving.
"""

# TODOs: 
#   - add minio_access_key and minio_secret_key as parameter for all components

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
                 minio_endpoint: str = "minio-service.kubeflow",
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
                    minio_endpoint: str = "minio-service.kubeflow", 
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
                minio_endpoint: str = "minio-service.kubeflow",
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
        model_filename = "data/" + model_name + ".joblib"
        joblib.dump(model, filename=model_filename)
        # Upload to MinIO
        if not minio_client.bucket_exists(bucket_name=model_bucket):
            print("--- Creating bucket {} ---".format(model_bucket))
            minio_client.make_bucket(bucket_name=model_bucket)
        try:
            minio_client.fput_object(bucket_name = model_bucket, 
                                    object_name = "simple_pipeline/"+ model_name + ".joblib",
                                    file_path = model_filename)
        except:
            raise RuntimeError("Error while uploading model.") 
        print("--- Model saved to MinIO location simple_pipeline/ ---")

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

def eval_model(minio_endpoint: str = "minio-service.kubeflow",
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
                                 object_name="simple_pipeline/"+ model_name + ".joblib",
                                 file_path="data/" + model_name + ".joblib")
    except:
        raise RuntimeError("Error while loading dataset and model.")

    # >>> SPLIT DATA
    test_df = pd.read_csv(f"data/test_data.csv", sep=sep)
    X_test = test_df.loc[:,"X"].values.reshape(-1, 1)
    Y_test = test_df.loc[:,"Y"].values.reshape(-1, 1)

    # >>> LOAD MODEL
    model = joblib.load(filename="data/" + model_name + ".joblib")

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

def serve_model(service_name: str = "simple-serving",
                add_service_version: bool = True,
                namespace: str = None,
                kserve_version: str = "v1beta1",
                model_bucket: str = "models",
                model_name: str = "simple_pipeline_model",
                service_account_name: str = "kserve-service-credentials",
                update_credentials: bool = False,
                minio_endpoint: str = "minio-service.kubeflow",
                minio_access_key: str = "minio",
                minio_secret_key: str = "minio123",
                access_key_name: str = None,
                secret_key_name: str = None,
                profile: str = "default",
                use_https: bool = False,
                verify_ssl: bool = False):
    """
    Create an InferenceService for the trained model.
    """
    # >>> IMPORT
    from kserve import (
        KServeClient, 
        utils, 
        constants, 
        V1beta1SKLearnSpec, 
        V1beta1PredictorSpec,
        V1beta1InferenceServiceSpec,
        V1beta1InferenceService
    )
    from kubernetes import client
    import base64
    import datetime

    KServe = KServeClient()
    if namespace == None:
        namespace = utils.get_default_target_namespace()

    # >>> MINIO CREDENTIALS
    if update_credentials:
        # Encode access and secret key in base64
        b64_access_key = base64.b64encode(minio_access_key.encode("ascii"))
        b64_access_key = b64_access_key.decode("ascii")
        b64_secret = base64.b64encode(minio_secret_key.encode("ascii"))
        b64_secret = b64_secret.decode("ascii")
        # Write credentials
        if access_key_name == None or access_key_name == "":
            access_key_name = constants.S3_ACCESS_KEY_ID_DEFAULT_NAME
        if secret_key_name == None or secret_key_name == "":
            secret_key_name = constants.S3_SECRET_ACCESS_KEY_DEFAULT_NAME
        credentials = "[{profile}]\n{access_key_name}={access_key}\n{secret_key_name}={secret_key}".format(
            profile = profile,
            access_key_name = access_key_name,
            access_key = b64_access_key, 
            secret_key_name = secret_key_name,
            secret_key = b64_secret)
        with open("data/credentials", "wt") as cred_file:
            cred_file.write(credentials)
        KServe.set_credentials(storage_type='S3',
                                namespace=namespace,
                                credentials_file='data/credentials',
                                service_account=service_account_name,
                                s3_profile=profile,
                                s3_endpoint=minio_endpoint+":9000",
                                s3_use_https='1' if use_https else '0',
                                s3_verify_ssl='1' if verify_ssl else '0')
        
    # >>> INFERENCE SERVICE OPTIONS
    api_version = constants.KSERVE_GROUP + '/' + kserve_version
    storage_uri = "s3://{}/simple_pipeline/{}.joblib".format(model_bucket, model_name)
    if add_service_version:
        service_name = service_name + "-" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    # >>> INFERENCE SERVICE DEFINITION
    metadata = client.V1ObjectMeta(name = service_name, namespace = namespace)
    sklearn_spec = V1beta1SKLearnSpec(storage_uri = storage_uri)
    predictor = V1beta1PredictorSpec(sklearn = sklearn_spec, service_account_name = service_account_name)
    spec = V1beta1InferenceServiceSpec(predictor = predictor)
    isvc = V1beta1InferenceService(api_version = api_version,
                                   kind = constants.KSERVE_KIND,
                                   metadata = metadata,
                                   spec = spec)
    
    # >>> INFERENCE SERVICE CREATION
    info = KServe.create(isvc)
    print(info)

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
serve_model_component = components.create_component_from_func(
    func = serve_model,
    base_image = "python:3.8",
    packages_to_install = ["kserve==0.10.1",
                           "kubernetes==25.3.0"]
)

# > PIPELINE
@dsl.pipeline(
        name="simple-pipeline",
        description="Simple pipeline example."
)
def simple_pipeline(m: float = 0.5,
                    q: float = 1.0,
                    noise_mean: float = 0,
                    noise_scale: float = 0.1,
                    test_size: float = 0.3,
                    random_state: int = 123,
                    update_dataset: bool = False,
                    sep: str = ',',
                    save_model: bool = True,
                    model_name: str = "simple_pipeline_model",
                    minio_endpoint: str = "minio-service.kubeflow",
                    data_bucket: str = "datasets",
                    model_bucket: str = "models",
                    inference_service_name: str = "simple-serving",
                    add_inference_service_version: bool = True,
                    namespace: str = None,
                    kserve_version: str = "v1beta1",
                    service_account_name: str = "kserve-service-credentials",
                    update_credentials: bool = False,
                    access_key_name: str = None,
                    secret_key_name: str = None,
                    profile: str = "default",
                    use_https: bool = False,
                    verify_ssl: bool = False):
    """
    Pipeline function
    """
    # >>> Create a volume for this pipeline
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
                                            update_dataset = update_dataset).add_pvolumes({"/data": vop.volume})
    print_desc_task = print_data_desc_component(sep = sep,
                                                minio_endpoint = minio_endpoint,
                                                bucket_name = data_bucket).add_pvolumes({"/data": vop.volume}).after(load_data_task)
    train_model_task = train_model_component(sep = sep,
                                            save_model = save_model,
                                            model_name = model_name,
                                            minio_endpoint = minio_endpoint,
                                            data_bucket = data_bucket,
                                            model_bucket = model_bucket).add_pvolumes({"/data": vop.volume}).after(load_data_task)
    eval_model_task = eval_model_component(sep = sep,
                                            model_name = model_name,
                                            minio_endpoint = minio_endpoint,
                                            data_bucket = data_bucket,
                                            model_bucket = model_bucket).add_pvolumes({"/data": vop.volume}).after(train_model_task)
    serve_model_task = serve_model_component(service_name = inference_service_name,
                                             add_service_version = add_inference_service_version,
                                             namespace = namespace,
                                             kserve_version = kserve_version,
                                             model_bucket = model_bucket,
                                             model_name = model_name,
                                             service_account_name = service_account_name,
                                             update_credentials = update_credentials,
                                             minio_endpoint = minio_endpoint,
                                             minio_access_key = "minio",
                                             minio_secret_key = "minio123",
                                             access_key_name = access_key_name,
                                             secret_key_name = secret_key_name,
                                             profile = profile,
                                             use_https = use_https,
                                             verify_ssl = verify_ssl).add_pvolumes({"/data": vop.volume}).after(eval_model_task)
    # >>> Disable caching
    load_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    print_desc_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    eval_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    serve_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    

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
        # serve_model(minio_endpoint = minio_endpoint)
    elif mode == "compile":
        # >>> Compile pipeline
        compiler.Compiler().compile(
            pipeline_func=simple_pipeline,
            package_path='pipelines/SimplePipeline.yaml'
        )
    else:
        print("USAGE: {} [OPTION]".format(sys.argv[0]))
        print("OPTIONS:\n \ttest: Run test.\n \tcompile Compile pipeline")

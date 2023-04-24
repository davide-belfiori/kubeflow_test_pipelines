"""
    Kubeflow Pipeline for MinIO connection test.
"""

# --------------
# --- IMPORT ---
# --------------

import kfp
import sys

# -----------------
# --- CONSTANTS ---
# -----------------

PIPELINE_NAME = "minio-connection-test"
PIPELINE_DESC = "MinIO connection test pipeline"
PIPELINE_FILENAME = "MinIOConnectionTest"

# ------------------
# --- CONN. TEST ---
# ------------------

def test_connection(endpoint_addr: str = "minio-service.kubeflow.svc.cluster.local",
                    endpoint_port: str = "9000",
                    access_key: str = "minio",
                    secret_key: str = "minio123",
                    secure: bool = False,
                    remove_tmp_bucket: bool = True):
    # > IMPORT
    from minio import Minio
    from datetime import datetime
    # > MINIO CLIENT
    minio_client = Minio(
        endpoint = endpoint_addr + ":" + endpoint_port,
        access_key = access_key,
        secret_key = secret_key,
        secure = secure
    )
    # > CREATE AN EMPTY BUCKET
    dt = datetime.now().strftime("%m%d%Y-%H%M%S")
    bucket_name = "tmp-bucket-"+dt
    print("--- Creating bucket {} ---".format(bucket_name))
    minio_client.make_bucket(bucket_name=bucket_name)
    # > REMOVE BUCKET
    if remove_tmp_bucket:
        print("--- Removing bucket {} ---".format(bucket_name))
        minio_client.remove_bucket(bucket_name=bucket_name)

# ----------------
# --- PIPELINE ---
# ----------------

# > DEFINE COMPONENTS
test_connection_component = kfp.components.create_component_from_func(
    func = test_connection,
    base_image = "python:3.8",
    packages_to_install = ["minio==7.1.14"]
)

# > PIPELINE
@kfp.dsl.pipeline(
    name = PIPELINE_NAME,
    description = PIPELINE_DESC
)
def test_connection_pipeline(endpoint_addr: str = "minio-service.kubeflow.svc.cluster.local",
                             endpoint_port: str = "9000",
                             access_key: str = "minio",
                             secret_key: str = "minio123",
                             secure: bool = False,
                             remove_tmp_bucket: bool = True):
    # > RUN TASKS
    test_connection_task = test_connection_component(endpoint_addr = endpoint_addr,
                                                     endpoint_port = endpoint_port,
                                                     access_key = access_key,
                                                     secret_key = secret_key,
                                                     secure = secure,
                                                     remove_tmp_bucket = remove_tmp_bucket)
    # > DISABLE CACHING
    test_connection_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

if __name__ == "__main__":
    a_count = len(sys.argv)
    if a_count > 1:
        mode = sys.argv[1]
    else:
        mode = "test"
    if mode == "test":
        # >>> Run test
        minio_endpoint = "127.0.0.1"
        test_connection(endpoint_addr=minio_endpoint)
    elif mode == "compile":
        # >>> Compile pipeline
        kfp.compiler.Compiler().compile(
            pipeline_func=test_connection_pipeline,
            package_path='pipelines/' + PIPELINE_FILENAME + '.yaml'
        )
    else:
        print("USAGE: {} [OPTION]".format(sys.argv[0]))
        print("OPTIONS:\n \ttest: Run test.\n \tcompile Compile pipeline")
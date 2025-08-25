from setuptools import setup, find_packages

setup(
    name='chesssimilarity',
    version='0.1',
    packages=find_packages(where='src') + find_packages(where='src/lichess_data_loading') + find_packages(where='src/data') + find_packages(where='src/model_utils') + find_packages(where='src/mlflow_demo') + find_packages(where='src/api'),
    package_dir={'': 'src'},
)
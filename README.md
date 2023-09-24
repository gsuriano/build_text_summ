## Jenkins CI/CD integration with SageMaker preprocessing, training and registration  

The template provides a starting point for bringing your SageMaker Pipeline development to production.

```
|-- CONTRIBUTING.md
|-- jenkins
|   |-- Jenkinsfile
|   `-- seed_job.groovy
|-- pipelines
|   |-- project
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |   `-- train.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

## Start here
This project focuses on implementing extractive text summarization using the Thext architecture, a state-of-the-art model for scientific paper summarization. I have fine-tuned this model on the CNN/Dailymail dataset from HuggingFace to adapt it for extractive summarization on a new dataset.

A description of some of the artifacts is provided below:
<br/><br/>
Your jenkins pipeline building and execution instructions. File `seed_job.groovy` contains the instructions for creation of jenkins pipeline using `Jenkinsfile`. `Jenkinsfile` contains pipeline definition for stages in jenkins pipeline, and you can modify it to add/delete/update stages in the pipeline. 

```
|-- jenkins
|   |-- Jenkinsfile
|   `-- seed_job.groovy
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- project
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```


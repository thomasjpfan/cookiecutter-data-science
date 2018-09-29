# {{ cookiecutter.project_name }}

Insert description of project.

## Usage

```bash
pip install -r requirements.txt
./download.sh
python model_linear.py train -id latest
python model_linear.py predict -id latest
```

## Observing experiments

### Using Neptune to log experiments

Install neptune: `pip install neptune-cli`. To upload metrics to neptune append `neptune run` to commands:

```bash
neptune run --open-webbrowser false model_linear.py train -id latest
```

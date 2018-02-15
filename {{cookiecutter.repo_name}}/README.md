# {{ cookiecutter.project_name }}

Insert description of project.

## Usage

1. Install requirements

```bash
pip install -r requirements.txt
```

2. Download and uncompress data

```bash
./dl_extract.sh
```

3. Create features

```bash
python src/features.py
```

4. Train Model

```bash
python src/run_constant_model.py train with run_id=latest
```

5. Make Predictions

```bash
python src/run_constant_model.py predict with run_id=latest
```

## Jupyter Notebook

To run the jupyter notebooks, set `PYTHONPATH` to include the `src` directory:

```bash
PYTHONPATH=$PWD/src jupyter notebook --no-browser --port=8999
```

## Using MongoDB to store experiment results

Add a `.envrc` file and use `direnv`!

```bash
# .envrc
export MONGODB_URL=mongodb://test:abc@localhost:27017/boo?authMechanism=SCRAM-SHA-1
export MONGODB_NAME=boo
```

```bash
source .envrc
pip install pymongo
```

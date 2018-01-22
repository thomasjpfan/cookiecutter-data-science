# {{ cookiecutter.project_name }}

Insert description of project.

## Usage

Install requirements

```bash
pip install -r requirements.txt
```

Download and uncompress data

```bash
./dl_extract.sh
```

Create features

```bash
python src/features.py
```

Train Model

```bash
python src/exp_random.py train with run_id=latest
```

Make Predictions

```bash
python src/exp_random.py predict with run_id=latest
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

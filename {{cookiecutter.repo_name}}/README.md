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

Extract, transform, and save data.

```bash
python src/ets.py
```

Train Model

```bash
python src/exp_example.py train
```

Make Predictions

```bash
python src/exp_example.py predict with run_dir=FOLDERNAME
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

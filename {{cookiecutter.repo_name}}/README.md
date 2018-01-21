# {{ cookiecutter.project_name }}

Insert description of project.

## Usage

1. Install requirements
1. Download and uncompress data
1. Extract, transform, and save data
1. Create features
1. Train model
1. Make predictions

```bash
make requirements
make download_extract
make ets
make features
make train
make predict
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

# {{ cookiecutter.project_name }}



## Instructions

1. Get data

1. Create and source .envrc if you want to use mongodb

```bash
# .envrc
export MONGODB_URL=mongodb://test:abc@localhost:27017/boo?authMechanism=SCRAM-SHA-1
export MONGODB_NAME=boo
```

```bash
source .envrc
pip install pymongo
```

1. Install requirements

```bash
make requirements
```

1. Clean data

```bash
make clean_data
```

1. Create features

```bash
make features
```

1. Train model(s)

```bash
make train
```

1. Predict

```bash
make predict
```

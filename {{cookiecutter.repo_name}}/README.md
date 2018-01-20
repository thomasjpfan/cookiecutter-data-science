# {{ cookiecutter.project_name }}



## Instructions

1. Get data

2. Create and source .envrc if you want to use mongodb

```bash
# .envrc
export MONGODB_URL=mongodb://test:abc@localhost:27017/boo?authMechanism=SCRAM-SHA-1
export MONGODB_NAME=boo
```

```bash
source .envrc
pip install pymongo
```

3. Install requirements

```bash
make requirements
```

4. Extract Transform Save

```bash
make ets
```

5. Create features

```bash
make features
```

6. Train model(s)

```bash
make train
```

7. Predict

```bash
make predict
```

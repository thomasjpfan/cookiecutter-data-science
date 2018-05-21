# {{ cookiecutter.project_name }}

Insert description of project.

## Usage

```bash
pip install -r requirements.txt
./download.sh
python src/01-process.py
python src/02-run_linear_model.py train with run_id=latest
python src/02-run_linear_model.py predict with run_id=latest
```

## Observing experiments

### Using MongoDB to store experiment results

Install mongodb: `pip install pymongo`. Then add a `.envrc` file and use `direnv`!

```bash
# .envrc
export MONGODB_URL=mongodb://test:abc@localhost:27017/boo?authMechanism=SCRAM-SHA-1
export MONGODB_NAME=boo
```

### Sacredboard to view experiments

To run sacredboard:

```bash
sacredboard -mu $MONGODB_URL $MONGODB_NAME
```

### Using Pushover for notifications

Install notifiers: `pip install notifiers`. Add notifier env vars to `.envrc`:

```bash
# .envrc
export NOTIFIERS_PUSHOVER_TOKEN=FOO
export NOTIFIERS_PUSHOVER_USER=BAR
```

Logs with level `INFO` and higher will be logged to pushover.

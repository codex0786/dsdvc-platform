PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: storage-status migrate-storage venv install test test-fast api dashboard legacy-v3 legacy-v2 docker-up docker-down admin-key demo

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -r requirements.txt && $(PIP) install -e .

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

api:
	$(PY) -m uvicorn api.main:app --reload

dashboard:
	.venv/bin/streamlit run dashboard/app.py

legacy-v3:
	.venv/bin/streamlit run legacy/run_v3_app.py

legacy-v2:
	cd legacy/dsdvc_mp_v2 && ../../.venv/bin/streamlit run app.py

admin-key:
	@$(PY) -c "import secrets;print('dsdvc_'+secrets.token_hex(6)+'_'+secrets.token_urlsafe(32))"

demo:
	$(PY) scripts/demo_research.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

storage-status:
	$(PY) -m dsdvc.storage status

migrate-storage:   # copy locally stored datasets into S3_BUCKET (local files are kept)
	$(PY) -m dsdvc.storage migrate

.PHONY: eval eval-subset serve up down

eval:
	uv run python -m eval.run --corpus full --sets hard main rare multi langsmith versions --rerank --versions --gate

eval-subset:
	uv run python -m eval.run --corpus subset

serve:
	uv run uvicorn rag.api:app --port 8000

up:
	docker compose up -d --build

down:
	docker compose down

.PHONY: eval eval-subset serve

eval:
	uv run python -m eval.run --corpus full --sets hard main rare --rerank --gate

eval-subset:
	uv run python -m eval.run --corpus subset

serve:
	uv run uvicorn rag.api:app --port 8000

.PHONY: eval eval-subset

eval:
	uv run python -m eval.run --corpus full --sets hard main --rerank --gate

eval-subset:
	uv run python -m eval.run --corpus subset

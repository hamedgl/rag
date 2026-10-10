.PHONY: eval eval-subset

eval:
	uv run python -m eval.run --corpus full --sets hard main rare --rerank --gate

eval-subset:
	uv run python -m eval.run --corpus subset

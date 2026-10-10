eval:
	uv run python -m eval.run --corpus full --sets hard main --gate

eval-subset:
	uv run python -m eval.run --corpus subset

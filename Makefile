.PHONY: install test lint benchmark notebook clean
install:
	python -m pip install -e ".[notebook,test]"
test:
	python -m pytest --cov=fastmatrix --cov-report=term-missing
lint:
	python -m ruff check .
benchmark:
	fastmatrix-benchmark --backend auto
notebook:
	python -m jupyter lab
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist *.egg-info

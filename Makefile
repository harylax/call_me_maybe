# export HF_HOME=/goinfre/haryandr/hf_cache
# export UV_PROJECT_ENVIRONMENT=/goinfre/haryandr/.venv
# export UV_CACHE_DIR=/goinfre/haryandr/uv_cache
export HF_HOME=/sgoinfre/haryandr/hf_cache
export UV_PROJECT_ENVIRONMENT=/sgoinfre/haryandr/.venv
export UV_CACHE_DIR=/sgoinfre/haryandr/uv_cache

.PHONY: install run clean


$(UV_PROJECT_ENVIRONMENT)/.installed:
	mkdir $(HF_HOME) $(UV_PROJECT_ENVIRONMENT) $(UV_CACHE_DIR)
	uv sync
	touch $(UV_PROJECT_ENVIRONMENT)/.installed

install: $(UV_PROJECT_ENVIRONMENT)/.installed

run: install
	uv run python3 -m src

clean:
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +

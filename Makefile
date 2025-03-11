.PHONY: setup test train evaluate demo clean coverage publish-sentiment update-model-card

# Setup environment
setup:
	chmod +x setup_env.sh
	./setup_env.sh

# Create test sets
test-set:
	python -m src.data.sentiment_create_test_set

# Train the model
train:
	python -m src.model.sentiment_train

# Evaluate the model
evaluate:
	python -m src.model.evaluate --test_file data/processed/sentiment_test_examples.json

# Run the demo
demo:
	python -m src.demo

# Interactive demo
demo-interactive:
	python -m src.demo --interactive

# Publish sentiment model to Hugging Face
publish-sentiment:
	@if [ -z "$(REPO_NAME)" ]; then \
		echo "Error: REPO_NAME is required (e.g., make publish-sentiment REPO_NAME=username/model-name)"; \
		exit 1; \
	fi
	python -m src.model.sentiment_publish --repo_name="$(REPO_NAME)"

# Update model card on Hugging Face
update-model-card:
	@if [ -z "$(REPO_NAME)" ]; then \
		echo "Error: REPO_NAME is required (e.g., make update-model-card REPO_NAME=username/model-name)"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_CARD)" ]; then \
		MODEL_CARD="model_card.md"; \
	fi
	python -m src.model.update_model_card --repo_name="$(REPO_NAME)" --model_card="$(MODEL_CARD)"

# Run tests
test:
	pytest

# Run tests with coverage
coverage:
	pytest --cov=src --cov-report=html
	@echo "HTML coverage report generated in htmlcov/ directory"
	@echo "View the report by opening htmlcov/index.html in your browser"
	@echo "On Linux: xdg-open htmlcov/index.html"
	@echo "On macOS: open htmlcov/index.html"
	@echo "On Windows: start htmlcov/index.html"

# Clean build artifacts
clean:
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

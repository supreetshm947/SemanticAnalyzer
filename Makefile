
DATASET_NAME = lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
OUTPUT_FILE = imdb-dataset.zip

setup_env:
	@echo "Setting up the environment..."
	pip install -r requirements.txt
	mkdir -p ./models

download_data:
	@echo "Downloading dataset from Kaggle..."
	mkdir -p ./data
	kaggle datasets download -d $(DATASET_NAME) -o $(OUTPUT_FILE)
	mv $(OUTPUT_FILE) ./data/

setup_modelbit:
	modelbit clone

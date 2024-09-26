
DATASET_NAME = lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
OUTPUT_FILE = imdb-dataset.zip

setup_env:
	pip install -r requirements.txt

download_data:
	@echo "Downloading dataset from Kaggle..."
	mkdir -p ./data
	kaggle datasets download -d $(DATASET_NAME) -o $(OUTPUT_FILE)
	mv $(OUTPUT_FILE) ./data/



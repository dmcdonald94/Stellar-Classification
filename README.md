# Stellar-Classification
TL/DR: Notebooks w/ ML/AI stuff:
- SDSS Photometric Classification.ipynb
- SDSS Image Classification - Modelling - Banded.ipynb
- Clustering.ipynb

This repo will be using data from the Sloan Digital Sky Survey (SDSS) to perform classification on the observed objects into three separate categories: `STAR`, `GALAXY` and `QSO`. This classification is completed using two separate, but related, datasets. The tabular photometric data for the `100,000` objects can be found at [this kaggle link](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17). Furthermore, I pull and associate the images within this dataset from the SDSS API to perform classification using the images used to derive the photometric data in the linked dataset. 

In summary, this is a two stage analysis:

1. Photometric Classification
2. Image Classification

Eventually, these classification algorithms would be combined into a single analysis to produces better results, but that lies beyond the scope of this project (for now...)
## How this repo is organized:

- Photometric Classification
    - SDSS Photometric Classification.ipynb
        - Performs the EDA + Modeling + Model Analysis using the tabular photometric data.
- Image Classification
    - SDSS_Images.py/.ipynb
        - Associates each of the objects (rows) within the photometric data set with an image from the SDSS API
        - Performs background subtract / noise reduction
        - Creates image "chips" (i.e. cutouts of the specific objects of interest.) and saves them to `./chips`
    - SDSS Image Classification - Modelling - Banded.ipynb
        - Reads in previously generated image chips
        - Creates and tunes a Convolutional Neural Network to classify objects.










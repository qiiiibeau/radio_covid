# Project RadioCovid

## Presentation

This repository contains the code for our project **RADIO_COVID**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The objective of the project is to diagnose patients with Covid-19 by analyzing lung X-rays. If classifying such data through deep learning proves effective in detecting positive cases, then this method can be used in hospitals and clinics when a classic test cannot be performed.

The dataset contains lung X-ray images for positive cases of Covid-19, as well as X-ray images of normal and viral pneumonias.
Link to the dataset: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

This project was developed by the following team :

- Patrizia Castiglione
- Isabelle Guillemin
- Qibo Sun

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Installation

To install and use this project, follow these steps:
1. Clone this repository to your local machine using the command:
```
git clone https://github.com/yourusername/yourproject.git
```
2. Navigate to the code directory:
```
cd code
```  
3. Create a virtual environment and activate it:
```
python -m venv env
source env/bin/activate (for Linux/Mac) or env\Scripts\activate.bat (for Windows)
```
4. Install the required dependencies using pip:
```
pip install -r requirements.txt
``` 
5. Download the dataset to your local machine from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
modify the paths in utils.py

## Streamlit App

**Add explanations on how to use the app.**

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

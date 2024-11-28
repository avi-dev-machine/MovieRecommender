# Movies Recommender System

This repository contains a machine learning algorithm implemented in **`app.py`**, designed to create a recommendation model and host a local web application.

## Features
- **Model Creation**: Running `app.py` will create a folder named `recommender_cache` containing the AI model.
- **Local Hosting**: Automatically hosts a web page on `localhost` to interact with the model.

## Requirements
- **Data**: To train the model, download the required data from [this link](#) and place the downloaded `data` folder in the same directory as `app.py`.
- **Python Packages**: Install the necessary Python packages by running:
  ```bash
  pip install pandas numpy scikit-learn faiss-cpu tqdm joblib flask
  ```

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/avi-dev-machine/MovieRecommender.git
cd MovieRecommender
```

### 2. Download and Place the Data
- Download the data from [this link](#).
- Place the `data` folder in the same directory as `app.py`.

### 3. Install the necessary Python packages by running:
```bash
  pip install pandas numpy scikit-learn faiss-cpu tqdm joblib flask
  ```

### 4. Run the Application
```bash
python app.py
```

This will:
1. Train the model and save it in the `recommender_cache` folder.
2. Host the web application on `localhost:5000`.

## Notes
- Ensure all dependencies are installed before running the application.
- If you wish to retrain the model with new data, replace the `data` folder and rerun `app.py`.

## Contributing
Feel free to open issues or submit pull requests to improve the project.



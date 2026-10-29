# Smart Agriculture: Crop Disease Detection

A student mini project that trains a Convolutional Neural Network (CNN) to classify crop-leaf images and serves predictions through Streamlit.

## 1. Dataset folder format

Download a labeled crop-leaf image dataset that you are permitted to use. Arrange it like this:

```text
dataset/
  Tomato_Healthy/
    image1.jpg
    image2.jpg
  Tomato_Early_Blight/
    image1.jpg
    image2.jpg
```

Each class folder name becomes a prediction label. Use several representative images per class. Do not mix training and test images manually; the script reserves 20% for validation.

## 2. Train the model (Google Colab or Python)

Install packages:

```bash
pip install -r requirements.txt
```

Put the `dataset/` folder beside `train.py`, then run:

```bash
python train.py
```

Training creates `crop_disease_model.keras`, `class_names.txt`, `accuracy_plot.png`, and `confusion_matrix.png`. For Google Colab, upload this project and dataset, then run the same script in a notebook cell using `!python train.py`.

## 3. Run the web app

Keep `app.py`, `crop_disease_model.keras`, and `class_names.txt` in the same folder:

```bash
streamlit run app.py
```

Upload a JPG or PNG leaf photo to see the predicted class and model confidence.

## 4. Publish on GitHub

1. Create a repository named `Smart-Agriculture-Crop-Detection`.
2. Upload `app.py`, `train.py`, `requirements.txt`, `.gitignore`, and this README.
3. Train the model and generate the two model files.
4. For public repositories, avoid committing large model files if they exceed GitHub limits. Use Git LFS or a model hosting service if needed.
5. For Streamlit Community Cloud, connect the repository and select `app.py` as the main file. Ensure model files are accessible to the deployed app.

## Important notes

- This ZIP contains source code, not a trained model or dataset. You must train it with a labeled dataset before making predictions.
- Validation accuracy depends on dataset size, quality, and class balance; no accuracy is guaranteed.
- This is an educational prototype, not a definitive agricultural diagnosis tool.

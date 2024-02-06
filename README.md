# Emotion Classification

- Used [transfer learning](https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4) to build a convolutional neural network with [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) that can classify emotion from different photos (Phoebe from F.R.I.E.N.D.S).

## Dataset  

### For Training (FER 2013)

Angry | Disgust | Fear
-- | -- | --
![angry](A3-Data/Fer2013/images/angry/Training_1021836.jpg) | ![disgust](A3-Data/Fer2013/images/disgust/Training_10371709.jpg) | ![fear](A3-Data/Fer2013/images/fear/Training_1044537.jpg)

Happy | Neutral | Sad | Surprise
-- | -- | -- | --
![happy](A3-Data/Fer2013/images/happy/Training_1018317.jpg) | ![neutral](A3-Data/Fer2013/images/neutral/Training_1017299.jpg) | ![sad](A3-Data/Fer2013/images/sad/Training_1015213.jpg) | ![surprise](A3-Data/Fer2013/images/surprise/Training_1002457.jpg)

### For Testing (Phoebe)

![test1](A3-Data/Phoebe/images/unknown/10_51.jpg) | ![test2](A3-Data/Phoebe/images/unknown/1_01.jpg) | ![test3](A3-Data/Phoebe/images/unknown/11_01.jpg)
-- | -- | --

## Predictions

Angry | Surprise
---- | ----
![angry](imgSrc/angryDetection.png) | ![surprise](imgSrc/surpriseDetection.png)

## Get Started

- (Optional) Create a virtual environment and activate it.

```bash
virtualenv venv
source venv/bin/activate
```

### Installation

- Make sure you have all the packages installed from requirements.txt.

```bash
pip install -m requirements-dev.txt

pip install -m requirements.txt
```

- Use the Jupyter Notebook (`test.ipynb`) to build the model and test the model's accuracy.

### Train the Model

```bash
python3 model/train.py
```

### Fine-tune the Model

```bash
python3 model/fine_tune.py
```

### Test the Model

```bash
pytest -v --cov --cov-report term-missing
```
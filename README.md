
# MNIST Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The model is trained using TensorFlow/Keras and achieves over 99% accuracy on the test set. It also includes real-time digit prediction from custom images and webcam input.

## 📌 Features

- CNN model trained on the MNIST dataset
- High accuracy (~99%) on test data
- Image preprocessing pipeline for custom digit images
- Webcam integration for live digit recognition
- Classification report and confusion matrix visualization

## 🛠️ Technologies Used

- Python
- TensorFlow & Keras
- NumPy, Matplotlib
- OpenCV (for webcam integration)
- PIL (for image preprocessing)
- TensorFlow Datasets (for loading MNIST)

## 📁 Project Structure

```
mnist_cnn/
│
├── mnist_cnn.py           # Main script with training, testing, and webcam prediction
├── digit.jpg              # Example custom digit image
├── README.md              # Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/amulyasingh02/digits.git
cd digits
```

### 2. Install Requirements
```bash
pip install tensorflow numpy matplotlib opencv-python pillow tensorflow-datasets
```

### 3. Run the Script
```bash
python mnist_cnn.py
```

## 🎯 Results

- Achieved 99%+ accuracy on the MNIST test set
- Successfully predicts digits from custom images and live webcam input

## 🧠 Learnings

- Practical implementation of CNNs using TensorFlow/Keras
- Real-time image preprocessing and digit prediction
- Hands-on experience with OpenCV, PIL, and ML model evaluation

## 📷 Example Output

- Predicted digit: 7
- Real-time image window displays captured digit for prediction

---

## 📎 License

This project is licensed under the MIT License.


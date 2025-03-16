# AI vs Human Text Classification using Reinforcement Learning

## Overview
This project uses Reinforcement Learning (RL) with Stable-Baselines3's PPO algorithm to classify text as either AI-generated or Human-written. The dataset consists of ~500K essays, labeled accordingly, and processed using TF-IDF vectorization. A custom Gymnasium environment is created to train the model to classify text based on rewards.

## Features
- Uses **TF-IDF Vectorization** to convert text into numerical form.
- Implements a **custom Gym environment** for text classification.
- Trains an **RL model (PPO)** from Stable-Baselines3.
- Achieves an **accuracy of ~97.7%** on the test dataset.
- Provides a **Streamlit web app** for user interaction.

## Dataset
- `AI_Human.csv`: Contains ~500K essays labeled as:
  - `generated = 1`: AI-generated text
  - `generated = 0`: Human-written text
- A balanced subset of 50,000 examples from each class is used for training.

## Installation
Ensure you have Python 3.8+ installed, then install dependencies:
```bash
pip install gymnasium numpy pandas scikit-learn stable-baselines3 streamlit
```

## Usage
### Training the Model
Run the training script:
```bash
python train_text_classifier.py
```
This will train the PPO model and save it to `model/text_classifier_rl.zip`.

### Running the Web App
Use Streamlit to launch the web application:
```bash
streamlit run app.py
```
This will open a browser UI where users can enter text and classify it.

## Model Training Details
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MLP (Multi-Layer Perceptron)
- **Training Steps**: 1,000,000
- **Reward System**:
  - +1 for correct classification
  - -1 for incorrect classification

## Performance
The model achieves an **average reward of 0.977** on the test dataset, indicating high accuracy in distinguishing AI-generated from human-written text.

## Example Prediction
```python
obs = vectorizer.transform(["This is a sample AI-generated text."]).toarray()
predicted_action, _ = model.predict(obs)
print(f"Prediction: {'AI-generated' if predicted_action[0] == 1 else 'Human-written'}")
```

## Future Improvements
- Implement **deep learning** methods like transformers for improved accuracy.
- Use **more diverse datasets** to improve generalization.
- Fine-tune **reward functions** for better learning dynamics.



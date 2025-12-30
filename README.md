# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune a BERT transformer model to classify news headlines into topic categories using transfer learning and deploy as an interactive web application.

## Dataset
**Source**: AG News Dataset (Hugging Face)  
**URL**: https://huggingface.co/datasets/ag_news

**Full Dataset**:
- Training: 120,000 samples
- Testing: 7,600 samples

**Subset Used** (for disk space optimization):
- Training: 10,000 samples
- Testing: 2,000 samples

**Categories (4 classes)**:
0. **World** - International news, politics, conflicts
1. **Sports** - Games, athletes, tournaments
2. **Business** - Markets, companies, economy
3. **Sci/Tech** - Technology, science, innovations

**Sample Headlines**:
```
Class 0 (World): "Iran Urges UN to Reject US Demand on Nuclear Program"
Class 1 (Sports): "Yankees Beat Red Sox 3-2 in Extra Innings"
Class 2 (Business): "Google Stock Surges After Earnings Beat Expectations"
Class 3 (Sci/Tech): "NASA Rover Discovers Evidence of Ancient Water on Mars"
```

---

## Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming | 3.8+ |
| **PyTorch** | Deep learning framework | 2.0+ |
| **Transformers** | BERT model & tokenizer | 4.30+ |
| **Datasets** | Dataset loading | 2.12+ |
| **scikit-learn** | Evaluation metrics | 1.0+ |
| **matplotlib/seaborn** | Visualization | Latest |
| **Streamlit** | Web deployment | 1.25+ |


## How to Run

### Prerequisites

**Install Dependencies**:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install matplotlib seaborn scikit-learn

```


```python
# Load saved model
from deploy_classifier import classify_news

# Test
headline = "Tesla announces new electric vehicle"
result = classify_news(headline)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Running Web App

```bash
# Install Streamlit
pip install streamlit

# Run app
streamlit run app_streamlit.py
```

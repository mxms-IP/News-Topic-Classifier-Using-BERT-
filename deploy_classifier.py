

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
model_path = './bert_news_classifier'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Class names
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

def classify_news(text):
    """Classify news headline"""
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=128, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return {
        'category': class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': {
            name: prob.item() 
            for name, prob in zip(class_names, predictions[0])
        }
    }

# Example usage
if __name__ == '__main__':
    example = "NASA launches new Mars rover mission"
    result = classify_news(example)
    print(f"Text: {example}")
    print(f"Category: {result['category']} ({result['confidence']:.2%})")

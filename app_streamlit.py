
@st.cache_resource
def load_model():
    """Load model (cached)"""
    model_path = './bert_news_classifier'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Load model
tokenizer, model = load_model()
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

# Page config
st.set_page_config(page_title="News Classifier", page_icon="")

# Title
st.title(" BERT News Topic Classifier")
st.markdown("Classify news headlines into: **World, Sports, Business, or Sci/Tech**")

# Input
text_input = st.text_area(
    "Enter a news headline:",
    placeholder="e.g., Tesla announces new electric vehicle with 500-mile range",
    height=100
)

if st.button("Classify", type="primary"):
    if text_input.strip():
        # Predict
        inputs = tokenizer(text_input, padding='max_length', truncation=True,
                         max_length=128, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        # Display results
        st.success(f"**Predicted Category:** {class_names[predicted_class]}")
        st.metric("Confidence", f"{confidence:.2%}")

        # Show all probabilities
        st.subheader("All Predictions:")
        for name, prob in zip(class_names, predictions[0]):
            st.progress(prob.item(), text=f"{name}: {prob.item():.2%}")
    else:
        st.warning("Please enter some text!")

# Example buttons
st.markdown("---")
st.markdown("**Try these examples:**")
col1, col2 = st.columns(2)

examples = [
    "Stock market hits record high",
    "Olympic champion breaks world record",
    "New COVID vaccine shows promising results",
    "Tech giant announces quarterly earnings"
]

for i, example in enumerate(examples):
    col = col1 if i % 2 == 0 else col2
    if col.button(example, key=f"ex_{i}"):
        st.rerun()

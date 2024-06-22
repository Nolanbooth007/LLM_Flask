

from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load pre-trained T5 model and tokenizer for summarization
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data['text']
    max_words = data['max_words']

    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Generate summary using the T5 model
    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_words,
        early_stopping=True
    )

    # Decode the summary tokens to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Return the summarized text as a JSON response
    return jsonify({'summary': summary})

if __name__ == '_main_':
    app.run(debug=True)


















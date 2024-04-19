import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import json
from langchain.cache import InMemoryCache
from langchain import PromptTemplate, LLMChain, OpenAI
from transformers import AutoTokenizer
from googletrans import Translator
warnings.filterwarnings("ignore")

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the intents
Path = r'c:\\Users\\Rohan\\Pictures\\rohan\\profile projects\\chatbot\\intents.json'
intents = json.loads(open(Path).read())

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize, remove stopwords, and lemmatize the pattern
        tokens = word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        patterns.append(' '.join(words))
        tags.append(intent['tag'])

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text.lower()])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define a function to tokenize and truncate the input
def tokenize_and_truncate(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokens

# Set up the OpenAI LLM with a custom tokenizer
model_name = "text-curie-001"
max_tokens = 256
temperature = 0.2 
token_limit = 100
callback = get_openai_callback(token_limit=token_limit)
cache = InMemoryCache()

openai_llm = OpenAI(
    model_name=model_name,
    max_tokens=max_tokens,
    temperature=temperature,
    tokenizer=tokenize_and_truncate,
    callback = [callback]
    cache= cache
)

# Set up the Google Translate API
translator = Translator()

template = """
You are a helpful and knowledgeable chatbot assistant for a popular clothing store called "Threads & Trends". Your goal is to provide excellent customer service by assisting customers with their inquiries related to our clothing products, sizes, styles, trends, and any other relevant information.

[...]

With this background information in mind, please provide a helpful and informative response to the following customer inquiry:

{query}

If the customer's inquiry is not directly related to our products or services, you can politely acknowledge their question and suggest that they rephrase their inquiry or provide more context related to our clothing store.
"""

prompt_template = PromptTemplate(input_variables=["query"], template=template)

# Create the LLMChain
llm_chain = LLMChain(prompt=prompt_template, llm=openai_llm)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text.lower()])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
# Define the chatbot function
def chatbot1():
    print("Welcome to the clothes sales chatbot! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break

        # Detect the language of the user's input
        detected_lang = translator.detect(prompt).lang

        # Translate the user's input to English
        prompt_en = translator.translate(prompt, dest='en').text

        # Get the response from the LLM
        response_en = llm_chain.run(prompt_en)

        # Translate the response back to the detected language
        response_translated = translator.translate(response_en, dest=detected_lang).text

        print(f"Chatbot ({detected_lang}): {response_translated}")

def answer():
    if input :
        chatbot()
    else:
        chatbot1()
    
# Streamlit app
def main():
    st.title("Clothing Store Chatbot")
    st.write("Welcome to our clothing store! Type your inquiry, and our chatbot will assist you.")

    user_input = st.text_input("You: ", key="input")
    if user_input:
        response = answer(user_input)
        st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
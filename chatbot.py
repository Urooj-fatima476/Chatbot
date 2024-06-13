import tkinter as tk
from tkinter import scrolledtext, END
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load necessary files and model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\\python\\intents.json').read())
model = load_model('chatbot_simplilearnmodel.h5')

# Load words and classes from pickle files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to get a response from the model
def get_response(user_input):
    ints = predict_class(user_input)
    res = get_response_from_intent(ints, intents)
    return res

def predict_class(sentence):
    bow = bow_sentence(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response_from_intent(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def bow_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return np.array([1 if word in sentence_words else 0 for word in words])

# Set up tkinter window
window = tk.Tk()
window.title("Urooj Chatbot")
window.geometry("500x550")
window.config(bg='black')  # Set window background to black

# Chat window
chat_window = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg='black', fg='white', font=("Arial", 12))
chat_window.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
chat_window.tag_config('user', foreground='yellow', background='black')  # Set user message style
chat_window.tag_config('bot', foreground='white', background='black')    # Set bot message style
chat_window.config(state=tk.DISABLED)

# User input field
input_frame = tk.Frame(window, bg='black')
input_frame.pack(pady=5, padx=10, fill=tk.X)

user_input = tk.StringVar()
input_entry = tk.Entry(input_frame, textvariable=user_input, font=("Arial", 14), bg='black', fg='white', insertbackground='white')
input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

def on_enter(event=None):
    user_text = user_input.get()
    if user_text.strip():
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "\nYou: " + user_text + "\n", "user")  # Added space before user message
        chat_window.yview(tk.END)
        response = get_response(user_text)
        chat_window.insert(tk.END, "\nBot: " + response + "\n", "bot")  # Added space before bot message
        chat_window.yview(tk.END)
        chat_window.config(state=tk.DISABLED)
        user_input.set("")

input_entry.bind("<Return>", on_enter)

# Send button
send_button = tk.Button(input_frame, text="Send", command=on_enter, bg='black', fg='white')
send_button.pack(side=tk.RIGHT)

# Start the GUI loop
window.mainloop()




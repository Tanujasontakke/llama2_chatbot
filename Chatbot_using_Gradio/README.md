# 🤖 LLaMA 2 Chatbot

## Introduction
This is a LLaMA 2 Chatbot project which is conversational AI powered by the LLaMA 2 model, designed to create an interactive chatbot experience. The chatbot uses a pre-trained model to generate human-like responses in a chat format, and I’ve built a simple yet effective interface using Gradio to make chatting with the AI a breeze.

## Why This Project?
The idea behind this project was the use of LLaMA 2, a state-of-the-art language model, to create a chatbot that can carry on meaningful conversations. The chatbot is designed to remember your chat history across sessions, which adds a personal touch to each interaction. You can think of it as a little helper that learns and improves over time, while also being a lot of fun to interact with!

## What You Need to Get Started
Before jumping into the code, let’s make sure you have everything set up:

### 1. Python 3.7 or Higher
Make sure you have Python 3.7 or a higher version installed on your machine.

### 2. Install Dependencies
The chatbot relies on a few libraries to work, so for installing them open your terminal or command prompt and run the following:

```bash
pip install torch 
pip install transformers
pip install gradio
```
These will get PyTorch, Transformers (from Hugging Face), and Gradio installed – the core components of the project.

### 3. GPU Setup (Optional Step)
If you have a GPU available, the code will automatically use it for faster performance. To check if your GPU is accessible, you can run this:

```bash
import torch
print("GPU Available:", torch.cuda.is_available())
```
If you don’t have a GPU, no worries! The chatbot will run on your CPU, but it may be slower.

## ⚡ Running the Chatbot
Once the dependencies are installed and you’re all set up, here’s how you can run the chatbot:

### Step 1: Clone the Repository
Start by cloning the repo to your local machine:

```bash
git clone https://github.com/Tanujasontakke/llama2_chatbot.git
cd llama2_chatbot
```
### Step 2: Start the Chatbot
Run the following command to start the chatbot. It will launch a local Gradio interface, which you can interact with through your browser.
```bash
python chatbot_autoSave.py
```
### Step 3: Start Chat!💬
Once the app is running, you’ll see a web interface where you can chat with the bot. You can:
	•	Start a new chat session.
	•	Select an existing chat session.
	•	Type your messages and receive responses from the chatbot.
The bot uses the LLaMA 2 model to generate replies, and it remembers the conversation context, so it’s like chatting with a real person!

## 💻 How It Works
The detailed working of the project is explained in the [How It Works](How_It_Works.md) file.


<sub>Emojis sourced from: [GitHub Emoji List](https://gist.github.com/rxaviers/7360908)</sub>

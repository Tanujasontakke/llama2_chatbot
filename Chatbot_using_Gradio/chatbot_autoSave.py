import torch
from transformers import AutoTokenizer, pipeline
import json
import os
import gradio as gr

# Check GPU availability
print("GPU Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
print("Device Name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")

# File to store chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Dictionary to store chat sessions
chat_sessions = {}

# Load Llama 2 model and tokenizer
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True, force_download=True)

# Create a text-generation pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful bot. Your answers are clear and concise.
<</SYS>>"""

# Load existing chat history
def load_chat_history():
    global chat_sessions
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_sessions = json.load(f)
    
    # Ensure there's at least one default chat session
    if not chat_sessions:
        chat_sessions["Chat 1"] = []

load_chat_history()

def save_chat_history():
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_sessions, f, indent=4)

def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if not history:  # Check if history is empty
        return SYSTEM_PROMPT + f"{message} [/INST]"

    formatted_message = SYSTEM_PROMPT

    for user_msg, model_answer in history:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    formatted_message += f"<s>[INST] {message} [/INST]"
    return formatted_message

def get_llama_response(message: str, history: list) -> str:
    query = format_message(message, history)
    sequences = llama_pipeline(
        query,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
    )
    response = sequences[0]['generated_text'][len(query):]

    if not response.endswith((".", "!", "?", "\"", "'")):
        response += " Please let me know if you need more details."

    return response

def process_message(message, chat_id):
    if chat_id not in chat_sessions:
        chat_sessions[chat_id] = []  # Initialize empty history if chat does not exist

    history = chat_sessions[chat_id]  # Get existing history
    response = get_llama_response(message, history)

    chat_sessions[chat_id].append({"role": "user", "content": message})
    chat_sessions[chat_id].append({"role": "assistant", "content": response})

    # Save chat history after each message
    save_chat_history()

    return chat_sessions[chat_id], ""

def update_chat(chat_id):
    return chat_sessions.get(chat_id, [])

# Ensure there is at least one default session
if not chat_sessions:
    chat_sessions["Chat 1"] = []

# Gradio UI with Save Button
with gr.Blocks() as demo:
    gr.Markdown("## LLaMA 2 Chatbot")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Chat Sessions")
            
            # Ensure chat_tabs has at least "Chat 1" as a valid choice
            #chat_tabs = gr.Radio(choices=list(chat_sessions.keys()), label="Chat History", value="Chat 1" if chat_sessions else None)
            chat_tabs = gr.Radio(
                choices=sorted(chat_sessions.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)))),
                label="Chat History",
                value="Chat 1" if chat_sessions else None
            )
            new_chat_btn = gr.Button("New Chat")

        with gr.Column(scale=5):
            chatbot = gr.Chatbot([], label="Chatbot", type="messages")
            user_input = gr.Textbox(label="Type your message here...")
            submit_button = gr.Button("Send")

    # Button Click Actions
    def create_new_chat():
        existing_numbers = [int(''.join(filter(str.isdigit, chat))) for chat in chat_sessions.keys()]
        new_chat_id = f"Chat {max(existing_numbers) + 1}" if existing_numbers else "Chat 1"
    
        chat_sessions[new_chat_id] = []
        return gr.update(choices=sorted(chat_sessions.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)))), value=new_chat_id), []
    
    new_chat_btn.click(create_new_chat, outputs=[chat_tabs, chatbot])
    chat_tabs.change(update_chat, inputs=[chat_tabs], outputs=[chatbot])
    submit_button.click(process_message, inputs=[user_input, chat_tabs], outputs=[chatbot, user_input])

# Start the chatbot
demo.launch(debug=True, share=True)
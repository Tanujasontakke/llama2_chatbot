# How It Works
A brief overview of how things work behind:

## LLaMA 2 Model
The chatbot is powered by the LLaMA-2-7B-chat-hf model, available from Hugging Face. It’s pre-trained on a large dataset and fine-tuned to handle conversational tasks.
```bash
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",  # Uses GPU if available
)
```
## Chat History
The bot stores your chat history in a JSON file (chat_history.json). This means it remembers your previous messages, so you can continue your conversations across sessions.
```bash
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_sessions = json.load(f)
```
Each session is stored in a dictionary, where the key is the session ID, and the value is a list of user and assistant messages. The history is saved and loaded every time the chatbot is run.

## Gradio Interface
The user interface is built with Gradio, making it super easy to interact with the chatbot. You can select your chat sessions, start a new one, and send messages through the interface.
```bash
with gr.Blocks() as demo:
    gr.Markdown("## LLaMA 2 Chatbot")
    chatbot = gr.Chatbot([], label="Chatbot", type="messages")
    user_input = gr.Textbox(label="Type your message here...")
    submit_button = gr.Button("Send")
```
## Chatbot Memory
The chatbot remembers the conversation within a session. It even formats each new message properly, making sure it doesn’t forget previous exchanges. Each message is processed and formatted in a way that the LLaMA model can understand the context.
```bash
def format_message(message: str, history: list) -> str:
    formatted_message = SYSTEM_PROMPT
    for user_msg, model_answer in history:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"
    formatted_message += f"<s>[INST] {message} [/INST]"
    return formatted_message
```
# Things to Remember - 
	•	Chat History: The chatbot keeps track of the conversation history. If the file becomes too large, you might want to limit how many past exchanges it remembers.
	•	GPU: If you have a GPU, the chatbot will use it for faster performance, but it will work fine on a CPU as well. It might be a bit slower, though.
	•	Model Size: The LLaMA-2-7B model is large, so make sure you have enough disk space and memory.

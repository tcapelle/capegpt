from anthropic import Anthropic
import base64
from dataclasses import dataclass
import json
import openai
import streamlit as st
from typing import Dict, List, Union
import weave
from pydantic import Field

weave.init("capeGPT")

# Initialize clients
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
llama_client = openai.OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1")
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])


class Model(weave.Model):
    name: str
    client: Union[openai.OpenAI, Anthropic]

    @weave.op
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        raise NotImplementedError

class OpenAIModel(Model):
    @weave.op
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        return self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            stream=True,
            temperature=temperature,
        )

class AnthropicModel(Model):
    @weave.op
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        # Extract system message and convert other messages to Anthropic format
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        anthropic_messages = [
            {
                "role": "user" if msg["role"] == "user" else "assistant",
                "content": [{"type": "text", "text": msg["content"]}]
            }
            for msg in messages if msg["role"] != "system"
        ]

        stream = self.client.messages.create(
            max_tokens=4096,
            model=self.name,
            temperature=temperature,
            system=system_message,
            messages=anthropic_messages,
            stream=True,
        )
        for event in stream:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                yield event.delta.text

# Define models
models = {
    "gpt-4o-mini": OpenAIModel(name="gpt-4o-mini", client=client),
    "gpt-4o": OpenAIModel(name="gpt-4o", client=client),
    "llama405": OpenAIModel(name="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8", client=llama_client),
    "claude-3.5-sonnet": AnthropicModel(name="claude-3-5-sonnet-20240620", client=anthropic_client)
}

@dataclass
class Chat:
    name: str
    system_message: str = "You are a helpful assistant, be brief."
    messages: list[dict[str, str]] = Field(default_factory=list)

    def __post_init__(self):
        self.clear_messages()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def clear_messages(self):
        self.messages = []

    def rename(self, new_name: str):
        self.name = new_name

    def generate_summary(self) -> str:
        if not self.messages:
            return self.name
        
        summary_prompt = "Summarize the following conversation in 5 words or less:\n\n"
        for message in self.messages:
            summary_prompt += f"{message['role']}: {message['content']}\n"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=10,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    def update_name_with_summary(self) -> bool:
        if self.name == "New Chat" and self.messages:
            summary = self.generate_summary()
            self.rename(summary)

    def set_system_message(self, message: str):
        self.system_message = message

    def to_json(self) -> str:
        chat_data = {
            "name": self.name,
            "system_message": self.system_message,
            "messages": self.messages
        }
        return json.dumps(chat_data, indent=2)



class ChatHistory:
    def __init__(self):
        if "chats" not in st.session_state:
            st.session_state.chats = [Chat("New Chat")]
        if "current_chat_index" not in st.session_state:
            st.session_state.current_chat_index = 0

    def get_current_chat(self) -> Chat:
        return st.session_state.chats[st.session_state.current_chat_index]

    def set_current_chat(self, index: int):
        st.session_state.current_chat_index = index

    def get_all_chats(self) -> List[Chat]:
        return st.session_state.chats

    def add_chat(self):
        current_chat = self.get_current_chat()
        
        # Check if the current chat is empty
        if not current_chat.messages:
            return  # Don't create a new chat if the current one is empty
        
        # Rename the current chat if it's still named "New Chat"
        current_chat.update_name_with_summary()
        
        # Add a new chat and set it as current
        st.session_state.chats.append(Chat("New Chat"))
        st.session_state.current_chat_index = len(st.session_state.chats) - 1

    def clear_current_chat(self):
        self.get_current_chat().clear_messages()

def main():    
    chat_history = ChatHistory()

    # Sidebar
    with st.sidebar:
        st.title("Previous Chats")
        
        if st.button("New Chat"):
            chat_history.add_chat()
            st.rerun()
        
        chat_options = [chat.name for chat in chat_history.get_all_chats()]
        selected_chat_index = st.selectbox(
            "Select Chat",
            options=range(len(chat_options)),
            format_func=lambda x: chat_options[x],
            index=st.session_state.current_chat_index
        )
        if selected_chat_index != st.session_state.current_chat_index:
            chat_history.set_current_chat(selected_chat_index)
            st.rerun()
        
        st.markdown("---")  # Horizontal line
        
        st.subheader("Model Settings")
        current_chat = chat_history.get_current_chat()
        system_message = st.text_area("Set system message:", value=current_chat.system_message)
        if system_message != current_chat.system_message:
            current_chat.set_system_message(system_message)
        if "model" not in st.session_state:
            st.session_state["model"] = "gpt-4o"
        
        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Choose a model",
            options=model_names,
            index=model_names.index(st.session_state["model"]),
            key="model_selectbox"
        )
        st.session_state["model"] = selected_model
        
        temperature = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

        st.markdown("---")  # Horizontal line
        st.markdown("- **GPT4o** : Our high-intelligence flagship model for complex, multi-step tasks\n"
                    "- **GPT4o-mini** : Our affordable and intelligent small model for fast, lightweight tasks\n"
                    "- **Llama405** : The Latest and baddest model from MetaAI (may not be available)\n"
                    "- **Claude 3.5 Sonnet** : A powerful model from Anthropic")
    # Main content
    current_chat = chat_history.get_current_chat()

    # Add chat header with download button
    st.subheader(f"Current Chat: {current_chat.name}")

    # Display chat history
    for message in current_chat.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        current_chat.add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            current_model = models[st.session_state["model"]]
            messages = [
                {"role": "system", "content": current_chat.system_message},
                *[{"role": m["role"], "content": m["content"]} for m in current_chat.messages]
            ]
            stream = current_model.generate_stream(messages, temperature)

            response = ""
            if isinstance(current_model, AnthropicModel):
                response = st.write_stream(stream)
            else:
                response = st.write_stream(stream)

        current_chat.add_message("assistant", response)
        st.caption(f"Model: {st.session_state['model']}")
    
    # Function to handle download
    def download_chat():
        current_chat.update_name_with_summary()
        return current_chat.to_json()
    
    st.download_button(
        label="Download Current Chat",
        data=download_chat(),
        file_name=f"{current_chat.name.replace(' ', '_').lower()}_chat.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
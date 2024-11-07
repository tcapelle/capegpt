import base64
from dataclasses import dataclass, field
import json
import openai
import streamlit as st
from typing import Dict, List, Union
import weave
import anthropic
from extra_streamlit_components import CookieManager
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta

def manage_cookies():
    cookie_manager = CookieManager()
    
    # Load cookies
    stored_data = None
    # Request all cookies and wait for the browser to send them
    cookies = cookie_manager.get_all()
    if cookies:
        print(cookies)
        stored_data = cookies.get("chat_history")
        if stored_data:
            stored_data = json.loads(stored_data)
    
    # Save cookies if needed
    if 'save_chat_history' in st.session_state and st.session_state.save_chat_history:
        chat_data = {
            "chats": [chat.model_dump() for chat in st.session_state.chats],
            "current_chat_index": st.session_state.current_chat_index
        }
        # Set an expiration date for the cookie
        expiration_date = (datetime.now() + timedelta(days=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        cookie_manager.set(
            "chat_history",
            json.dumps(chat_data),
            expires=expiration_date
        )
        st.session_state.save_chat_history = False
        st.sidebar.success("Chat history saved to cookie!")

    # Clear cookies if needed
    if 'clear_cookies' in st.session_state and st.session_state.clear_cookies:
        cookie_manager.delete("chat_history")
        st.session_state.clear_cookies = False
        stored_data = None
    
    return stored_data

# Model classes
class Model(weave.Model):
    name: str
    client: Union[openai.OpenAI, anthropic.Anthropic]

    @weave.op
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        raise NotImplementedError

class OpenAIModel(Model):
    @weave.op
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        if "o1" in self.name:
            # For o1 models, combine system message with first user message
            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            processed_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    continue
                if msg["role"] == "user" and not processed_messages:
                    # Prepend system message to first user message
                    msg = {"role": "user", "content": f"{system_message}\n\n{msg['content']}"}
                processed_messages.append(msg)
            
            # Non-streaming response for o1 models
            response = self.client.chat.completions.create(
                model=self.name,
                messages=processed_messages,
                stream=False,
                temperature=temperature,
            )
            yield response.choices[0].message.content
        else:
            # Streaming response for other models
            yield from self.client.chat.completions.create(
                model=self.name,
                messages=messages,
                stream=True,
                temperature=temperature,
            )

class AnthropicModel(Model):
    @weave.op
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        anthropic_messages = [
            {"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]}
            for msg in messages if msg["role"] != "system"
        ]
        with self.client.messages.stream(
            max_tokens=4096,
            model=self.name,
            temperature=temperature,
            system=system_message,
            messages=anthropic_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

# Define models
models = {
    "claude-3.5-sonnet": AnthropicModel(name="claude-3-5-sonnet-latest", client=anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])),
    "claude-3.5-haiku": AnthropicModel(name="claude-3-5-haiku-latest", client=anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])),
    "gpt-4o-mini": OpenAIModel(name="gpt-4o-mini", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    "gpt-4o": OpenAIModel(name="gpt-4o", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    # "gpt-4": OpenAIModel(name="gpt-4", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    # "gpt-4-turbo": OpenAIModel(name="gpt-4-turbo", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    "o1-preview": OpenAIModel(name="o1-preview", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    "o1-mini": OpenAIModel(name="o1-mini", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
}

class Chat(BaseModel):
    name: str
    system_message: str = Field(default="You are a helpful assistant, be brief.")
    messages: List[Dict[str, str]] = Field(default_factory=list)
    model_name: str = Field(default="claude-3.5-sonnet")

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
        response = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"]).chat.completions.create(
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

class ChatHistory:
    def __init__(self):
        if 'chats' not in st.session_state:
            print("Loading chats from cookies")
            stored_data = manage_cookies()
            if stored_data:
                st.session_state.chats = [Chat.model_validate(chat_data) for chat_data in stored_data.get("chats", [])]
                st.session_state.current_chat_index = stored_data.get("current_chat_index", 0)
            else:
                st.session_state.chats = [Chat(name="New Chat")]
                st.session_state.current_chat_index = 0

    def save_chats(self):
        st.session_state.save_chat_history = True

    def get_current_chat(self) -> Chat:
        return st.session_state.chats[st.session_state.current_chat_index]

    def set_current_chat(self, index: int):
        st.session_state.current_chat_index = index

    def add_chat(self):
        current_chat = self.get_current_chat()
        if current_chat.messages:
            current_chat.update_name_with_summary()
            st.session_state.chats.append(Chat(name="New Chat"))
            st.session_state.current_chat_index = len(st.session_state.chats) - 1
            self.save_chats()

    def clear_current_chat(self):
        self.get_current_chat().clear_messages()
        self.save_chats()

def main():
    chat_history = ChatHistory()

    with st.sidebar:
        st.title("Previous Chats")
        if st.button("New Chat"):
            chat_history.add_chat()
            st.rerun()
        
        chat_options = [chat.name for chat in st.session_state.chats]
        selected_chat_index = st.selectbox(
            "Previous Chats",
            options=range(len(chat_options)),
            format_func=lambda x: chat_options[x],
            index=st.session_state.current_chat_index
        )
        if selected_chat_index != st.session_state.current_chat_index:
            chat_history.set_current_chat(selected_chat_index)
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("Model Settings")
        current_chat = chat_history.get_current_chat()
        system_message = st.text_area("Set system message:", value=current_chat.system_message)
        if system_message != current_chat.system_message:
            current_chat.set_system_message(system_message)

        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Choose a model",
            options=model_names,
            index=model_names.index(current_chat.model_name),
            key="model_selectbox"
        )
        if selected_model != current_chat.model_name:
            current_chat.model_name = selected_model

        temperature = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

        st.markdown("---")
        if st.button("Save Chat History"):
            st.session_state.save_chat_history = True
            st.rerun()

        if st.button("Clear All Cookies"):
            st.session_state.clear_cookies = True
            st.rerun()

        # Display current cookie content
        st.subheader("Current Cookie Content:")
        cookie_manager = CookieManager(key='cookie_manager_2')
        current_cookie = cookie_manager.get(cookie="chat_history")
        if current_cookie:
            st.json(json.loads(current_cookie))
        else:
            st.write("No chat history found.")

    current_chat = chat_history.get_current_chat()
    st.subheader(f"Current Chat: {current_chat.name}")

    for message in current_chat.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        current_chat.add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            current_model = models[current_chat.model_name]
            messages = [
                {"role": "system", "content": current_chat.system_message},
                *current_chat.messages
            ]
            stream = current_model.generate_stream(messages, temperature)
            response = st.write_stream(stream)

        current_chat.add_message("assistant", response)
        chat_history.save_chats()
        st.rerun()

    st.download_button(
        label="Download Current Chat",
        data=json.dumps(current_chat.model_dump(), indent=2),
        file_name=f"{current_chat.name.replace(' ', '_').lower()}_chat.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
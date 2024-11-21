import base64
from dataclasses import dataclass, field
import json
import openai
import streamlit as st
from typing import Dict, List, Union
import anthropic
from pydantic import BaseModel, Field

# Model classes
class Model(BaseModel):
    model_name: str
    client: Union[openai.OpenAI, anthropic.Anthropic]

    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        raise NotImplementedError

class OpenAIModel(Model):
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        if "o1" in self.model_name:
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
                model=self.model_name,
                messages=processed_messages,
                stream=False,
                temperature=temperature,
            )
            yield response.choices[0].message.content
        else:
            # Streaming response for other models
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=temperature,
            )
            for chunk in stream:
                if chunk.choices:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

class AnthropicModel(Model):
    def generate_stream(self, messages: List[Dict[str, str]], temperature: float):
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        anthropic_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages if msg["role"] != "system"
        ]
        with self.client.messages.stream(
            max_tokens=4096,
            model=self.model_name,
            temperature=temperature,
            system=system_message,
            messages=anthropic_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

# Define models
models = {
    "claude-3.5-sonnet": AnthropicModel(model_name="claude-3-5-sonnet-latest", client=anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])),
    "claude-3.5-haiku": AnthropicModel(model_name="claude-3-5-haiku-latest", client=anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])),
    "gpt-4o-mini": OpenAIModel(model_name="gpt-4o-mini", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    "gpt-4o": OpenAIModel(model_name="gpt-4o", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    # "gpt-4": OpenAIModel(name="gpt-4", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    # "gpt-4-turbo": OpenAIModel(name="gpt-4-turbo", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    "o1-preview": OpenAIModel(model_name="o1-preview", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
    "o1-mini": OpenAIModel(model_name="o1-mini", client=openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])),
}

class Chat(BaseModel):
    name: str
    system_message: str = Field(default="You are a helpful assistant, be brief.")
    messages: List[Dict[str, str]] = Field(default_factory=list)
    model_name: str = Field(default="claude-3.5-sonnet")

    def add_message(self, role: str, content: str):
        if content.strip():
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
            st.session_state.chats = [Chat(name="New Chat")]
            st.session_state.current_chat_index = 0

    def save_chats(self):
        pass  # No action needed since we're using st.session_state

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

    def clear_current_chat(self):
        self.get_current_chat().clear_messages()

def main():
    st.markdown("""
        <style>
        .stChatInput {
            height: 50px !important;
            width: 100% !important;
            margin: 0 auto !important;
        }
        textarea.st-cf {
            height: 80px !important;
            width: 80% !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'should_rerun' not in st.session_state:
        st.session_state.should_rerun = False

    chat_history = ChatHistory()
    current_chat = chat_history.get_current_chat()
    temperature = st.session_state.get('temperature', 1.0)

    with st.sidebar:
        st.title("Chat Management")

        if st.button("New Chat"):
            chat_history.add_chat()
            st.session_state.should_rerun = True

        chat_options = [chat.name for chat in st.session_state.chats]
        selected_chat_index = st.selectbox(
            "Previous Chats",
            options=range(len(chat_options)),
            format_func=lambda x: chat_options[x],
            index=st.session_state.current_chat_index
        )
        if selected_chat_index != st.session_state.current_chat_index:
            chat_history.set_current_chat(selected_chat_index)
            st.session_state.should_rerun = True

        st.markdown("---")

        st.subheader("Model Settings")
        system_message = st.text_area("Set system message:", value=current_chat.system_message)
        if system_message != current_chat.system_message:
            current_chat.set_system_message(system_message)
            st.session_state.should_rerun = True

        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Choose a model",
            options=model_names,
            index=model_names.index(current_chat.model_name),
            key="model_selectbox"
        )
        if selected_model != current_chat.model_name:
            current_chat.model_name = selected_model
            st.session_state.should_rerun = True

        temperature = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=temperature, step=0.1)
        st.session_state.temperature = temperature

        st.markdown("---")

        if st.button("Clear Current Chat"):
            chat_history.clear_current_chat()
            st.session_state.should_rerun = True

    st.subheader(f"Current Chat: {current_chat.name}")

    for message in current_chat.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        current_chat.add_message("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                current_model = models[current_chat.model_name]
                messages = [
                    {"role": "system", "content": current_chat.system_message},
                    *current_chat.messages
                ]

                # Filter out messages with empty content
                messages = [msg for msg in messages if msg['content'].strip()]
                stream = current_model.generate_stream(messages, temperature)
                response_placeholder = st.empty()
                assistant_response = ""

                for chunk in stream:
                    assistant_response += chunk
                    response_placeholder.markdown(assistant_response)

                # Ensure the assistant's response is not empty
                if assistant_response.strip():
                    current_chat.add_message("assistant", assistant_response)
                else:
                    error_message = "The assistant did not produce a response."
                    st.error(error_message)
                    current_chat.add_message("assistant", error_message)
            except Exception as e:
                error_message = "Sorry, there was an error communicating with the model. Please try again in a moment."
                st.error(error_message)
                raise e
                current_chat.add_message("assistant", error_message)

        # Set the flag to rerun after processing the message
        st.session_state.should_rerun = True

    st.download_button(
        label="Download Current Chat",
        data=json.dumps(current_chat.model_dump(), indent=2),
        file_name=f"{current_chat.name.replace(' ', '_').lower()}_chat.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
    if st.session_state.get('should_rerun', False):
        st.session_state.should_rerun = False
        # st.experimental_rerun()
from openai import OpenAI
import streamlit as st
from typing import List, Dict

class Chat:
    def __init__(self, name: str, client: OpenAI):
        self.name = name
        self.messages: List[Dict[str, str]] = []
        self.client = client

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
        
        response = self.client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=10,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    def update_name_with_summary(self) -> bool:
        if self.name == "New Chat" and self.messages:
            summary = self.generate_summary()
            self.rename(summary)
            return True
        return False

class ChatHistory:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        if "chats" not in st.session_state:
            st.session_state.chats = [Chat("New Chat", self.client)]
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
        st.session_state.chats.append(Chat("New Chat", self.client))
        st.session_state.current_chat_index = len(st.session_state.chats) - 1

    def clear_current_chat(self):
        self.get_current_chat().clear_messages()

def main():    
    chat_history = ChatHistory()
    client = chat_history.client

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
        
        st.title("Model Settings")
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4o"
        
        st.session_state["openai_model"] = st.selectbox(
            "Choose a model:",
            options=model_options,
            index=model_options.index(st.session_state["openai_model"]),
            key="model_selectbox"
        )
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    # Main content
    current_chat = chat_history.get_current_chat()
    for message in current_chat.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        current_chat.add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in current_chat.messages
                ],
                stream=True,
                temperature=temperature,
            )
            response = st.write_stream(stream)
        current_chat.add_message("assistant", response)
        st.caption(f"Model: {st.session_state['openai_model']}")

if __name__ == "__main__":
    main()
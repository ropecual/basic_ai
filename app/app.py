import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms import Ollama
from streamlit.web.server import Server

Server.enableCORS = True
# Configuração do modelo e memória
def setup_chain():
  llm = Ollama(base_url="http://ollama:11434", model="llama3")
  memory = ConversationBufferMemory()
  chain = ConversationChain(llm=llm, memory=memory)
  return chain


# Interface do usuário
def main():
  st.title("Chat com Llama3 🦙")

  if "chain" not in st.session_state:
    st.session_state.chain = setup_chain()

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  prompt = st.chat_input("Digite sua mensagem...")

  if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.spinner("Pensando..."):
      response = st.session_state.chain.run(prompt)

      st.session_state.messages.append({"role": "assistant", "content": response})
      with st.chat_message("assistant"):
        st.markdown(response)


if __name__ == "__main__":
  main()
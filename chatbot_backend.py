from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrockConverse

def demo_chatbot():
    demo_llm = ChatBedrockConverse(model="us.deepseek.r1-v1:0",
                                      temperature=0.7,
                                      max_tokens=500)

    return demo_llm

def demo_memory():
    llm_data= demo_chatbot()
    memory = ConversationBufferMemory(llm=llm_data, max_token=100)
    return memory

def demo_conversation(input_text,memory):
    llm_chain_data= demo_chatbot()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory,verbose=True)

    chat_reply = llm_conversation.invoke(input_text)
    return chat_reply['response']

# Importing Libraries

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit Framework

st.title("Celebrity Search App")
input_text = st.text_input("Search the topic you want")

first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# OpenAI LLMs

llm=OpenAI(temperature=0.8)
chain = LLMChain(
    llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory
    )

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)

chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory
    )

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 05 major events around {dob} in the world"
)

chain3 = LLMChain(
    llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory
    )

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], verbose=True, input_variables=['name'], output_variables=['person', 'dob', 'description']
    )

if input_text:
    st.write(parent_chain({'name': input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
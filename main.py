from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from langchain.llms import CTransformers

template = """Question: {question}"""


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        # model="../downloads/mistral-7b-instruct-v0.1.Q8_0.gguf",
        model="../downloads/mistral-7b-code-16k-qlora.Q8_0.gguf",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
    )
    return llm


@cl.on_chat_start
def main():
    llm = load_llm()

    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()

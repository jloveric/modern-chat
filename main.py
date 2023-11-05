from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.prompts.chat import ChatPromptTemplate
import chainlit as cl
from langchain.llms import CTransformers

template = """Question: {question}"""


models = {
    "mistral": {
        "path": "../downloads/mistral-7b-instruct-v0.1.Q8_0.gguf",
        "message": [
            ("system", "You are Mistral."),
            ("user", "{input}"),
        ],
    },
    "openhermes": {
        "path": "../downloads/openhermes-2.5-mistral-7b.Q5_K_M.gguf",
        "message": [
            ("system", "You are Hermes 2."),
            ("user", "{input}"),
        ],
    },
}


# Loading the model
def load_llm(model_name):
    # Load the locally downloaded model here
    llm = CTransformers(
        model=models.get(model_name)["path"],
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
        gpu_layers=100,
    )
    return llm


@cl.on_chat_start
def main():
    model_name = "mistral"
    llm = load_llm(model_name=model_name)

    # Instantiate the chain for that user session
    prompt = ChatPromptTemplate.from_messages(models.get(model_name)["message"])
    print("prompt", prompt)
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
    print("res", res)
    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()

import os
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



#pip install langsmith==0.1.105
#pip install openai
#pip install langchain==0.2.14
#pip install -qU langchain-openai

os.environ['OPENAI_API_KEY'][:3]
#print(os.environ)
modelAi = "gpt-4o-mini" #"gpt-3.5-turbo-0125"
client = OpenAI()
system_prompt = "You're a seasoned chef working as a helpful mentor to cooking enthusiasts."

def completions(message: str):
    result = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        model=modelAi,
    )
    result = result.choices[0].message.content
    print(result.choices[0].message.content)
    return result

#Langchain ofrece un servicio para observabilidad y evaluación de aplicaciones con LLMs (y otras),
# podemos probar la version mas simple de una traza en Langsmith
# Ahora podemos ir a Langsmith y revisar nuestras trazas

# Wrap OpenAI client to trace its inputs/outputs
ls_client = wrap_openai(client)

# We can decorate any function with @traceable too!
@traceable
def ask_chef(message: str) -> str:
    result = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        model=modelAi,
    )
    msg = result.choices[0].message.content
    return msg

#Langchain basico
#Vamos a construir lo mismo de antes pero con las primitivas de Langchain
def langchain_chat(message):
    model = ChatOpenAI(model=modelAi)
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(message),
    ]
    #result = model.invoke(messages)
    #return result.content
    # podemos empezar a combinar bloques, como por ejemplo para extraer el mensaje de la respuesta
    chain = modelAi | StrOutputParser()
    return chain.invoke(messages)

#pasar por Prompt Templates, que nos permiten armar prompts más dinámicos
def langchain_prompt_template():
    model = ChatOpenAI(model=modelAi)
    prompt_template = ChatPromptTemplate.from_messages([
        # Note that Langchain can almost always take (type, content) tuples in place of the specific message object types
        ("system", system_prompt + " You always answer in {language}."),
        ("human", "{input}"),
    ])
    chain = prompt_template | model | StrOutputParser()
    return chain.invoke({"language": "french", "input": "how do I make tomatoes soup?"})

if __name__ == '__main__':
    print('PyCharm')
    # print(completions("Tengo huevos, mantequilla, cilantro, y cebollín, qué comida rica puedo hacer?"))
    # print(ask_chef("Tengo un pollo entero y quiero cocinarlo lo más rápido posible. Alguna sugerencia?"))
    #print(langchain_chat("como preparo un yaguarlocro?"))
    print(langchain_prompt_template())
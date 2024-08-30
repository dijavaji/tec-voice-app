import os
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from collections import defaultdict

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


#Hagamos un Chatbot con Memoria
def chatBot_memory():
    model = ChatOpenAI(model=modelAi)
    messages = [
        SystemMessage("You're a simple companion. You go straight to the point and give very concise answers."),
        HumanMessage("Hola, me llamo Rodrigo"),
        AIMessage("Hola Rodrigo!"),
        HumanMessage("Cómo me llamo?"),
    ]
    chain = model | StrOutputParser()
    return chain.invoke(messages)

#hacemos que sea mas interactivo
history = [
    HumanMessage("Hola, me llamo dj"),
    AIMessage("Hola dj!"),
]

messages = [
    SystemMessage("You're a simple companion. You go straight to the point and give very concise answers."),
    # We need a place for the chat history until "now"
    MessagesPlaceholder(variable_name="history"),
    # And then the new user message
    HumanMessagePromptTemplate.from_template("{user_message}"),
]
model = ChatOpenAI(model=modelAi)
chain = ChatPromptTemplate.from_messages(messages) | model | StrOutputParser()

@traceable
def chat(message: str) -> str:
    return chain.invoke({"history": history, "user_message": message})


#Pero nosotros tenemos que manejar la memoria "a mano", agregar los mensajes en la historia, etc, etc.
#Langchain facilita todo eso, envolviendo un Runnable (e.g. una cadena) en algo que inyecta historia,
# y captura el output para agregarlo a la historia para después: RunnableWithMessageHistory.
chat_history = InMemoryChatMessageHistory()
def get_history() -> InMemoryChatMessageHistory:
    return chat_history
def get_chat_history(human_msg):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    base_chain = prompt | model
    chain = RunnableWithMessageHistory(base_chain, get_history, input_messages_key="messages") | StrOutputParser()
    return chain.invoke({"messages":human_msg })

#Langchain tiene varias implementaciones de ChatMessageHistory, para guardar historial en bases de datos, en archivos, etc.
# Explorar esas queda como ejercicio para el lector.
#Ejemplo de como mantener varias sesiones en paralelo. podemos pasar configuracion como input de la cadena,
# que podra ser usada por sus componentes. Ejemplo:

histories: dict[str, InMemoryChatMessageHistory] = defaultdict(InMemoryChatMessageHistory)
def get_history_by_session_id(session_id: str) -> InMemoryChatMessageHistory:
    return histories[session_id]

prompt = ChatPromptTemplate.from_messages([
        SystemMessage(system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

base_chain = prompt | model
chain = RunnableWithMessageHistory(base_chain, get_history_by_session_id, input_messages_key="messages") | StrOutputParser()
@traceable
def chat(session_id: str, message: str) -> str:
    config = {
        "configurable": {
            "session_id": session_id,
        }
    }
    return chain.invoke(
        {
            "messages": [HumanMessage(message)],
        },
        config=config,
    )

if __name__ == '__main__':
    print('PyCharm')
    # print(completions("Tengo huevos, mantequilla, cilantro, y cebollín, qué comida rica puedo hacer?"))
    # print(ask_chef("Tengo un pollo entero y quiero cocinarlo lo más rápido posible. Alguna sugerencia?"))
    #print(langchain_chat("como preparo un yaguarlocro?"))
    #print(langchain_prompt_template())
    #print(chat("cual es mi nombre?"))
    #tenemos memoria y funciona como un chat interactivo.
    #print(get_chat_history([HumanMessage(content="hola, quiero cocinar algo con huevos y papas y cebolla")]))
    #print(get_chat_history([HumanMessage("y si tengo chorizo?"), HumanMessage("Ah no, me equivoqué, no tengo chorizo, pero tengo jamón"), ]))
    #sesiones en paralelo
    print(chat("dj", "quiero cocinar pastel de choclo, qué ingredientes necesito comprar?"))
    print(chat("juan", "tengo platanos, naranjas, y kiwis, como hago tutti-fruti?"))
    print(chat("dj", "no quedaba carne, lo puedo hacer veggie?"))
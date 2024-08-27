import os
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


#pip install langsmith==0.1.105
#pip install openai
#pip install langchain==0.2.14
#pip install -qU langchain-openai

os.environ['OPENAI_API_KEY'][:3]
#print(os.environ)
model = "gpt-4o-mini" #"gpt-3.5-turbo-0125"
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
        model=model,
    )
    result = result.choices[0].message.content
    print(result.choices[0].message.content)
    return result


if __name__ == '__main__':
    print('PyCharm')
    # print(completions("Tengo huevos, mantequilla, cilantro, y cebollín, qué comida rica puedo hacer?"))
    # print(ask_chef("Tengo un pollo entero y quiero cocinarlo lo más rápido posible. Alguna sugerencia?"))
    #print(langchain_chat("como preparo un yaguarlocro?"))
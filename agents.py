from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

#pip install langchain-community==0.2.14
#pip install wikipedia==1.4.0

#Agentes
#Ya sabemos hacer chatbots Ahora vamos un paso mas alla: si queremos que el chatbot pueda hacer cosas,
# necesitamos reestructurar el flujo. Podemos usar function calling para que el LLM pueda tomar acciones en respuesta al usuario,
# y luego re-evaluar en base al resultado de esas acciones.


#Tools
#Empecemos por explorar las herramientas, y que mejor para partir que usar alguna herramienta predefinida
def wiki_tool(query):
    wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=5000)
    wikipedia_query = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)
    return wikipedia_query.invoke({"query":query})


if __name__ == '__main__':
    print('PyCharm')
    print(wiki_tool("hp lovecraft"))
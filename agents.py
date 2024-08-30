from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.tools import tool

from typing import Type
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

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
    #Esencialmente es una funcion con algunos metadatos para que el LLM la entienda
    #print(wikipedia_query.name, wikipedia_query.description, wikipedia_query.args_schema.schema())
    return wikipedia_query.invoke({"query":query})

#Podemos hacer nuestras propias herramientas con anotaciones
@tool
def weather(city_name: str) -> str:
    """Obtiene el pronostico del tiempo para una ciudad."""
    match city_name.lower():
        case 'santiago':
            return 'cold'
        case 'valparaiso':
            return 'windy'
        case 'concepcion':
            return 'rainy'
        case _:
            return 'fair'

#O podemos hacer tools con clases derivadas de BaseTool "a mano"
class CompoundInterestCalculatorArgs(BaseModel):
    n_periods: int = Field(description="Numero de periodos a calcular")
    rate: float = Field(description="Tasa de interes por periodo, en puntos porcentuales")
    initial_amount: float = Field(description="Importe al inicio del primer periodo")

class CompoundInterestCalculator(BaseTool):
    args_schema: Type[BaseModel] = CompoundInterestCalculatorArgs
    name = 'compound_interest_calculator'
    description = 'Calcula el interes compuesto de una cantidad inicial durante una serie de periodos a una tasa especifica.'

    def _run(self, n_periods: int, rate: float, initial_amount: float) -> str:
        return str(initial_amount*(1+rate/100)**n_periods)



if __name__ == '__main__':
    print('PyCharm')
    #print(wiki_tool("hp lovecraft"))
    print(weather.name, weather.description, weather.args_schema.schema())
    compound_interest_calc = CompoundInterestCalculator()
    print(compound_interest_calc.name, compound_interest_calc.description, compound_interest_calc.args_schema.schema())

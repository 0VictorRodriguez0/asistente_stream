# Importaciones:
import streamlit as st
from dotenv import load_dotenv
import os
import openai
import pandas as pd
import io
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getattr

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser

from langchain_experimental.utilities import PythonREPL
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.tools import tool
from langchain import hub


import os

#import seaborn as sns
import matplotlib.pyplot as plt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from email.message import EmailMessage
import ssl
import smtplib

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import AgentExecutor

from langchain.agents import create_tool_calling_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langchain_ollama import ChatOllama
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

#-----------------------------------------------------------------------

#Variables globales
filtro_df = pd.DataFrame() #dataframe global que guarda los filtros de consulta_df
mensaje_creado = "" #str global que guarda el mensaje creado por create_email
asunto_creado = "" #str global que guarda el asunto creado por create_email
#iniciar clave hola si no existe
if "filtro_df" not in st.session_state:
    st.session_state["filtro_df"] = pd.DataFrame()   # Puedes cambiar el valor inicial
if "mensaje_creado" not in st.session_state:
    st.session_state["mensaje_creado"] = ""  # Puedes cambiar el valor inicial
if "asunto_creado" not in st.session_state:
    st.session_state["asunto_creado"] = ""  # Puedes cambiar el valor inicial

# Inicializar estado de la sesión 'confirmar' en 'no' si no existe
if 'confirmar' not in st.session_state:
    st.session_state['confirmar'] = 'no'

#bandera para confirmar escenarios
if 'bandera' not in st.session_state:
    st.session_state['bandera'] = False

# print("papu ",st.session_state["filtro_df"])
# print("papu ",st.session_state["mensaje_creado"])
# print("papu ",st.session_state["asunto_creado"])

# Funciones:
def preprocesar_codigo(codigo):
    lineas = codigo.strip().split('\n')
    
    if not lineas[-1].strip().startswith('res ='):
        if lineas[-1].strip().startswith('return'):
            lineas[-1] = lineas[-1].replace('return', 'res =')
        else:
            lineas.append(f'res = {lineas[-1]}')

    codigo_preprocesado = '\n'.join(lineas)
    return codigo_preprocesado



@tool
def consulta_df(codigo):
    """
        Esta función recibe código de Python para realizar diferentes consultas sobre la tabla de datos df
        el código emplea diferentes funcionalidades de la librearía Pandas y se ejecuta 
        en un entorno seguro empleando la librería RestrictedPython
    """
    st.session_state["hola"] = "guarda"
    # global filtro_df  # Declarar filtro_df como global para modificarla
    
    try:

        codigo_preprocesado = preprocesar_codigo(codigo)
        # Compilar el código de forma restringida
        byte_code = compile_restricted(codigo_preprocesado, '<string>', 'exec')
        # Ejecutar el código
        exec(byte_code, namespace)
        # Acceder a los resultados (la variable debe ser un parámetro de la función)
        resultado = namespace.get('res')
        # Suponiendo que "resultado" es de tipo dataframe
        #print("Hola, entrad")
        if isinstance(resultado, (pd.DataFrame, pd.Series)):
            print("ENTRO")
            # Vaciar el dataframe filtro_df
            st.session_state["filtro_df"] = pd.DataFrame() 

            #agarrar los indices unicos de la columna matricula
            indices  = resultado.index
            nuevo_resultado = df.loc[indices]
            #Eliminar duplicados de la columna 'matricula'.
            alumnos_unicos = nuevo_resultado.drop_duplicates(subset='matricula', keep='first')
            st.session_state["filtro_df"] = alumnos_unicos.copy()
        print("FUERA")
        return resultado
    except Exception as e:
        print(f"Error al ejecutar código: {e.__class__.__name__}: {e}")
        return "No fue posible ejecutar el código!!"
    
    
#Creacion de graficas
@tool
def run_python_repl(code: str):
    """
    Especialmente diseñado para generar gráficos a partir de consultas sobre la tabla de datos df 
    o una tabla llamada resultados. Si el código incluye la creación de gráficos, asegúrate de importar 
    las bibliotecas necesarias, como matplotlib, y de llamar a plt.show() al final para visualizar el gráfico. 
    Además, ajusta automáticamente el tamaño y la configuración para cualquier tipo de gráfico, y guarda 
    las gráficas en el estado de la sesión.
    """
    # Asegura que realice graficas
    if "plt." in code and "plt.show()" not in code:
        code += "\nplt.show()"
    code = code.replace("plt.show()", "")
    code += "\nbuf = io.BytesIO()\nplt.savefig(buf, format='png')\nbuf.seek(0)\nst.session_state['current_graph'] = buf"

    try:
        exec(code)
        return {"output": "Gráfica generada."}  # Asegúrate de devolver algo si es necesario
    except Exception as e:
        st.error(f"Error al ejecutar el código: {e}")
        return {"output": str(e)}  # Retorna el error también
    
# Para envios de correos electronicos
# Para envios de correos electronicos
@tool
def create_email(mensaje,asunto):
    """
        Esta función recibe mensaje y el asunto del mensaje,
        Esta función crea un mensaje solo cuando el usuario lo solicite con saltos de linea y triple("),
        Esta función crea el asunto especificado por el usuario.
        El mensaje debe contar con las especificaciones del usuario que puede tener el nombre de las columnas del dataframe df
        columnas de la varible 'df' los datos de su matricula,p1,p2,p3,alumno,clave_asig,asignatura,seccion,periodo,num_docente,docente.
        Considera que el mensaje debe tener un formato y debe incluir saltos de línea para que sea más legible.

        considera el siguiente ejemplo del usuario solicitando crear un mensaje para los alumnos reprobados,el mensaje cuenta con su formato y saltos de linea(\n) :
          '\n Estimado/a {nombre},\n\n Se ha detectado que su calificación en el primer parcial de la asignatura {asignatura} es menor a 7.
           Le recomendamos asistir a las asesorías para mejorar su rendimiento.\n\n Saludos,\n Equipo Académico\n '
  
    """ 
    print("entra",st.session_state["bandera"])
    # global mensaje_creado, asunto_creado
    st.session_state["mensaje_creado"] = mensaje
    st.session_state["asunto_creado"] = asunto

    #Unir el mensaje con su asunto
    mensaje_con_asunto = asunto + "\n\n" + mensaje
    st.session_state["bandera"] = True 

    print("Salida", st.session_state["confirmar"])
    print("sale",st.session_state["bandera"])
    return mensaje_con_asunto


def send_email():
    print("Entro PAPUS")
    filtro_df = st.session_state["filtro_df"]
    mensaje_creado = st.session_state["mensaje_creado"]
    asunto_creado = st.session_state["asunto_creado"]

    if filtro_df.empty and mensaje_creado == "" and asunto_creado == "":
        st.warning("No se encuentran datos del mensaje y de los alumnos")
        st.session_state["confirmar"] = None
        if st.button("Reintentar"):
            st.session_state["bandera"] = False
            st.experimental_rerun()  # Reinicia la aplicación
    elif filtro_df.empty or mensaje_creado == "" or asunto_creado == "":
        vacias = []
        
        if filtro_df.empty:
            vacias.append("datos de alumnos")
        if mensaje_creado == "":
            vacias.append("mensaje")
        if asunto_creado == "":
            vacias.append("asunto del mensaje")
        
        st.warning(f"No se encuentran los siguientes datos: {', '.join(vacias)}")
        st.session_state["confirmar"] = None
        if st.button("Reintentar"):
            st.session_state["bandera"] = False
            st.experimental_rerun()  # Reinicia la aplicación
    else:
        #restablecer el estado confirmar, para no seguir enviando mensajes
        st.session_state['confirmar'] = None

        # proceso para enviar el mensaje
        password = os.getenv("PASSWORD")
        email_sender = "ceprintdo@gmail.com"

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(email_sender, password)
            for _, row in filtro_df.iterrows():
                email_reciver = str(row['matricula']) + "@ucaribe.edu.mx"  # Convertir la matrícula a cadena
                body = mensaje_creado.format(alumno=row['alumno'], asignatura=row['asignatura'], p1=row['p1'],
                                                p2=row['p2'],p3=row['p3'],final=row['final'],clave_asig=row['clave_asig'],
                                                seccion=row['seccion'],periodo=row['periodo'],
                                                num_docente=row['num_docente'],docente=row['docente'])
                em = EmailMessage()
                em["From"] = email_sender
                em["To"] = email_reciver
                em["Subject"] = asunto_creado
                em.set_content(body)
                smtp.sendmail(email_sender, email_reciver, em.as_string())
        # Para vaciar las variables nuevamente:
        st.session_state["filtro_df"] = pd.DataFrame()    # Vaciar el DataFrame
        st.session_state["mensaje_creado"] = ""  # Vaciar el string
        st.session_state["asunto_creado"] = ""  # Vaciar el string
        st.success("mensaje enviado con exito") 
        st.session_state["confirmar"] = None
        if st.button("Finalizar"):
            st.session_state["bandera"] = False
            st.experimental_rerun()  # Reinicia la aplicación   

#----------------------------------------------------------------------------------------------

# Configuración de la página
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="auto", 
    page_title="Asistente", 
    page_icon="./img/coco2.png",
)

st_callback = StreamlitCallbackHandler(st.container())


#with open('C:/Users/axeli/Desktop/Asistente/src/style.css') as f:
    #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cargar las variables de entorno
load_dotenv()

# Configura tu API Key
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Nombre del asistente
assistant_name = "Pedro"

# Avatares
avatares = {
    "assistant": "./img/coco2.png",
    "user": "./img/user2.png"
}

# Título de la aplicación
st.title(f"*Tu Aliado Académico*")

# Insertar CSS personalizado para efecto de resplandor
st.markdown(
    """
    <style>
    .body{
        background: white; 
    }
    .st-emotion-cache-janbn0 {
        border: 1px solid transparent;
        padding: 5px 10px;
        margin: 0px 7px;
        max-width: 50%;
        margin-left: auto;

        background: #2F2F2F;
        color: white;
        border-radius: 20px;

        flex-direction: row-reverse;
        text-align: right;
        
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Barra lateral (sidebar) con información del asistente
with st.sidebar:
    st.image("./img/cocodrilo.png", use_column_width=True, channels="RGB", output_format="auto", width="auto")
    st.sidebar.markdown("---")
    show_basic_info = st.sidebar.checkbox("Mostrar interacciones", value=True)
    if show_basic_info:
        st.sidebar.markdown(f"""
            ### Interacciones del asistente virtual:
            - *Analizar calificaciones*: Sube un archivo csv con las calificaciones de los estudiantes y el asistente las procesará automáticamente.
            - *Generar reportes*: El asistente puede crear reportes automáticos basados en los datos de los estudiantes.
            - *Identificar estudiantes en riesgo*: El asistente revisa los datos para identificar estudiantes que podrían estar en riesgo académico.
            - *Responder preguntas*: Haz preguntas sobre los datos cargados o pide ayuda con tareas académicas relacionadas.
            """)
    st.sidebar.markdown("---")
    #st.image("./img/logo.png", use_column_width=True, channels="RGB", output_format="auto", width="auto")

# Inicializar mensajes y estado de archivo
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"] = [{"role": "assistant", "content": f"👋 Hola, soy {assistant_name}."}]

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False

if "df" not in st.session_state:
    st.session_state["df"] = None
    
# Mostrar mensajes anteriores, junto con gráficas si existen
for msg in st.session_state["messages"]:
    avatar_image = avatares.get(msg["role"], None)
    st.chat_message(msg["role"], avatar=avatar_image).write(msg["content"])

    # Si el mensaje tiene una gráfica, mostrarla
    if msg.get("graph") is not None:
        st.image(msg["graph"])

# Mostrar el cargador de archivos solo si el archivo no ha sido subido
if not st.session_state["file_uploaded"]:
    uploaded_file = st.file_uploader("Sube un archivo CSV para comenzar:", type="csv")

    if uploaded_file is not None:
        # Leer el archivo CSV
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # Actualizar estado de archivo cargado
        st.session_state["file_uploaded"] = True

        # Mostrar mensaje de que el archivo ha sido cargado
        st.success("He procesado el archivo CSV correctamente. ¿En qué más te puedo ayudar con estos datos?")
else:
    df = st.session_state.get('df', None)
    
# Si el archivo se subio comenzamos a chatear   
if st.session_state["file_uploaded"]:
    #st.write(df)
    df = st.session_state.get('df', None)
    # Espacio de nombres seguro
    namespace = safe_globals.copy()
    namespace['_getattr_'] = default_guarded_getattr
    namespace['_getitem_'] = default_guarded_getitem
    namespace['__builtins__'] = None  # Deshabilitar acceso a built-ins inseguros
    namespace['pd'] = pd  # Permitir uso de pandas
    namespace['df'] =  df  # Permitir acceso al DataFrame

    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    template = """
        
        Considera que eres un asistente que apoya a un usuario a realizar consultas en una tabla de datos 
        empleando código python con la librería pandas. La tabla de datos se identifica con la variable df y
        guarda calificaciones de estudiantes en diferentes cursos. La tabla contiene las siguientes columnas:
        - 'matricula': variable categórica que identifica de forma única a un estudiante.
        - 'p1': variable numérica que indica la calificación del estudiante en el primer periodo parcial de la asignatura.
        - 'p2': variable numérica que indica la calificación del estudiante en el segundo periodo parcial de la asignatura.
        - 'p3': variable numérica que indica la calificación del estudiante en el tercer periodo parcial de la asignatura.
        - 'final': variable numérica que indica la calificación final del estudiante en la asignatura
        - 'alumno': variable categórica que guarda el nombre del estudiante.
        - 'clave_asig': variable categórica que indica alguna asignatura.
        - 'asignatura': variable categórica que indica el nombre de la asignatura.
        - 'seccion': variable categórica que indica la sección de la asignatura.
        - 'periodo': variable categórica que indica el periodo en que se imparte la asignatura.
        - 'num_docente': variable categórica que identifica al profesor de la asignatura.
        - 'docente': variable categórica que guarda el nombre del profesor de la asignatura.

        Para determinar el estado de aprobación de un estudiante en una asignatura, se considera una de las siguientes opciones:
        - Un estudiante aprueba si su calificación final, calculada como el promedio de 'p1', 'p2' y 'p3', es mayor o igual a 7.
        - Un estudiante reprueba si su calificación final es menor a 7.

        Los datos de un estudiante pueden distribuirse en varias filas, pues puede haber cursado varias asignaturas, por lo que 
        una misma matrícula puede estar presente en varias filas.
        Para evaluar tasas de reprobación de alguna asignatura, debes identificar el total de registros de la asignatura con calificación menor 
        a 7 y dividir por el total de registros de la asignatura.

        Considera que el resultado final de la consulta se debe guardar en una variable llamada 'res', por ejemplo, 
        si el usuario necesita conocer el total de estudiantes en la tabla, el código que debes generar es: res=df['matricula'].nunique()
        Otro ejemplo, si la consulta busca conocer la tasa de reprobación por asignatura, entonces el código que debes generar es: 
        res = df[df['final'] < 7].groupby('clave_asig')['final'].count()/df.groupby('clave_asig')['final'].count()
        res = asig_reprobacion.idxmax()
        res = df[df['clave_asig'] == asig_mayor_reprobacion]['asignatura'].unique()[0]
        Observa que en ambos ejemplos, la última línea de código asigna el resultado de la consulta a la variable 'res', cuida que en el código 
        que generes la última línea de código sea una asignación a la variable 'res', de otra forma, obtendrás resultados incorrectos.

        Solo usa la herramienta create_email solo cuando el usuario desea crear un mensaje o escribir un mensaje.
        considera que al crear mensajes si te especifica el usuario de una columna, ponerlo la columna de la varibale df estre corchetes, ejemplo,
        el mensaje debe tener la asignatura y calificación en el parcial 1, el mensaje que debe generar podria ser: 'su calificación en {{asignatura}} es de {{p1}}'

    
        Tu nombre es Pedro
        Si te saludan solo saluda
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("user", "Tu consulta actual es: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm_chat = ChatOllama(
        model="llama3.1",
        temperature=0, verbose=True
    )
    tools = [consulta_df, run_python_repl,create_email]
    llm_with_tools = llm_chat.bind_tools(tools)
    # Inicializar el modelo Llama3
    # Inicializar el modelo Llama3 y vincular las herramientas

    agent = ({"input": lambda x: x["input"],"agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),}| prompt | llm_with_tools | OpenAIToolsAgentOutputParser())

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    if prompt := st.chat_input("Escribe tu mensaje aquí..."):
        print("holis",prompt)
        # Chat del usuario para el asistente
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user", avatar=avatares["user"]).write(prompt)
        
        # Ejecutar la consulta y guardar la respuesta del asistente
        with st.chat_message("assistant", avatar=avatares["assistant"]):

            

            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke(
                {"input": prompt}, {"callbacks": [st_callback]}
            )
            
            # Asegúrate de que la respuesta esté definida
            if 'output' in response:
                # if st.session_state["bandera"] == True:
                #    message_content = (
                #         f"**Asunto:** {st.session_state['asunto_creado']}\n\n"
                #         f"**Mensaje:**\n\n{st.session_state['mensaje_creado']}\n\n"
                #         f"{response['output']}"
                #     )
                # else: 
                #     message_content = response["output"]
                message_content = response["output"]
            else:
                message_content = "No se obtuvo respuesta del asistente."

            current_graph = st.session_state.get("current_graph", None)
            
            st.session_state["messages"].append({"role": "assistant", "content": message_content, "graph": current_graph}
            )

            st.write(message_content)
            # if st.session_state["bandera"] == True:
            #     st.experimental_rerun()  

                    
            if current_graph:
                st.image(current_graph)

            # Limpiar la gráfica actual para no mostrarla en futuros mensajes
            st.session_state["current_graph"] = None  
            
    if st.session_state.get("bandera", False):
        st.write("¿Confirmar envío de mensajes?")
        
        # Botones de confirmación
        col1, col2 = st.columns(2)  # Crear dos columnas para los botones

        with col1:
            if st.button("Sí", key="confirmar_si"):
                send_email()  # Llamada a la función de envío de email
                st.success("Has confirmado el envío de mensajes.")
                st.session_state['bandera'] = False  # Resetear bandera

        with col2:
            if st.button("No", key="confirmar_no"):
                st.warning("Has cancelado el envío de mensajes.")
                st.session_state['bandera'] = False  # Resetear bandera
else:
        st.error("Por favor, sube primero el archivo CSV para poder interactuar con el asistente.",icon="⚠️")    



# ----------------------------------------------------------------------------------------------
# links de apoyo:
# https://github.com/AdieLaine/Streamly/blob/main/streamly.py
# https://github.com/albertgilopez/streamlit/blob/main/pregunta_a_tu_pdf.py


# Para realizarlo en .exe y ejecutar en windows
# https://discuss.streamlit.io/t/streamlit-deployment-as-an-executable-file-exe-for-windows-macos-and-android/6812/2
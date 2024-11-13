# Importaciones:
import streamlit as st
from dotenv import load_dotenv
import os
import openai
import pandas as pd
import io
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getattr
from datetime import datetime

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
from langchain.agents import AgentExecutor, create_react_agent, load_tools

from langchain.agents import create_openai_tools_agent

from langchain_openai import OpenAI
import os
from openai import OpenAI
#import seaborn as sns
import matplotlib.pyplot as plt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from email.message import EmailMessage
import ssl
import smtplib
import time
#Nueva libreria:
from langgraph.prebuilt import create_react_agent
import unicodedata
import seaborn as sns
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
if "res" not in st.session_state:
    st.session_state["res"] = None  # Puedes cambiar el valor inicial

# Inicializar estado de la sesión 'confirmar' en 'no' si no existe
if 'confirmar' not in st.session_state:
    st.session_state['confirmar'] = 'no'

#bandera para confirmar escenarios
if 'bandera' not in st.session_state:
    st.session_state['bandera'] = False
    
#bandera para confirmar escenarios
if 'file' not in st.session_state:
    st.session_state["file"] = False

# Funciones:

# Función para eliminar acentos de una cadena o una columna de DataFrame
def remove_accents(data):
    if isinstance(data, pd.Series):  # Si es una columna de un DataFrame
        return data.apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn') if isinstance(x, str) else x)
    elif isinstance(data, str):  # Si es una cadena individual
        return ''.join(c for c in unicodedata.normalize('NFD', data) if unicodedata.category(c) != 'Mn')
    else:
        raise ValueError("El argumento debe ser un string o una columna de DataFrame (pandas Series)")


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
    Esta función recibe código de Python para realizar diferentes consultas sobre el 
    DataFrame global 'df'. Utiliza diversas funcionalidades de la biblioteca Pandas, 
    permitiendo a los usuarios realizar un análisis de datos de manera efectiva. 
    El código se ejecuta en un entorno seguro, empleando la librería RestrictedPython, 
    lo que previene la ejecución de código malicioso y garantiza la integridad del sistema. 
    Esta función está diseñada para ser flexible y accesible, permitiendo a los usuarios 
    realizar operaciones personalizadas adaptadas a sus necesidades específicas.
    """
    
    try:
        codigo_preprocesado = preprocesar_codigo(codigo)

        # Compilar el código de forma restringida
        byte_code = compile_restricted(codigo_preprocesado, '<string>', 'exec')
        
        # Ejecutar el código
        exec(byte_code, namespace)
        
        # Acceder a los resultados
        resultado = namespace.get('res')
        #guardar res 
        st.session_state["res"] = resultado

        # Verificar si el resultado es un DataFrame o Series
        if isinstance(resultado, (pd.DataFrame, pd.Series)):
            st.session_state["filtro_df"] = pd.DataFrame()  # Vaciar el dataframe
            
            try:
                #agarrar los indices unicos de la columna matricula
                indices  = resultado.index
                nuevo_resultado = df.loc[indices]
            except:
                #si falla significa que no son filtros de df y solo return resultado
                return resultado
            
            # Eliminar duplicados de la columna 'matricula'
            alumnos_unicos = nuevo_resultado.drop_duplicates(subset='matricula', keep='first')
            st.session_state["filtro_df"] = alumnos_unicos.copy()

        return resultado
    except KeyError as e:
        return f"Error: La columna {str(e)} no se encuentra en el DataFrame."
    except Exception as e:
        print(f"Error al ejecutar código: {e.__class__.__name__}: {e}")
        return "No fue posible ejecutar el código!!"

#Creacion de graficas
@tool
def create_graph(code: str):
    """
    Si te pide una variable res, hazla para que funcione todo bien.
    Especialmente diseñado para generar gráficos a partir de consultas sobre la tabla de datos df 
    o una tabla llamada resultados. Si el código incluye la creación de gráficos, asegúrate de importar 
    las bibliotecas necesarias, como matplotlib, y de llamar a plt.show() al final para visualizar el gráfico. 
    Además, ajusta automáticamente el tamaño y la configuración para cualquier tipo de gráfico, y guarda 
    las gráficas en el estado de la sesión.
    """

    res =  st.session_state["res"] 
    print("indices", res.shape)
    # Asegura que realice graficas
    if "plt." in code and "plt.show()" not in code:
        code += "\nplt.show()"
    code = code.replace("plt.show()", "")
    code += "\nbuf = io.BytesIO()\nplt.savefig(buf, format='png',bbox_inches='tight')\nbuf.seek(0)\nst.session_state['current_graph'] = buf"

    try:
        exec(code)
        # st.session_state["res"] = None
        return {"output": "Gráfica generada."}  # Asegúrate de devolver algo si es necesario
    except Exception as e:
        st.error(f"Error al ejecutar el código: {e}")
        return {"output": str(e)}  # Retorna el error también
    
   
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
        print("Buenas"+ filtro_df["alumno"]) 
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


# Función para manejar la carga de archivos CSV
def cargar_archivo_csv():
    """Maneja la carga de un archivo CSV y permite al usuario cargar otro si lo desea."""
    
    if not st.session_state["file_uploaded"]:
        uploaded_file = st.file_uploader("Sube un archivo CSV para comenzar:", type="csv")

        if uploaded_file is not None:
            # Leer el archivo CSV
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            # Actualizar estado de archivo cargado
            st.session_state["file_uploaded"] = True
            st.session_state["file"] = True
            # Mostrar mensaje de que el archivo ha sido cargado
            st.success("He procesado el archivo CSV correctamente. ¿En qué más te puedo ayudar con estos datos?")
    else:
        df = st.session_state.get('df', None)

    return st.session_state.get('df', None)

def subir_archivo(): 
    if st.button("📎", key="upload_another"):
        st.session_state["file_uploaded"] = False
        cargar_archivo_csv()
        
def mensajes_anteriores():
    # Mostrar mensajes anteriores, junto con gráficas si existen
    for msg in st.session_state["messages"]:
        avatar_image = avatares.get(msg["role"], None)
        st.chat_message(msg["role"], avatar=avatar_image).write(msg["content"])
        
        # Si el mensaje tiene una gráfica, mostrarla
        if msg.get("graph") is not None:
            st.image(msg["graph"])

#----------------------------------------------------------------------------------------------

# Configuración de la página
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="auto", 
    page_title="Asistente", 
    page_icon="./img/coco2.png",
)


st_callback = StreamlitCallbackHandler(st.container())

load_dotenv()

# Configura tu API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Nombre del asistente
assistant_name = "Pedro"

# Avatares
avatares = {
    "assistant": "./img/coco2.png",
    "user": "./img/user2.png"
}



# Usar el estado de la sesión para rastrear la página actual
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Pasar a la ventana home
def home():
    st.session_state.page = 'home'

# Para pasar a la ventana tutorial
def tutorial():
    st.session_state.page = 'tutorial'
    
if st.session_state.page == 'home':
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
        
        # Usar un key fijo pero modificado para evitar duplicados
        # value_show_basic_info = False
        # if st.session_state["bandera"] == True:
        #     value_show_basic_info = False
        

    
        
        st.sidebar.markdown(f"""
                ### ¡Hola! 👋 Soy Pedro 🐊. ¡Bienvenido! 😄
                """)
        st.sidebar.markdown("---")
        col1, col2 = st.columns(2)
        col1.subheader("Mejora tu experiencia 🚀")
        if st.button("Ver Tutorial",type="primary", on_click=tutorial):
            pass
        st.sidebar.markdown("---")

    # Inicializar mensajes y estado de archivo
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["messages"] = [{"role": "assistant", "content": f"👋 Hola, soy {assistant_name}."}]

    if st.session_state.get("messages"):
                mensajes_anteriores()  


    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False

        
    if "df" not in st.session_state:
        st.session_state["df"] = None
        


    df = cargar_archivo_csv()


    # Si el archivo se subio comenzamos a chatear   
    if st.session_state["file_uploaded"]:
        if "snow_displayed" not in st.session_state:
            st.snow()
            st.session_state["snow_displayed"] = True  # Set the flag to indicate that st.snow() has been displayed
            
        df = st.session_state.get('df', None)
        # Espacio de nombres seguro
        namespace = safe_globals.copy()
        namespace['_getattr_'] = default_guarded_getattr
        namespace['_getitem_'] = default_guarded_getitem
        namespace['__builtins__'] = None  # Deshabilitar acceso a built-ins inseguros
        namespace['pd'] = pd  # Permitir uso de pandas
        namespace['df'] =  df  # Permitir acceso al DataFrame
        namespace['remove_accents'] = remove_accents

        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

        template = """
            Considera que eres un asistente que apoya a un usuario a realizar consultas en una tabla de datos 
            empleando código Python con la librería pandas. La tabla de datos se identifica con la variable df y
            guarda calificaciones de estudiantes en diferentes cursos. La tabla contiene las siguientes columnas:
            - 'matricula': variable categórica que identifica de forma única a un estudiante.
            - 'p1': variable numérica que indica la calificación del estudiante en el primer periodo parcial de la asignatura.
            - 'p2': variable numérica que indica la calificación del estudiante en el segundo periodo parcial de la asignatura.
            - 'p3': variable numérica que indica la calificación del estudiante en el tercer periodo parcial de la asignatura.
            - 'final': variable numérica que indica la calificación final del estudiante en la asignatura.
            - 'alumno': variable categórica que guarda el nombre del estudiante.
            - 'clave_asig': variable categórica que indica un ID para una asignatura.
            - 'asignatura': variable categórica que indica el nombre de la asignatura o materia.
            - 'seccion': variable categórica que indica la sección de la asignatura de tipo entero.
            - 'periodo': variable categórica que indica el periodo en que se imparte la asignatura.
            - 'num_docente': variable categórica que identifica al profesor de la asignatura.
            - 'docente': variable categórica que guarda el nombre del profesor de la asignatura.

            Para determinar el estado de aprobación de un estudiante en una asignatura, se considera lo siguiente:
            - Un estudiante aprueba si su calificación final, calculada como el promedio de 'p1', 'p2' y 'p3', es mayor o igual a 7.
            - Un estudiante reprueba si su calificación final es menor a 7.

            Los datos de un estudiante pueden distribuirse en varias filas, ya que puede haber cursado varias asignaturas, por lo que 
            una misma matrícula puede estar presente en varias filas. Para evaluar las tasas de reprobación de una asignatura, debes identificar el total de registros de la asignatura con calificación menor 
            a 7 y dividirlo por el total de registros de la asignatura.

            El resultado final de la consulta debe guardarse en una variable llamada 'res'. Por ejemplo, 
            si el usuario necesita conocer el total de estudiantes en la tabla, el código que debes generar es: 
            res = df['matricula'].nunique(). Otro ejemplo, si la consulta busca conocer la tasa de reprobación por asignatura, el código sería: 
            res = df[df['final'] < 7].groupby('clave_asig')['final'].count() / df.groupby('clave_asig')['final'].count().
            La última línea de código siempre debe asignar el resultado a la variable 'res'; de lo contrario, obtendrás resultados incorrectos.

            Solo usa la herramienta create_email cuando el usuario desea crear o escribir un mensaje. Por ejemplo,
            en la siguiente consulta no se menciona nada sobre escribir o enviar un mensaje: 
            "realiza una gráfica de los alumnos reprobados y no reprobados de la materia de ecuaciones diferenciales".
            Considera que al crear mensajes, si el usuario se refiere a una columna, debes incluirla de la variable df entre corchetes. Por ejemplo,
            el mensaje podría ser: 'su calificación en {{asignatura}} es de {{p1}}'.

            Si el usuario desea enviar un mensaje o correo, realiza un invoke de create_email. 
            Si el usuario solo escribe 'enviar un mensaje' o algo similar, también realiza un invoke de create_email.

            Siempre que el usuario se refiera a hacer una gráfica solo invoke create_graph,
            Si el usuario desea generar, realizar o crear una gráfica, invoke la herramienta create_graph.
            Por ejemplo, si la consulta busca generar una gráfica de los alumnos reprobados y no reprobados de ecuaciones diferenciales, 
            realiza invoke de create_graph, ten en cuenta que existen diferentes tipos de graficas, por ejemplo histogramas.

            Considera que cuentas con una función definida remove_accents() que permite quitar acentos de las columnas del df y de strings. 
            Por ejemplo, si el usuario desea conocer los alumnos reprobados en los parciales 1 y 2 de cálculo, el código sería:
            df[(df['p1'] < 7) | (df['p2'] < 7)].loc[remove_accents(df['asignatura']).str.contains(remove_accents('Cálculo'), case=False)].
            Usa remove_accents tanto en el df en la columna de asignatura como en la palabra 'Cálculo' para que la búsqueda no se vea afectada por acentos.

            Al realizar consultas de la base de datos o generar gráficas, si se está buscando una palabra específica, utiliza str.contains() con case=False
            para encontrar patrones similares a la palabra que se busca, y realiza búsquedas sin distinción entre mayúsculas y minúsculas.
            Ten en cuenta que algunas asignaturas pueden tener nombres largos que incluyen conjunciones o preposiciones, por lo cual siempre considera
            todas las palabras que puedan ser parte del nombre de la asignatura. 

            Recuerda que es importante utilizar el nombre completo de la materia al usar str.contains().
            siempre usa str.contains() en periodo para encontrar patrones deacuerdo a la consulta, ejemplo (df['periodo'].str.contains('202301', case=False),
            siempre usalo en comparaciones.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("user", "Tu consulta actual es: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        from langgraph.prebuilt import create_react_agent
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        
        from langgraph.prebuilt import ToolNode

        
        
        
        tools = [consulta_df,create_graph,create_email]
        tool_node = ToolNode(tools)
        
        llm_chat = ChatOpenAI(temperature=0, streaming=True)
        
        llm_chat = llm_chat.bind_tools(tools)
        
        agent = create_tool_calling_agent(llm_chat, tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        
        
        
        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            print("holis",prompt)
            # Chat del usuario para el asistente
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.chat_message("user", avatar=avatares["user"]).write(prompt)
            
            # Ejecutar la consulta y guardar la respuesta del asistente
            with st.chat_message("assistant", avatar=avatares["assistant"]):

                st_callback = StreamlitCallbackHandler(st.container())
                
                #Muestra todo el profeso de lenguaje natural
                response = agent_executor.invoke(
                    {"input": prompt}, {"callbacks": [st_callback]}
                )
                
                #Solo muestre respuesta
                #response = agent_executor.invoke({"input": prompt})
                
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
            col1, col2 = st.columns([.1,.7])  # Crear dos columnas para los botones

            with col1:
                if st.button("Sí", key="confirmar_si"):
                    send_email()  # Llamada a la función de envío de email
                    st.success("Has confirmado el envío de mensajes.")
                    st.session_state['bandera'] = False  # Resetear bandera

            with col2:
                if st.button("No", key="confirmar_no",type="primary"):
                    st.warning("Has cancelado el envío de mensajes.")
                    st.session_state['bandera'] = False  # Resetear bandera
        
    else:
            st.error("Por favor, sube primero el archivo CSV para poder interactuar con el asistente.",icon="⚠️")    


elif st.session_state.page == 'tutorial':
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
    if st.button("Regresar", type="primary", on_click=home):
        pass
    with st.sidebar:
        st.image("./img/cocodrilo.png", use_column_width=True, channels="RGB", output_format="auto", width="auto")
        st.sidebar.markdown("---")
        show_basic_info = st.sidebar.checkbox("Mostrar interacciones", value=False)
        if show_basic_info:
            st.sidebar.markdown(f"""
                ### Interacciones del asistente virtual:
                - *Analizar calificaciones*: Sube un archivo csv con las calificaciones de los estudiantes y el asistente las procesará automáticamente.
                - *Generar reportes*: El asistente puede crear reportes automáticos basados en los datos de los estudiantes.
                - *Identificar estudiantes en riesgo*: El asistente revisa los datos para identificar estudiantes que podrían estar en riesgo académico.
                - *Responder preguntas*: Haz preguntas sobre los datos cargados o pide ayuda con tareas académicas relacionadas.
                """)
        # Usar un key fijo pero modificado para evitar duplicados
        # value_show_basic_info = False
        # if st.session_state["bandera"] == True:
        #     value_show_basic_info = False
        
    
    st.write("### **Instrucciones de uso: 🕵️‍♂️**")
    with st.expander(
    "Intrucciones del asistente virtual Pedro", expanded=False
    ):
        
        st.write("""
        ### **Bienvenido al asistente virtual Pedro**  
        Este asistente está diseñado para ayudarte a consultar y analizar las calificaciones de estudiantes almacenadas en una tabla de datos usando Python con la librería pandas. Aquí tienes una lista de las funcionalidades que puedes utilizar:

        ### Funciones principales:
        
        1. **Realizar consultas sobre los datos**:
            - Puedes pedirle al asistente que realice diferentes tipos de consultas sobre los datos de las calificaciones. 
            - Ejemplo: _"Muestra los estudiantes reprobados en la asignatura de cálculo."_
            - Ejemplo: _"¿Cuántos estudiantes han aprobado el primer parcial de ecuaciones diferenciales?"_
        
        2. **Generar gráficos**:
            - Puedes solicitarle al asistente que genere gráficos para visualizar los datos. 
            - Ejemplo: _"Genera una gráfica de barras de los alumnos reprobados y aprobados en álgebra lineal."_
            - Ejemplo: _"Muestra un gráfico de líneas con las calificaciones finales de todos los estudiantes en la asignatura de física."_

        3. **Enviar mensajes personalizados**:
            - El asistente puede generar y enviar mensajes basados en la información de las calificaciones de los estudiantes.
            - Ejemplo: _"Envía un correo a los estudiantes que reprobaron el segundo parcial de programación."_
            - Ejemplo: _"Genera un mensaje para informar a los estudiantes sus calificaciones finales en álgebra."_

        4. **Cálculo de tasas de reprobación**:
            - Puedes calcular la tasa de reprobación de una asignatura con base en las calificaciones finales.
            - Ejemplo: _"Calcula la tasa de reprobación de cálculo."
            
        ### Instrucciones adicionales:
        **Variables que maneja el asistente**:
        
        - 'matricula': Identificación única del estudiante.
        - 'p1', 'p2', 'p3': Calificaciones de los parciales 1, 2 y 3.
        - 'final': Calificación final (promedio de p1, p2, p3).
        - 'alumno': Nombre del estudiante.
        - 'clave_asig': ID único de la asignatura.
        - 'asignatura': Nombre de la asignatura.
        - 'seccion': Sección de la asignatura.
        - 'periodo': Periodo académico de la asignatura.
        - 'num_docente': Identificación del docente.
        - 'docente': Nombre del docente.
            
        ### Ejemplos de consultas comunes:
    
        - _"¿Cuántos estudiantes reprobaron el primer parcial de álgebra?"_
        - _"Genera una gráfica de pastel de los estudiantes aprobados y reprobados en física."_
        - _"Envía un mensaje a los estudiantes que tienen una calificación final menor a 7 en ecuaciones diferenciales."_  

        Si necesitas más ayuda, simplemente pregunta al asistente sobre lo que quieras consultar o realizar.
        """)
    st.write("")
    st.write("### **Casos de uso** 📈")        

    option = st.selectbox(
        "Selecciona 👇",
        ["Selecciona una opción", "Consultas", "Generaracion de graficas"]
    )

    if option == "Consultas":
        st.write("Haz seleccionado Consultas")
        

        st.chat_message("user", avatar=avatares["user"]).write("Lista de la tasa de reprobacion de cada asignatura")
    
        with st.chat_message("assistant", avatar=avatares["assistant"]):
            with st.spinner("Thinking..."):
                time.sleep(2)

            
            st.write(""" 
                     La tasa de reprobación de cada asignatura es la siguiente:

                    - Cálculo diferencial: 41.49%
                    -  álculo integral: 31.85%
                    - Cálculo vectorial: 30.91%
                    - Ecuaciones diferenciales: 27.99%
                    - Estadística analítica: 19.89%
                    - Probabilidad y estadística: 29.56%
                    - Propedéutico de matemáticas para ingenierías: 34.14%
                    - Álgebra lineal: 28.17%""")
        
    elif option == "Generaracion de graficas":
        st.write("Haz seleccionado Creacion de graficas")
        
        st.chat_message("user", avatar=avatares["user"]).write("¿Cuántos estudiantes hay en la tabla?")
        
        with st.chat_message("assistant", avatar=avatares["assistant"]):
            with st.spinner("Thinking..."):
                time.sleep(2)
                
                st.write("Se ha generado la gráfica que muestra el total de alumnos en cada asignatura. ¿Hay algo más en lo que pueda ayudarte?")
                image_path = "./img/grafica.jpg"  # Reemplaza esto con la ruta de tu imagen
                st.image(image_path, use_column_width=True)
                
    #Mostrar codigo por si lo pide
    #st.code("""
    #res = df[(df['asignatura'].str.contains('Matemáticas', case=False)) & (df['final'] < 7)].shape[0] / df[df['asignatura'].str.contains('Matemáticas', case=False)].shape[0]
    #""", language='python')
    
    
    
# ----------------------------------------------------------------------------------------------
# links de apoyo:
# https://github.com/AdieLaine/Streamly/blob/main/streamly.py
# https://github.com/albertgilopez/streamlit/blob/main/pregunta_a_tu_pdf.py


# Para realizarlo en .exe y ejecutar en windows
# https://discuss.streamlit.io/t/streamlit-deployment-as-an-executable-file-exe-for-windows-macos-and-android/6812/2
import spacy
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd



fecha_actual = datetime.now().strftime('%d-%m-%Y')
ruta_csv_scrape_diario = f'Noticias_{fecha_actual}.csv'
df_noticias = pd.read_csv(ruta_csv_scrape_diario)


#MODELO 1 NLP
#Este código primero muestra las categorías disponibles y permite al usuario seleccionar una categoría específica. Luego, muestra las noticias disponibles en esa categoría para que el usuario elija una noticia en particular para analizar las entidades nombradas en su contenido.


# Función para graficar la frecuencia de las entidades nombradas en una noticia específica
def graficar_entidades_noticia_especifica(df_noticias, indice_noticia):
    # Cargar el modelo de spaCy
    nlp = spacy.load('en_core_web_sm')

    # Obtener el texto de la noticia seleccionada
    texto_noticia = df_noticias['Contenido'].iloc[indice_noticia]
    titulo_noticia = df_noticias['Título'].iloc[indice_noticia]

    # Procesar el texto con spaCy para la identificación de entidades
    doc = nlp(texto_noticia)

    # Obtener las entidades nombradas del texto
    entidades = [ent.label_ for ent in doc.ents]

    # Contar la frecuencia de cada tipo de entidad
    frecuencia_entidades = {ent: entidades.count(ent) for ent in set(entidades)}

    # Graficar la frecuencia de las entidades nombradas
    plt.figure(figsize=(8, 6))
    plt.bar(frecuencia_entidades.keys(), frecuencia_entidades.values())
    plt.xlabel('Tipo de Entidad')
    plt.ylabel('Frecuencia')
    plt.title(f'Frecuencia de Entidades Nombradas en la Noticia: {titulo_noticia}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Mostrar las opciones de categorías disponibles al usuario
categorias_disponibles = df_noticias['Categoría'].unique()
print("Categorías disponibles para analizar:")
for idx, categoria in enumerate(categorias_disponibles):
    print(f"{idx}: {categoria}")

# Obtener la entrada del usuario para seleccionar una categoría específica
while True:
    try:
        indice_categoria_seleccionada = int(input("Ingrese el número de índice de la categoría que desea analizar (o 'q' para salir): "))
        if indice_categoria_seleccionada == 'q':
            break
        if indice_categoria_seleccionada not in range(len(categorias_disponibles)):
            raise ValueError("Índice no válido. Por favor, ingrese un número de índice válido.")

        # Filtrar las noticias por la categoría seleccionada
        noticias_categoria_seleccionada = df_noticias[df_noticias['Categoría'] == categorias_disponibles[indice_categoria_seleccionada]]

        # Mostrar las noticias disponibles en la categoría seleccionada
        print("\nNoticias disponibles para la categoría seleccionada:")
        for idx, titulo in enumerate(noticias_categoria_seleccionada['Título']):
            print(f"{idx}: {titulo}")

        # Obtener la entrada del usuario para seleccionar una noticia específica de esa categoría
        while True:
            try:
                indice_noticia_seleccionada = int(input("Ingrese el número de índice de la noticia que desea analizar (o 'q' para salir): "))
                if indice_noticia_seleccionada == 'q':
                    break
                if indice_noticia_seleccionada not in range(len(noticias_categoria_seleccionada)):
                    raise ValueError("Índice no válido. Por favor, ingrese un número de índice válido.")

                # Llamar a la función para generar la gráfica con la noticia específica
                graficar_entidades_noticia_especifica(noticias_categoria_seleccionada, indice_noticia_seleccionada)
                break
            except ValueError as e:
                print(e)
        break
    except ValueError as e:
        print(e)



#MODELO 2 RECOMENDACIONES DE NOTICIAS 
#El código busca noticias en un DataFrame basadas en palabras clave ingresadas por el usuario. 
#Utiliza el módulo fuzzywuzzy para encontrar coincidencias parciales en los títulos de las noticias 
#con las palabras clave. Luego, muestra al usuario las coincidencias encontradas, permitiéndole 
#seleccionar un número para obtener recomendaciones relacionadas con la noticia seleccionada. #
#Este proceso se repite hasta que el usuario decida salir indicando 'no'.



from fuzzywuzzy import fuzz

continuar = True  # Variable para controlar la ejecución del bucle
noticias_encontradas = set()
recomendaciones_unicas = set()

while continuar:
    print("Ingrese palabras clave para encontrar noticias relacionadas:")
    palabras_clave = input().split()

    for keyword in palabras_clave:
        for index, row in df_noticias.iterrows():
            title = row['Título']
            content = row['Contenido']
            if keyword.lower() in title.lower() or keyword.lower() in content.lower():  # Buscar palabra clave en título o contenido
                ratio_title = fuzz.partial_ratio(keyword.lower(), title.lower())
                ratio_content = fuzz.partial_ratio(keyword.lower(), content.lower())
                if ratio_title >= 80 or ratio_content >= 80:
                    noticias_encontradas.add(row['Título'])
                    recomendaciones_unicas.add(row['Título'])  # Almacenar identificador único

    if noticias_encontradas:
        print("Títulos encontrados con las palabras claves ingresadas:")
        for i, noticia in enumerate(noticias_encontradas, start=1):
            print(f"{i}. {noticia}")

        print("\nSeleccione un número de título de la lista para obtener recomendaciones:")
        num_seleccionado = int(input())
        if 1 <= num_seleccionado <= len(noticias_encontradas):
            selected_title = list(noticias_encontradas)[num_seleccionado - 1]

            # Filtrar recomendaciones basadas en el título o contenido con la palabra clave introducida
            filtered_recs = [rec for rec in recomendaciones_unicas if any(keyword.lower() in df_noticias[df_noticias['Título'] == rec]['Contenido'].str.lower().values[0] or keyword.lower() in df_noticias[df_noticias['Título'] == rec]['Título'].str.lower().values[0] for keyword in palabras_clave)]

            print("\nNoticias recomendadas similares a la que seleccionó:")
            for i, rec in enumerate(filtered_recs, start=1):
                print(f"{i}. {rec}")
        else:
            print("Número seleccionado no válido. Por favor, seleccione un número de la lista.")
    else:
        print("No se encontraron títulos relacionados con las palabras clave ingresadas.")

    print("\n¿Desea buscar más noticias? (Sí o No)")
    respuesta = input().lower()
    if respuesta != 'si':
        continuar = False



#MODELO 3 CLASIFICACION DE CLUSTERS
        
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from nltk.corpus import stopwords
import string
import warnings

# Filtrar advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Obtener el contenido de las noticias del DataFrame
news_corpus = df_noticias['Contenido'].tolist()

# Definir una lista personalizada de stop words
custom_stop_words = [
    # ... (otras palabras comunes),
    # Lista de stop words en español
    'de', 'la', 'el', 'los', 'y', 'o', 'pero', 'por', 'con', 'sin', 'más', 'menos',
    'un', 'una', 'unos', 'unas', 'al', 'del', 'a', 'ante', 'tras', 'desde', 'hasta',
    'como', 'cuando', 'donde', 'quien', 'cual', 'cuyo', 'sí', 'no', 'si', 'tal', 'para',
    'eso', 'esto', 'aquí', 'ahí', 'allí', 'entonces', 'mientras', 'aunque', 'porque', 'pues',
    'primero', 'segundo', 'tercero', 'cuarto', 'quinto', 'sexto', 'séptimo', 'octavo', 'noveno', 'décimo',
    'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'usted', 'ustedes',
    'mi', 'tu', 'su', 'nuestro', 'vuestro', 'suyo', 'mío', 'tuyo', 'suyo', 'nuestro', 'vuestro', 'suyo',
    'mía', 'tuya', 'suya', 'nuestra', 'vuestra', 'suya', 'míos', 'tuyos', 'suyos', 'nuestros', 'vuestros', 'suyos',
    'mías', 'tuyas', 'suyas', 'nuestras', 'vuestras', 'suyas', 'que', 'cómo', 'cuál', 'cuáles', 'cuando', 'cuánto', 'cuánta',
    'cuántos', 'cuántas', 'dónde', 'adónde', 'a dónde', 'cuán', 'qué', 'quién', 'quienes', 'cuyo', 'cuya', 'cuyos',
    'cuyas', 'lo', 'los', 'las', 'el', 'la', 'uno', 'una', 'unos', 'unas', 'alguno', 'alguna', 'algunos', 'algunas',
    'otro', 'otra', 'otros', 'otras', 'mismo', 'misma', 'mismos', 'mismas', 'tan', 'tanto', 'tanta', 'tantos', 'tantas',
    'mucho', 'mucha', 'muchos', 'muchas', 'poco', 'poca', 'pocos', 'pocas', 'cada', 'cualquier', 'cualquiera', 'cualesquiera',
    'demás', 'ese', 'esa', 'esos', 'esas', 'estos', 'estas', 'aquel', 'aquella', 'aquellos', 'aquellas', 'este', 'esta', 'estos', 'estas',
    'aquel', 'aquella', 'aquellos', 'aquellas', 'estos', 'estas', 'mi', 'tu', 'su', 'nuestro', 'vuestro', 'suyo', 'mío', 'tuyo', 'suyo',
    'nuestro', 'vuestro', 'suyo', 'mía', 'tuya', 'suya', 'nuestra', 'vuestra', 'suya', 'míos', 'tuyos', 'suyos', 'nuestros', 'vuestros',
    'suyos', 'mías', 'tuyas', 'suyas', 'nuestras', 'vuestras', 'suyas', 'tal', 'tamaña', 'cual', 'cuales', 'cuantos', 'cuantas', 'cuanto',
    'cuanta', 'cuánto', 'cuánta', 'de', 'la', 'en', 'ver', 'comentarios', 'comentario', 'portada', 'tema', 'foto', 'lea más', 'regresar', 
    'keywords', 'lee', 'también',
]

# Obtener las stop words en español de NLTK
stop_words = set(stopwords.words('spanish'))

# Agregar las stop words personalizadas a la lista
stop_words.update(custom_stop_words)

# Puntuaciones
punctuation = set(string.punctuation)

# Vectorizar el texto utilizando TF-IDF con las stop words personalizadas
tfidf = TfidfVectorizer(stop_words=list(stop_words))
tfidf_matrix = tfidf.fit_transform(news_corpus)

# Aplicar K-Means para agrupar las noticias en clusters
num_clusters = 5  # Define el número de clusters deseado
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# Agregar los resultados de clustering al DataFrame
df_noticias['Cluster'] = clusters

# Visualizar palabras más frecuentes en cada cluster usando WordCloud
for cluster_id in range(num_clusters):
    cluster_words = " ".join(df_noticias[df_noticias['Cluster'] == cluster_id]['Contenido'])

    # Filtrar palabras de stopwords
    cluster_words_filtered = " ".join(word for word in cluster_words.split() if word.lower() not in stop_words and word not in punctuation)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_words_filtered)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Cluster {cluster_id} - Word Cloud')
    plt.show()
    
# Visualizar relación entre palabras usando MDS y un gráfico de dispersión
mds_embeddings = MDS(n_components=2, dissimilarity="euclidean").fit_transform(tfidf_matrix.toarray())
plt.figure(figsize=(12, 8))
scatter = plt.scatter(mds_embeddings[:, 0], mds_embeddings[:, 1], c=clusters, cmap='viridis', s=10)

# Agregar leyendas
plt.title('Visualización de Clusters con MDS')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')

# Crear leyenda para los clusters
legend_labels = [f'Cluster {i}' for i in range(num_clusters)]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Clusters')

plt.show()







#MODELO 4

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage

# Suponiendo que ya tienes tu DataFrame df_noticias con las columnas 'Título' y 'Contenido'

# Vectorizar el texto utilizando TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_noticias['Contenido'])

# Aplicar K-Means para agrupar las noticias en clusters
num_clusters = 5  # Define el número de clusters deseado
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)

# Agregar los resultados de clustering al DataFrame
df_noticias['Cluster'] = clusters

# Reducir la dimensionalidad con PCA para visualización
pca = PCA(n_components=2)
tfidf_matrix_pca = pca.fit_transform(tfidf_matrix.toarray())

# 1. Gráfico de dispersión con colores para cada cluster
plt.figure(figsize=(8, 6))
for cluster_id in range(num_clusters):
    plt.scatter(tfidf_matrix_pca[df_noticias['Cluster'] == cluster_id, 0],
                tfidf_matrix_pca[df_noticias['Cluster'] == cluster_id, 1], label=f'Cluster {cluster_id}')

plt.title('Gráfico de Dispersión de Clusters')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Gráfico de dispersión interactivo (Plotly)
fig = px.scatter(x=tfidf_matrix_pca[:, 0], y=tfidf_matrix_pca[:, 1], color=df_noticias['Cluster'])
fig.update_layout(title='Gráfico de Dispersión Interactivo de Clusters')
fig.show()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pickle
from datetime import datetime
from wordcloud import WordCloud 
from textblob import TextBlob 
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords

st.set_option('deprecation.showPyplotGlobalUse', False)

# Cargar datos
fecha_actual = datetime.now().strftime('%d-%m-%Y')
ruta_csv_scrape_diario = f'Noticias_{fecha_actual}.csv'
df_noticias = pd.read_csv(ruta_csv_scrape_diario)

# Cargar las variables necesarias para el modelo de Machine Learning
with open('variables_modelo.pkl', 'rb') as f:
    variables_modelo = pickle.load(f)

# Extraer las variables
X_train = variables_modelo['X_train']
X_test = variables_modelo['X_test']
y_train = variables_modelo['y_train']
y_test = variables_modelo['y_test']
vectorizer = variables_modelo['vectorizer']
selector = variables_modelo['selector']
svm_model = variables_modelo['svm_model']

# Utilizar todo el conjunto de prueba para la evaluación
X_test_sampled = X_test
X_test_tfidf = vectorizer.transform(X_test_sampled)
X_test_selected = selector.transform(X_test_tfidf)

# Realizar predicciones con el modelo SVM
y_pred = svm_model.predict(X_test_selected)

# Título de la aplicación
st.title('Análisis de Noticias y Machine Learning')

# Sección de Análisis Exploratorio de Datos (EDA)
st.markdown("## Análisis Exploratorio de Datos (EDA)")

# Mostrar la tabla de resultados
st.write('Tabla de Noticias:', df_noticias)

# Función para visualizar la frecuencia de noticias por categoría
def visualizar_frecuencia_por_categoria():
    fig = px.bar(df_noticias, x='Categoría', title='Frecuencia de Noticias por Categoría')
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig)

# Función para visualizar la cantidad de noticias por autor (Top 10)
def visualizar_cantidad_por_autor():
    st.subheader('TOP 10 de Autores con más noticias')
    cantidad_por_autor = df_noticias['Autor'].value_counts().head(10).reset_index()
    cantidad_por_autor.columns = ['Autor', 'Cantidad']
    fig = px.bar(cantidad_por_autor, x='Autor', y='Cantidad', title='Cantidad de Noticias por Autor (Top 10)')
    st.plotly_chart(fig)

# Función para visualizar el gráfico de líneas de la cantidad de noticias desde la última fecha hasta la más reciente
def visualizar_grafico_lineas(df_noticias):
    st.subheader('Cantidad de noticias por mes')
    df_noticias['Fecha'] = pd.to_datetime(df_noticias['Fecha'], errors='coerce')
    df_noticias = df_noticias.sort_values(by='Fecha', ascending=False)
    fig = px.line(df_noticias, x='Fecha', y=range(1, len(df_noticias) + 1), title='Cantidad de Noticias desde la Última Fecha hasta la Más Reciente')
    fig.update_layout(xaxis_title='Fecha', yaxis_title='Cantidad de Noticias')
    st.plotly_chart(fig)

# Función para visualizar el WordCloud de las palabras más comunes en las noticias
def visualizar_wordcloud():
    st.subheader('Palabras más comunes en las noticias')
    todas_palabras = " ".join(df_noticias['Contenido']).split()
    stop_words = set(stopwords.words('spanish'))
    additional_stopwords = ['portada', 'dos', 'ser', 'año', 'keywords', 'ser', 'ver', 'tema', 'puede', 'foto:', 'años', 'nuevo', 'tras', 'solo', 'Foto:', 'hacer', 'mejor', 'San', 'san', 'sido', 'así', 'cada']
    stop_words.update(additional_stopwords)
    palabras_filtradas = [word for word in todas_palabras if word.lower() not in stop_words and len(word) > 2]
    palabras_comunes_filtradas = Counter(palabras_filtradas).most_common(20)
    wordcloud_filtrado = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(palabras_comunes_filtradas))
    fig = px.imshow(wordcloud_filtrado)
    fig.update_layout(title='WordCloud de las Palabras Más Comunes en las Noticias')
    st.plotly_chart(fig)

# Función para visualizar el análisis de sentimientos promedio por categoría
def visualizar_analisis_sentimientos():
    st.subheader('Polaridad Promedio por Categoría')
    df_noticias['Polaridad'] = df_noticias['Contenido'].apply(lambda x: TextBlob(x).sentiment.polarity)
    polaridad_promedio = df_noticias.groupby('Categoría')['Polaridad'].mean().reset_index()
    fig = px.bar(polaridad_promedio, x='Categoría', y='Polaridad', title='Polaridad Promedio por Categoría', color='Polaridad', color_continuous_scale='viridis')
    st.plotly_chart(fig)

# Función para visualizar la distribución de frecuencia de las palabras
def visualizar_distribucion_frecuencia_palabras():
    st.subheader('Distribución de Frecuencia de las Palabras')
    stop_words = set(stopwords.words('spanish'))
    additional_stopwords = ['portada', 'dos', 'ser', 'año', 'keywords', 'ser', 'ver', 'tema', 'puede', 'foto:', 'años', 'nuevo', 'tras', 'solo', 'Foto:', 'hacer', 'mejor', 'San', 'san', 'sido', 'así', 'cada']
    stop_words.update(additional_stopwords)
    todas_palabras = " ".join(df_noticias['Contenido']).split()
    palabras_filtradas = [word for word in todas_palabras if word.lower() not in stop_words and len(word) > 2]
    palabras_comunes_filtradas = Counter(palabras_filtradas).most_common(20)
    
    # Crear un DataFrame a partir de las palabras comunes filtradas
    df_palabras_comunes = pd.DataFrame(palabras_comunes_filtradas, columns=['Palabra', 'Frecuencia'])
    
    # Crear gráfico interactivo con Plotly Express
    fig = px.bar(df_palabras_comunes, x='Palabra', y='Frecuencia', title='Distribución de Frecuencia de las Palabras')
    st.plotly_chart(fig)

# Visualizar gráficas de EDA
visualizar_frecuencia_por_categoria()
visualizar_cantidad_por_autor()
visualizar_grafico_lineas(df_noticias)
visualizar_wordcloud()
visualizar_analisis_sentimientos()
visualizar_distribucion_frecuencia_palabras()

# Sección de Machine Learning centrada
st.markdown("<h1 style='text-align: center;'>Sección de Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Poniendo a prueba los resultados del modelo Machine Learning</p>", unsafe_allow_html=True)

# Depuración: Mostrar la forma de y_test e y_pred en el panel de Streamlit
st.subheader('Información de Depuración')
st.write("Forma de y_test:", y_test.shape)
st.write("Forma de y_pred:", y_pred.shape)

# Visualizar las predicciones
st.subheader('Resultados de las Predicciones')
resultados_df = pd.DataFrame({'Texto': X_test_sampled, 'Predicción': y_pred})
st.write(resultados_df)

# Depuración: Mostrar los valores únicos en y_test e y_pred en tres columnas con gráfica
st.subheader('Valores Únicos en y_test y y_pred con Gráfica de Efectividad')
unique_values_df = pd.DataFrame({
    'y_test': pd.unique(y_test),
    'y_pred': pd.unique(y_pred),
    'Efectividad': [''] * max(len(pd.unique(y_test)), len(pd.unique(y_pred)))  # Columna en blanco para separar
})

# Calcular la efectividad del modelo
accuracy = (y_test == y_pred).mean()
unique_values_df['Efectividad'] = unique_values_df.apply(lambda row: f'{accuracy:.2%}', axis=1)

# Mostrar el DataFrame
st.write(unique_values_df)

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Crear un DataFrame de la matriz de confusión para mejorar la visualización
conf_df = pd.DataFrame(conf_matrix, index=svm_model.classes_, columns=svm_model.classes_)

# Graficar la matriz de confusión con seaborn
st.subheader('Matriz de Confusión del Modelo SVM')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, cmap='Blues', fmt='g', ax=ax)
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Reales')
st.pyplot(fig)

# Calcular el reporte de clasificación
classification_rep = classification_report(y_test, y_pred, target_names=svm_model.classes_)

# Mostrar el reporte de clasificación en Streamlit
st.subheader('Reporte de Clasificación del Modelo SVM')
st.text(classification_rep)

# Muestra algunos ejemplos de datos reales
st.subheader('Ejemplos de Datos de Prueba Real')
st.write(X_test.head())
st.write(y_test.head())

# Visualiza las distribuciones de las clases
st.subheader('Distribución de Clases en Datos de Prueba Real')
st.bar_chart(y_test.value_counts())

# Añade Información de Depuración
st.subheader('Información Adicional de Depuración')
st.write("Forma de X_test:", X_test.shape)
st.write("Forma de y_test:", y_test.shape)

# Verifica la Predicción en un Ejemplo Aleatorio
random_example_index = st.slider('Elije un ejemplo aleatorio', 0, len(X_test) - 1, 0)  # Slider para elegir el índice aleatorio
st.subheader('Verificación de Predicción en un Ejemplo Aleatorio')
st.write("Texto:", X_test.iloc[random_example_index])
st.write("Etiqueta Real:", y_test.iloc[random_example_index])
st.write("Predicción del Modelo:", y_pred[random_example_index])

# Ejecutar la aplicación
if __name__ == '__main__':
    st.write('¡Gracias por compartir sus conocimientos Juan y Valentín, Exitos!')

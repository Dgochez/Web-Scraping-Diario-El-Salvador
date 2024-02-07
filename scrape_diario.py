import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import concurrent.futures
import pandas as pd
from datetime import datetime

def extraer_enlaces_unicos(url, url_especifica):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            enlaces = {a['href'] for a in soup.find_all('a', href=True) if url_especifica in a['href']}
            return enlaces
        else:
            print(f"No se pudo obtener el contenido de la URL {url}. Estado de la respuesta: {response.status_code}")
    except Exception as e:
        print(f"Error al procesar la URL {url}: {e}")
    return set()

# URL base
url_base = "https://www.elsalvador.com/category/noticias/"

# URL específica que estás buscando
url_especifica_noticias = 'https://www.elsalvador.com/noticias/'

# Lista para almacenar todos los enlaces del apartado "Noticias"
lista_noticias = []

# Conjunto para almacenar enlaces únicos
enlaces_unicos = set()

# Número máximo de páginas a extraer (ajusta según sea necesario)
max_paginas = 13

# Recorre cada sección de noticias
for seccion_url in ["nacional", "internacional", "negocios", "gente-y-empresas"]:
    for pagina in range(1, max_paginas + 1):
        url = f"{url_base}{seccion_url}/page/{pagina}/"
        print(f"Enlaces encontrados en {url} que contienen '{url_especifica_noticias}':")
        enlaces = extraer_enlaces_unicos(url, url_especifica_noticias)
        for enlace in enlaces:
            print(enlace)
            enlaces_unicos.add(enlace)
            lista_noticias.append(enlace)
        print("\n")

def extraer_enlaces_unicos(url, url_especifica):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            enlaces = {a['href'] for a in soup.find_all('a', href=True) if url_especifica in a['href']}
            return enlaces
        else:
            print(f"No se pudo obtener el contenido de la URL {url}. Estado de la respuesta: {response.status_code}")
    except Exception as e:
        print(f"Error al procesar la URL {url}: {e}")
    return set()

# URL base
url_base = "https://www.elsalvador.com/category/entretenimiento/"

# URL específica que estás buscando
url_especifica_entretenimiento = 'https://www.elsalvador.com/entretenimiento/'

# Lista para almacenar todos los enlaces del apartado "ENTRETENIMIENTO"
lista_entretenimiento = []

# Conjunto para almacenar enlaces únicos
enlaces_unicos = set()

# Número máximo de páginas a extraer (ajusta según sea necesario)
max_paginas = 13

# Recorre cada sección de Entretenimiento
for seccion_url in ["espectaculos","tecnologia","turismo","cultura"]:
    for pagina in range(1, max_paginas + 1):
        url = f"{url_base}{seccion_url}/page/{pagina}/"
        print(f"Enlaces encontrados en {url} que contienen '{url_especifica_entretenimiento}':")
        enlaces = extraer_enlaces_unicos(url, url_especifica_entretenimiento)
        for enlace in enlaces:
            print(enlace)
            enlaces_unicos.add(enlace)
            lista_entretenimiento.append(enlace)
        print("\n")


def extraer_enlaces_unicos(url, url_especifica):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            enlaces = {a['href'] for a in soup.find_all('a', href=True) if url_especifica in a['href']}
            return enlaces
        else:
            print(f"No se pudo obtener el contenido de la URL {url}. Estado de la respuesta: {response.status_code}")
    except Exception as e:
        print(f"Error al procesar la URL {url}: {e}")
    return set()

# URL base
url_base = "https://www.elsalvador.com/category/vida/"

# URL específica que estás buscando
url_especifica_vida = 'https://www.elsalvador.com/vida/'

# Lista para almacenar todos los enlaces del apartado "Vida"
lista_vida = []

# Conjunto para almacenar enlaces únicos
enlaces_unicos = set()

# Número máximo de páginas a extraer (ajusta según sea necesario)
max_paginas = 13

# Recorre cada sección de vida
for seccion_url in ["mujeres","salud"]:
    for pagina in range(1, max_paginas + 1):
        url = f"{url_base}{seccion_url}/page/{pagina}/"
        print(f"Enlaces encontrados en {url} que contienen '{url_especifica_vida}':")
        enlaces = extraer_enlaces_unicos(url, url_especifica_vida)
        for enlace in enlaces:
            print(enlace)
            enlaces_unicos.add(enlace)
            lista_vida.append(enlace)
        print("\n")


def extraer_enlaces_unicos(url, url_especifica):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            enlaces = {a['href'] for a in soup.find_all('a', href=True) if url_especifica in a['href']}
            return enlaces
        else:
            print(f"No se pudo obtener el contenido de la URL {url}. Estado de la respuesta: {response.status_code}")
    except Exception as e:
        print(f"Error al procesar la URL {url}: {e}")
    return set()

# URL base
url_base = "https://www.elsalvador.com/category/deportes/"

# URL específica que estás buscando
url_especifica_deportes = 'https://www.elsalvador.com/deportes/'

# Lista para almacenar todos los enlaces del apartado "Deportes"
lista_deportes = []

# Conjunto para almacenar enlaces únicos
enlaces_unicos = set()

# Número máximo de páginas a extraer (ajusta según sea necesario)
max_paginas = 13

# Recorre cada sección de deportes
for seccion_url in ["opinion-deportes","viral-deportes","futbol","otros-deportes","selecciones"]:
    for pagina in range(1, max_paginas + 1):
        url = f"{url_base}{seccion_url}/page/{pagina}/"
        print(f"Enlaces encontrados en {url} que contienen '{url_especifica_deportes}':")
        enlaces = extraer_enlaces_unicos(url, url_especifica_deportes)
        for enlace in enlaces:
            print(enlace)
            enlaces_unicos.add(enlace)
            lista_deportes.append(enlace)
        print("\n")

# Union de todas las listas de las secciones del dirio digital
Lista_all_noticias = lista_noticias + lista_entretenimiento + lista_vida + lista_deportes

def obtener_categoria(url):
    try:
        partes_url = url.split('/')
        if len(partes_url) > 3:
            categorias = partes_url[3:]
            palabras_sin_guion = [palabra for palabra in categorias if '-' not in palabra and not palabra.isdigit()][:2]
            if palabras_sin_guion:
                categoria = ' '.join(palabras_sin_guion)
                return categoria
    except Exception as e:
        print(f"Error al obtener la categoría de la URL {url}: {e}")
    return 'Categoría no encontrada'

def scrape_news_info(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            title_element = soup.find('article', class_='detail').find('h1') if soup.find('article', class_='detail') else None
            title = title_element.text.strip() if title_element else 'Título no encontrado'

            content_element = soup.find('div', class_='entry-content') if soup.find('div', class_='entry-content') else None
            content = content_element.text.strip() if content_element else 'Contenido no encontrado'

            date_element = soup.find('span', class_='ago') if soup.find('span', class_='ago') else None
            date = date_element.text.strip() if date_element else 'Fecha no encontrada'

            author_element = soup.find('span', class_='author') if soup.find('span', class_='author') else None
            author = author_element.text.strip() if author_element else 'Autor no encontrado'

            etiqueta_element = soup.find('a', class_='tag') if soup.find('a', class_='tag') else None
            etiqueta = etiqueta_element.text.strip() if etiqueta_element else 'Etiqueta no encontrada'

            categoria = obtener_categoria(url)

            return {
                'Título': title,
                'Categoría': categoria,
                'Autor': author,
                'Fecha': date,
                'Etiqueta': etiqueta,
                'Contenido': content,
                'URL': url
            }
    except Exception as e:
        print(f"Error al obtener la noticia {url}: {e}")
    return None

def scrape_multiple_news(urls):
    info_noticias = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        resultados = executor.map(scrape_news_info, urls)
        for resultado in resultados:
            if resultado:
                info_noticias.append(resultado)
    return info_noticias

#Crear Dataframe
info_noticias = scrape_multiple_news(Lista_all_noticias)

df_noticias = pd.DataFrame(info_noticias)

print(df_noticias)

# Crear un archivo unico para recopilar noticias por dia

info_noticias = scrape_multiple_news(Lista_all_noticias)
df_noticias = pd.DataFrame(info_noticias)

# fecha
fecha_actual = datetime.now().strftime('%d-%m-%Y')
datos_noticias_csv = f'Noticias_{fecha_actual}.csv'

#guardar en archivo CSV
df_noticias.to_csv(datos_noticias_csv, index=False)

print(f"DataFrame guardado como: {datos_noticias_csv}")


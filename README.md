from pymongo import MongoClient
# Conectando ao banco de dados MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['mental_health_monitoring']
collection = db['student_data']
# Exemplo de dados de alunos coletados (podem vir de diversas fontes)
student_data = {
"student_id": 1,
"name": "John Doe",
"age": 20,
"survey_responses": [4, 3, 5, 2], # Respostas da pesquisa de bem-estar
"platform_interactions": 120, # Número de interações na plataforma de ensino
"social_media_posts": 5 # Número de posts em redes sociais
}
# Inserindo dados no MongoDB
collection.insert_one(student_data)
2. Processamento e Análise de Dados (Hadoop + Spark):
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean
# Iniciando uma sessão Spark
spark = SparkSession.builder.appName("MentalHealthMonitoring").getOrCreate()
# Carregando dados do Hadoop HDFS
student_data = spark.read.csv("hdfs://path_to_data/student_data.csv", header=True, inferSchema=True)
# Calculando a média de interações dos alunos na plataforma
average_interactions = student_data.agg(mean("platform_interactions")).collect()[0][0]
print(f"Média de interações na plataforma: {average_interactions}")
3. Análise de Dados com Pandas:
import pandas as pd
# Carregando os dados processados pelo Spark para o Pandasdf = pd.DataFrame(student_data.collect(), columns=student_data.columns)
# Exemplo de análise - Identificar alunos com interações abaixo da média
students_below_average = df[df['platform_interactions'] < average_interactions]
print(students_below_average)
4. Aprendizado de máquina com Scikit-learn:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Dividindo os dados em features e target (exemplo: estado emocional)
X = df[['platform_interactions', 'survey_responses', 'social_media_posts']]
y = df['emotional_state'] # Supomos que temos esse dado rotulado
# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Treinando um modelo de Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Prevendo o estado emocional de novos alunos
predictions = clf.predict(X_test)
print(predictions)
5. Visualização de Dados com Matplotlib:
import matplotlib.pyplot as plt
# Gráfico de barras - interações na plataforma
plt.figure(figsize=(10,6))
plt.bar(df['student_id'], df['platform_interactions'])
plt.title('Interações na Plataforma por Aluno')
plt.xlabel('ID do Aluno')
plt.ylabel('Número de Interações')
plt.show()
# Gráfico de pizza - distribuição do estado emocional
emotional_distribution = df['emotional_state'].value_counts()
plt.figure(figsize=(7,7))
plt.pie(emotional_distribution, labels=emotional_distribution.index, autopct='%1.1f%%')
plt.title('Distribuição do Estado Emocional dos Alunos')
plt.show()

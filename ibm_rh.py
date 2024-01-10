#!/usr/bin/env python
# coding: utf-8

# # 1. INTRODUÇÃO

# A análise a seguir foi baseada nos dados da IBM, disponibilizados no site Kaggle. 
# 
# O estudo foi objetivado para entender padrões de saída e entrada de colaboradores, a fim de melhorar a qualidade do trabalho
# e otimizar os resultados da empresa.
# 
# Perguntas a serem respondidas:
# 
# 1) Há diferença na saída de homens e mulheres?
# 2) Há diferença salarial entre homens e mulheres?
# 3) O relação entre salário e saída da empresa?
# 4) Há relação entre tempo de empresa e cargo? e tempo de empresa e salário?

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats


# # 2. CARREGANDO DATASET

# In[2]:


#carregando dataset

caminho = 'rh_ibm.csv'

df = pd.read_csv(caminho)

df


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


#resumo das colunas

df.describe()


# In[6]:


get_ipython().system('pip install matplotlib')


# In[7]:


#contagem linhas e colunas

df.shape


# In[8]:


# Supondo que seu DataFrame seja chamado df e a coluna seja 'coluna_desejada'
sns.histplot(df['Age'], bins=10, kde=False, color='Red')
plt.title('Histograma Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()


# In[9]:


# Supondo que seu DataFrame seja chamado df e a coluna seja 'coluna_desejada'
sns.histplot(df['MonthlyIncome'], bins=10, kde=False, color='Red')
plt.title('Histograma Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()


# In[10]:


#propoção entre funcionários homens e mulheres

df.Gender.value_counts(1)*100


# In[11]:


#contar a quantidade de funcionários de acordo com seu campo de estudo.

df.EducationField.value_counts(1)*100


# In[12]:


sns.histplot(df['EducationField'], bins=5, kde=False, color='Red')
plt.title('Histograma EducationField')
plt.xlabel('Field')
plt.ylabel('Frequência')
plt.xticks(rotation=90)
plt.show()


# In[13]:


sns.boxplot(x='Gender', y='MonthlyIncome', data=df)
plt.title('Boxplot de Salário x Gênero')
plt.ylabel('Salário')
plt.xlabel('Gênero')


# Embora os boxplots sejam parecidos, realizaremos um teste de hipótese estatístico para verificar se h

# In[52]:


# Divida o DataFrame com base no gênero
salario_homem = df[df['Gender'] == 'Male']['MonthlyIncome']
salario_mulher = df[df['Gender'] == 'Female']['MonthlyIncome']

# Realize o teste t
t_statistic, p_value = scipy.stats.ttest_ind(salario_homem, salario_mulher)

print(f'Test Statistic: {t_statistic}')
print(f'Valor p: {p_value}')


# Podemos observar pelo p-valor 0.13 que a 0.05 de confiança não podemos afirmar que há diferença estatística entre os salários 
# de homens e mulheres.

# In[15]:


#exibir uma matriz que mostra o grau de correlação entre as variáveis numéricas

numeric_columns = df.select_dtypes(include='number')

# Calcula a matriz de correlação
correlation_matrix = numeric_columns.corr()

# Cria o heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f', linewidths=.5)
plt.title('Heatmap de Correlação entre Colunas Numéricas')
plt.show()


# In[16]:


#media salarial por cargo e a quantidade de pessoas por nível de cargo.

media_salario_nivel = df.groupby('JobLevel')['MonthlyIncome'].sum()/df['JobLevel'].value_counts()
nivel_cargo = df['JobLevel'].value_counts()

print(f'A média salárial por cargo é: {media_salario_nivel}')
print(f'A quantidade de pessoas por nível de cargo é: {nivel_cargo}')


# De forma que quanto mais escuro, mais forte a relação, é possível perceber que há variáveis correlacionadas.

# In[53]:


#contagem do estado civil dos funcionários

df['MaritalStatus'].value_counts()


# In[54]:


#Procurar saber se há uma diferença na quantidade de treinamentos realizados por mulheres e homens no último ano

porcentagem = df.groupby('Gender')['TrainingTimesLastYear'].value_counts(normalize=True) * 100

porcentagem


# In[56]:


#Analisar se há diferença entre a frequencia de treinamentos de homens e mulheres


treinamento_homem = df[df['Gender'] == 'Male']['TrainingTimesLastYear']
treinamento_mulher = df[df['Gender'] == 'Female']['TrainingTimesLastYear']

# Realize o teste t
t_statistic, p_value = scipy.stats.ttest_ind(treinamento_homem, treinamento_mulher)

print(f'Test Statistic: {t_statistic}')
print(f'Valor p: {p_value}')


# Podemos observar pelo p-valor 0.13 que a 0.05 de confiança não podemos afirmar que há diferença estatística entre a frequência
# de treinamentos entre homens e mulheres.

# Agora vamos analisar um pouco a coluna "Atrittion" e ver se há algumas relação com outras variáveis.

# In[65]:


#relação entre a função e se atittion.

porcentagem = df.groupby('JobRole')['Attrition'].value_counts(normalize=True) * 100

porcentagem


# media_salarial_cargo = df.groupby('JobRole')['MonthlyIncome'].mean()
# 
# media_salarial_cargo.sort_index()
# 

# Podemos observar que o cargo que tem a menor média salarial (Sales Representative) também é a que tem a maior
# porcentagem de saída de pessoas da empresa.

# In[72]:


#relação entre a função e se atittion.

porcentagem = df.groupby('JobRole')['JobSatisfaction'].value_counts(normalize=True) * 100

porcentagem


# Vamos procurar fazer uma análisa baseada na faixa etária dos funcionários. para isso, vamos criar uma coluna com a ajuda
# da coluna "Age"

# In[77]:


# Função para atribuir classes com base nos gastos médios
def atribuir_classe(valor):
    if valor >= 18 and valor < 29:
        return '18-28'
    if valor >=29 and valor < 40:
        return '29-39'
    if valor >= 40 and valor < 50:
        return '40-49'
    if valor >= 50 and valor < 60:
        return '50-59'
    elif valor >= 60:
        return '60+'

# Aplicar a função à coluna existente para criar a nova coluna 'classe'
df['faixa_etaria'] = df['Age'].apply(atribuir_classe)

df.head()


# In[90]:


porcentagem = df.groupby('faixa_etaria')['Attrition'].value_counts(normalize=True) * 100

porcentagem


# In[ ]:


porcentagem = df.groupby('TrainingTimesLastYear')['Attrition'].value_counts(normalize=True) * 100

porcentagem


# In[94]:


#analise para verificar se há uma relação entre a quantidade de hora extra e saída da empresa

porcentagem = df.groupby('OverTime')['Attrition'].value_counts(normalize=True) * 100

porcentagem


# Percebemos que funcionários que não fizeram treinamentos no último ano possuem maiores chances de sair da empresa.

# In[97]:


#teste estatistico para confirmar se há uma relação entre as duas colunas categóricas

contingency_table = pd.crosstab(df['OverTime'], df['Attrition'])

# Realize o teste qui-quadrado de independência
chi2_stat, p_value, dof, expected = scipy.stats.chi2_contingency(contingency_table)

print(f'Estatística Qui-Quadrado: {chi2_stat}')
print(f'Valor p: {p_value}')


# Pelo valor do p-valor ser muito baixo, podemos afirmar a 0.05 de confiança que os valores de Overtime e Attrition têm 
# têm uma relação.

# # CONCLUSÕES

# In[ ]:


- é possivel perceber que há uma valorização do funcionário na empresa, pois as variáveis salário, cargo e 
aumento salarial crescem de acordo com o tempo de empresa e desempenho do colaborador.
- Há o dobro de homens ocupando o cargo de nível mais alto da empresa se comparado a quantidade de mulheres
- A satisfação com a empresa está ligada com o tamanho do salário
- Podemos observar que o cargo que tem a menor média salarial (Sales Representative) também é a que tem a maior 
porcentagem de saída de pessoas da empresa.
- Percebemos que funcionários que não fizeram treinamentos no último ano possuem maiores chances de sair da empresa.


# Melhorias e Sugestões
- Criar novos "níveis" para classificação de desempenho, pois só há notas 3 ou 4;
- Criar um programa que vise estimular os funcionários a realizarem os os treinamentos oferecidos, já que colaboradores que 
realizam cursos de capacitação estão menos propensos a abandonarem a empresa;
- Buscar entender o motivo dos jovens talentos terem uma porcentagem maior de saída da empresa. Isso pode ser feito através
de uma pesquisa de satisfação ou relatório de acompanhamento.
- Diminuir as horas extras da empresa, pois isso está relacionado à saída do funcionário;
- Buscar entender o motivo dos funcionários que ocupam o cargo de "Sales Representative" terem uma taxa mais alta de saída
da empresa.


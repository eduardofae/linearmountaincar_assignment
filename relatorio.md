Eduardo Dalmás Faé - 00334087

Todas as funções requeridas foram implementadas com êxito.

# Parâmetros padrões para o Experimento
Os parâmetros padrões para o experimento foram alterados para:

## AGENT
learning_rate: float = 0.01,
epsilon: float = 0.1,
epsilon_decay: float = 0.9999,
epsilon_min: float = 0.02,
discount_factor: float = 0.99

## FEATURE EXTRACTOR
n_centers: int = 25, 
sigma: float = 0.12

Tais valores foram selecionados por demonstrarem uma maior velocidade de convergência.

# Configurações selecionadas.
Foram selecionadas 10 alterações, seguindo as sugestões fornecidas pela professor no README.md.
Desses 10 agentes, foram selecionados os 5 agentes que desempenharam da melhor maneira.
Para a execução, foi setada uma seed específica para cada episódio, a fim de que nenhum agente tivesse vantagem por mérito de uma aleatoriedade.

# Tendências observadas.
Quando mantemos o discount_factor em 1.0, a convergência é muito demorada, uma pequena alteração para 0.99 já permite uma maior velocidade de convergência.
Um maior número de centros possui grande influência na velocidade de convergência.
Learning rates muito baixos, como 0.004 (testado) apresentam uma demora imensa de convergência, sendo inviáveis para 50.000 steps.
O aumento do epsilon mínimo diminuiu o tempo convergência.

# Implementação Run
Para a implementação do run foram utilizados 2 dicionários, um com os resultados de cada agente e um com cada agente que será executado.
Na implementação, passamos por cada agente e salvamos seu resultado no dicionário de resultados.
Por fim, selecionamos os 5 melhores agentes através das menores medianas.
O dicionário com os 5 melhores agente então é salvo em um json.

# Implementação Plot
O json com os resultados é lido e é realizado um box plot dos resultados fazendo uso da biblioteca matplotlib.
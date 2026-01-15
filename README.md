# Anatomia do Chat Completions

> Material didatico para AI Engineers - Dominando a API de Chat Completions da OpenAI

## Objetivo

Este repositorio contem exemplos praticos e explicacoes detalhadas sobre como utilizar a API de Chat Completions de forma profissional. Ao final, voce sera capaz de:

- Configurar parametros de geracao para diferentes casos de uso
- Construir classificadores deterministicos
- Extrair dados estruturados (JSON Schema)
- Implementar agentes com Tool Calling
- Otimizar custos e latencia em producao

---

## Indice

1. [Setup](#setup)
2. [Anatomia de uma Request](#anatomia-de-uma-request)
3. [Parametros de Geracao](#parametros-de-geracao)
   - [Temperature](#temperature)
   - [Top P](#top-p)
   - [Presence Penalty](#presence-penalty)
   - [Frequency Penalty](#frequency-penalty)
   - [Max Tokens](#max-tokens)
   - [Stop Sequences](#stop-sequences)
4. [Padroes de Uso](#padroes-de-uso)
   - [Classificador Deterministico](#1-classificador-deterministico)
   - [Saida Estruturada](#2-saida-estruturada-json-schema)
   - [Tool Calling](#3-tool-calling-agentes)
5. [Configs de Producao](#configs-de-producao)
6. [Otimizacao de Custos](#otimizacao-de-custos)
7. [Referencias](#referencias)

---

## Setup

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install openai python-dotenv

# Configurar API Key
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Anatomia de uma Request

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",                    # Modelo a ser usado
    messages=[                               # Historico de conversa
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    temperature=0.7,                         # Criatividade
    max_tokens=150,                          # Limite de resposta
    top_p=1.0,                               # Nucleus sampling
    presence_penalty=0.0,                    # Penaliza repeticao de ideias
    frequency_penalty=0.0,                   # Penaliza repeticao de palavras
    stop=["\n"]                              # Sequencias de parada
)

print(response.choices[0].message.content)
```

### Estrutura da Response

```python
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Resposta do modelo..."
        },
        "finish_reason": "stop"  # ou "length", "tool_calls"
    }],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 50,
        "total_tokens": 65
    }
}
```

---

## Parametros de Geracao

### Temperature

**O que e:** Controle de entropia (aleatoriedade/criatividade) na geracao de tokens.

**Como funciona:** Valores mais altos aumentam a probabilidade de tokens menos provaveis serem escolhidos.

| Valor | Comportamento | Caso de Uso |
|-------|---------------|-------------|
| `0.0` | Deterministico (sempre igual) | Classificadores, codigo, dados fiscais |
| `0.2` | Muito consistente | Engenharia, juridico, financeiro |
| `0.7` | Balanceado | Chatbots, conversas gerais |
| `0.9-1.0` | Criativo | Brainstorming, copywriting |
| `1.2+` | Exploratorio | Geracao de ideias, arte |

```python
# Codigo - sempre consistente
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Escreva uma funcao de fatorial"}],
    temperature=0.0
)

# Brainstorming - mais criativo
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Ideias para app de saude mental"}],
    temperature=0.9
)
```

---

### Top P

**O que e:** Nucleus sampling - controle estatistico alternativo ao temperature.

**Como funciona:** Considera apenas os tokens que representam X% da massa de probabilidade.

```
top_p=0.1  -> Usa apenas os tokens mais provaveis (10% da probabilidade)
top_p=0.9  -> Usa tokens que somam 90% da probabilidade
top_p=1.0  -> Considera todas as opcoes (padrao)
```

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Complete: O futuro da IA sera..."}],
    temperature=1.0,
    top_p=0.9  # Limita o espaco de tokens mesmo com temp alta
)
```

> **Regra de Ouro:** Use `temperature` OU `top_p` bem ajustado. Em producao, raramente ambos altos.

| Combinacao | Resultado |
|------------|-----------|
| `temp=0.2, top_p=1.0` | Consistente (recomendado para precisao) |
| `temp=1.0, top_p=0.9` | Criativo controlado (recomendado para variacao) |
| `temp=1.5, top_p=1.0` | Caotico (evitar em producao) |

---

### Presence Penalty

**O que e:** Penaliza a repeticao de **ideias/topicos** ja mencionados.

**Como funciona:** Aplica penalidade fixa a tokens que ja apareceram, independente de quantas vezes.

| Valor | Efeito |
|-------|--------|
| `0.0` | Sem penalidade - pode repetir temas |
| `0.6` | Moderado - bom para brainstorming |
| `2.0` | Forte - forca topicos completamente novos |

```python
# Bom para listas diversificadas
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Liste 10 beneficios do exercicio"}],
    temperature=0.7,
    presence_penalty=0.6  # Incentiva variedade de ideias
)
```

**Casos de uso:**
- Brainstorming
- Agentes autonomos (evita loops de raciocinio)
- Geracao de conteudo diversificado

---

### Frequency Penalty

**O que e:** Penaliza a repeticao **literal de palavras**.

**Como funciona:** Quanto mais vezes uma palavra aparece, maior a penalidade para usa-la novamente.

| Valor | Efeito |
|-------|--------|
| `0.0` | Sem penalidade |
| `0.5` | Moderado |
| `2.0` | Forte - evita repetir mesmas palavras |

```python
# Evita texto repetitivo
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Continue: O sol brilhava. O sol..."}],
    temperature=0.5,
    frequency_penalty=0.8  # Forca uso de sinonimos
)
```

**Casos de uso:**
- Evitar loops de texto
- Reducao de redundancia
- Melhoria de legibilidade

### Presence vs Frequency - Resumo

| Parametro | Penaliza | Quando Usar |
|-----------|----------|-------------|
| `presence_penalty` | Ideias/topicos repetidos | Brainstorming, agentes |
| `frequency_penalty` | Palavras repetidas | Copywriting, textos longos |

---

### Max Tokens

**O que e:** Limite fisico maximo de tokens na resposta.

**Impacto direto em:**
- Custo (mais tokens = mais caro)
- Latencia (mais tokens = mais lento)
- Prolixidade (controla tamanho)

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explique machine learning"}],
    max_tokens=100  # Limita a ~75 palavras
)

# Verificar se foi cortado
if response.choices[0].finish_reason == "length":
    print("Resposta foi truncada!")
```

**Valores de `finish_reason`:**
| Valor | Significado |
|-------|-------------|
| `stop` | Modelo terminou naturalmente |
| `length` | Atingiu `max_tokens` (truncado) |
| `tool_calls` | Modelo quer chamar uma ferramenta |

---

### Stop Sequences

**O que e:** Sequencias de caracteres que interrompem a geracao.

**Casos de uso:**
- Controlar formato de saida
- Parar em delimitadores
- Limitar respostas

```python
# Parar na primeira frase
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explique recursao"}],
    stop=["."]
)

# Classificador - parar no newline
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Classifique: POSITIVO, NEGATIVO, NEUTRO"},
        {"role": "user", "content": "Adorei o produto!"}
    ],
    temperature=0,
    stop=["\n"]
)

# Multiplas stop sequences
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Liste 5 linguagens"}],
    stop=["\n4.", "4."]  # Para antes do 4o item
)
```

---

## Padroes de Uso

### 1. Classificador Deterministico

```python
def classificar_ticket(texto: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Classifique o ticket: BUG, FEATURE, FINANCEIRO ou OUTRO. Responda apenas a categoria."
            },
            {"role": "user", "content": texto}
        ],
        temperature=0,      # Deterministico
        max_tokens=5,       # Apenas a categoria
        stop=["\n"]         # Para no primeiro newline
    )
    return response.choices[0].message.content

# Uso
classificar_ticket("O sistema esta cobrando imposto errado")  # -> "FINANCEIRO"
classificar_ticket("Gostaria de exportar em PDF")             # -> "FEATURE"
```

---

### 2. Saida Estruturada (JSON Schema)

```python
import json

def extrair_dados_pessoa(texto: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extraia dados do texto."},
            {"role": "user", "content": texto}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "pessoa",
                "schema": {
                    "type": "object",
                    "properties": {
                        "nome": {"type": "string"},
                        "idade": {"type": "number"},
                        "cidade": {"type": "string"}
                    },
                    "required": ["nome", "idade", "cidade"]
                }
            }
        },
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

# Uso
dados = extrair_dados_pessoa("Joao tem 32 anos e mora em Recife")
# -> {"nome": "Joao", "idade": 32, "cidade": "Recife"}
```

> **Nota:** Com `client.chat.completions.create()`, o JSON vem como string em `.content`. Use `json.loads()` para converter.

---

### 3. Tool Calling (Agentes)

```python
import json

# 1. Definir a ferramenta
def get_weather(city: str) -> dict:
    # Aqui seria a chamada real a uma API de clima
    return {"city": city, "weather": "28C, ensolarado"}

# 2. Configurar tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Obtem o clima atual de uma cidade",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

# 3. Primeira chamada
messages = [
    {"role": "system", "content": "Voce informa o clima usando ferramentas."},
    {"role": "user", "content": "Como esta o clima em Recife?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

msg = response.choices[0].message

# 4. Se o modelo quer usar ferramenta
if msg.tool_calls:
    for tool_call in msg.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_weather(args["city"])

        # Adicionar resultado ao historico
        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # 5. Segunda chamada com resultado
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print(final.choices[0].message.content)
```

**Fluxo do Tool Calling:**

```
Usuario: "Clima em Recife?"
    |
    v
[1a Chamada] -> Modelo decide: preciso da tool "get_weather"
    |
    v
[Execucao] -> get_weather("Recife") -> {"weather": "28C"}
    |
    v
[2a Chamada] -> Modelo com resultado -> "Em Recife esta 28C, ensolarado!"
```

---

## Configs de Producao

### Chatbot de Atendimento
```python
config = {
    "temperature": 0.3,      # Consistente mas natural
    "max_tokens": 150,       # Respostas concisas
    "presence_penalty": 0.3  # Evita repeticao
}
```

### Gerador de Codigo
```python
config = {
    "temperature": 0,        # Deterministico
    "max_tokens": 300        # Espaco para codigo completo
}
```

### Copywriter Criativo
```python
config = {
    "temperature": 0.9,       # Alta criatividade
    "presence_penalty": 0.6,  # Ideias variadas
    "frequency_penalty": 0.5  # Vocabulario diverso
}
```

### Classificador
```python
config = {
    "temperature": 0,   # Deterministico
    "max_tokens": 5,    # Apenas a categoria
    "stop": ["\n"]      # Para no newline
}
```

### Extracao de Dados
```python
config = {
    "temperature": 0,                    # Precisao
    "response_format": {"type": "json_schema", ...}
}
```

---

## Otimizacao de Custos

### Precos GPT-4o-mini (referencia)
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

### Estrategias

1. **Limitar `max_tokens`**
```python
# Economico
max_tokens=100  # ~$0.00006 por resposta

# Caro
max_tokens=500  # ~$0.00030 por resposta
```

2. **Usar `stop` sequences**
```python
# Para antes de gerar conteudo desnecessario
stop=["\n\n", "---"]
```

3. **System prompts curtos**
```python
# Ruim - verboso
{"role": "system", "content": "Voce e um assistente muito util..."}

# Bom - direto
{"role": "system", "content": "Assistente tecnico. Seja direto."}
```

4. **Escolher modelo adequado**
```
gpt-4o-mini  -> Tarefas simples, alto volume
gpt-4o       -> Tarefas complexas, raciocinio
gpt-4-turbo  -> Contexto longo (128k)
```

---

## Executando os Exemplos

```bash
# Rodar todos os demos
python main.py

# Rodar demo especifico (editar main.py)
if __name__ == "__main__":
    demo_temperature()        # Descomentar o desejado
    # demo_top_p()
    # demo_stop_sequences()
```

---

## Referencias

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

---

## Checklist do AI Engineer

- [ ] Entendo a diferenca entre `temperature` e `top_p`
- [ ] Sei quando usar `presence_penalty` vs `frequency_penalty`
- [ ] Consigo criar classificadores deterministicos
- [ ] Sei extrair dados estruturados com JSON Schema
- [ ] Implemento Tool Calling para agentes
- [ ] Otimizo custos com `max_tokens` e `stop`
- [ ] Escolho configs adequadas para cada caso de uso

---

**Autor:** AI Engineering Academy
**Versao:** 1.0
**Data:** 2025

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# 1. REQUEST BASE
# =========================

def base_request():
    print("\n=== REQUEST BASE ===")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um engenheiro de software sênior."},
            {"role": "user", "content": "Explique o que é Docker em uma frase."}
        ],
    )

    print(resp.choices[0].message.content)


# =========================
# 2. CLASSIFICADOR DETERMINÍSTICO
# =========================

def classifier():
    print("\n=== CLASSIFICADOR ===")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Você é um classificador determinístico. Responda apenas com: BUGGADO, FEATURE, FINANCEIRO ou OUTRO."
            },
            {
                "role": "user",
                "content": "O sistema está cobrando imposto errado no boleto."
            }
        ],
        temperature=1,
        max_tokens=5000,
        stop=["\n"]
    )

    print("Classificação:", resp.choices[0].message.content)


# =========================
# 3. SAÍDA ESTRUTURADA
# =========================

def structured_output():
    print("\n=== SAÍDA ESTRUTURADA ===")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extraia dados do texto."},
            {"role": "user", "content": "João tem 32 anos e mora em Recife e gosta de comer cachorro quente com maionse bebendo coca-cola."}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "nome": {"type": "string"},
                        "idade": {"type": "number"},
                        "cidade": {"type": "string"},
                        "comida": {"type": "string"},
                            "bebida": {"type": "string"}
                    },
                    "required": ["nome", "idade", "cidade", "comidaaaaa"]
                }
            }
        },
        temperature=1
    )

    parsed_data = json.loads(resp.choices[0].message.content)
    print("Objeto estruturado:", parsed_data)





# =========================
# 5. PARÂMETROS DE GERAÇÃO - EXEMPLOS PRÁTICOS
# =========================

def demo_temperature():
    """
    TEMPERATURE: Controle de entropia (criatividade)
    - 0.0 → determinístico (sempre igual)
    - 0.2 → engenharia/código/fiscal
    - 0.7 → conversa/brainstorming
    - 1.0+ → exploração criativa
    """
    print("\n" + "="*60)
    print("  DEMO: TEMPERATURE")
    print("="*60)

    prompt = "Sugira um nome para uma startup de IA"

    temperatures = [0.0, 0.2, 0.7, 1.2]

    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        # Faz 3 chamadas para mostrar variação
        for i in range(3):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=20
            )
            print(f"  [{i+1}] {resp.choices[0].message.content.strip()}")


def demo_temperature_use_cases():
    """
    TEMPERATURE: Casos de uso práticos
    """
    print("\n" + "="*60)
    print("  TEMPERATURE: CASOS DE USO")
    print("="*60)

    # Caso 1: Código (baixa temperature)
    print("\n--- CODIGO (temp=0.0) - Deterministico ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Voce e um programador Python expert."},
            {"role": "user", "content": "Escreva uma funcao para calcular fatorial."}
        ],
        temperature=0.0,
        max_tokens=150
    )
    print(resp.choices[0].message.content)

    # Caso 2: Fiscal/Juridico (baixa temperature)
    print("\n--- FISCAL (temp=0.2) - Consistente ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Voce e um contador fiscal brasileiro."},
            {"role": "user", "content": "Qual a aliquota do ICMS interestadual para produtos industrializados entre SP e RJ?"}
        ],
        temperature=0.2,
        max_tokens=100
    )
    print(resp.choices[0].message.content)

    # Caso 3: Brainstorming (alta temperature)
    print("\n--- BRAINSTORMING (temp=0.9) - Criativo ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "De 3 ideias inovadoras para um app de saude mental."}
        ],
        temperature=0.9,
        max_tokens=200
    )
    print(resp.choices[0].message.content)


def demo_top_p():
    """
    TOP_P: Nucleus sampling (controle estatistico)
    - Usa apenas tokens que representem X% da probabilidade
    - top_p=0.1 → muito restritivo (palavras mais provaveis)
    - top_p=0.9 → mais variado
    - top_p=1.0 → considera todas as opcoes

    Regra: Use temperature OU top_p, raramente ambos altos
    """
    print("\n" + "="*60)
    print("  DEMO: TOP_P (Nucleus Sampling)")
    print("="*60)

    prompt = "Complete a frase: O futuro da inteligencia artificial sera"

    # Comparacao top_p baixo vs alto (com temperature=1 para ver efeito)
    configs = [
        {"top_p": 0.1, "desc": "Muito restritivo - so tokens mais provaveis"},
        {"top_p": 0.5, "desc": "Moderado"},
        {"top_p": 0.95, "desc": "Amplo - mais variacao"},
    ]

    for config in configs:
        print(f"\n--- top_p={config['top_p']} ({config['desc']}) ---")
        for i in range(2):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,  # Alto para ver efeito do top_p
                top_p=config["top_p"],
                max_tokens=300
            )
            print(f"  [{i+1}] {resp.choices[0].message.content.strip()}")


def demo_temperature_vs_top_p():
    """
    Demonstra por que NAO usar ambos altos simultaneamente
    """
    print("\n" + "="*60)
    print("  TEMPERATURE vs TOP_P - Combinacoes")
    print("="*60)

    prompt = "Invente uma palavra nova e defina seu significado."

    configs = [
        {"temperature": 0.2, "top_p": 1.0, "desc": "RECOMENDADO: temp baixa, top_p padrao"},
        {"temperature": 1.0, "top_p": 0.9, "desc": "RECOMENDADO: temp alta, top_p limitado"},
        {"temperature": 1.5, "top_p": 1.0, "desc": "CUIDADO: ambos altos = caotico"},
    ]

    for config in configs:
        print(f"\n--- {config['desc']} ---")
        print(f"    temperature={config['temperature']}, top_p={config['top_p']}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=50
        )
        print(f"    Resultado: {resp.choices[0].message.content.strip()}")


def demo_presence_penalty():
    """
    PRESENCE_PENALTY: Penaliza repeticao de IDEIAS/TOPICOS
    - 0.0 → sem penalidade (pode repetir temas)
    - 0.6 → moderado (bom para brainstorming)
    - 2.0 → forte (forca novos topicos)

    Bom para: criatividade, agentes, brainstorming
    """
    print("\n" + "="*60)
    print("  DEMO: PRESENCE_PENALTY (Penaliza repeticao de ideias)")
    print("="*60)

    prompt = "Liste 10 beneficios de fazer exercicio fisico."

    penalties = [0.0, 1.0, 2.0]

    for penalty in penalties:
        print(f"\n--- presence_penalty={penalty} ---")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            presence_penalty=penalty,
            max_tokens=100
        )
        print(resp.choices[0].message.content)


def demo_frequency_penalty():
    """
    FREQUENCY_PENALTY: Penaliza repeticao LITERAL de palavras
    - 0.0 → sem penalidade
    - 0.5 → moderado
    - 2.0 → forte (evita repetir mesmas palavras)

    Bom para: evitar loops, eco de texto, redundancia
    """
    print("\n" + "="*60)
    print("  DEMO: FREQUENCY_PENALTY (Penaliza repeticao de palavras)")
    print("="*60)

    # Prompt que tende a gerar repeticao
    prompt = """Continue este texto mantendo o mesmo estilo:
    "O sol brilhava intensamente. O sol iluminava as montanhas. O sol..."
    """

    penalties = [0.0, 0.8, 2.0]

    for penalty in penalties:
        print(f"\n--- frequency_penalty={penalty} ---")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            frequency_penalty=penalty,
            max_tokens=100
        )
        print(resp.choices[0].message.content)


def demo_presence_vs_frequency():
    """
    Comparacao direta: presence vs frequency penalty
    """
    print("\n" + "="*60)
    print("  PRESENCE vs FREQUENCY PENALTY")
    print("="*60)

    prompt = "Escreva um paragrafo sobre inteligencia artificial repetindo conceitos importantes."

    configs = [
        {"presence": 0.0, "frequency": 0.0, "desc": "Sem penalidades"},
        {"presence": 1.5, "frequency": 0.0, "desc": "So presence (novos topicos)"},
        {"presence": 0.0, "frequency": 1.5, "desc": "So frequency (novas palavras)"},
        {"presence": 0.8, "frequency": 0.8, "desc": "Ambos moderados"},
    ]

    for config in configs:
        print(f"\n--- {config['desc']} ---")
        print(f"    presence={config['presence']}, frequency={config['frequency']}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            presence_penalty=config["presence"],
            frequency_penalty=config["frequency"],
            max_tokens=150
        )
        print(resp.choices[0].message.content)


def demo_max_tokens():
    """
    MAX_TOKENS: Limite fisico da resposta

    Controla:
    - Custo (mais tokens = mais caro)
    - Latencia (mais tokens = mais lento)
    - Prolixidade (limita tamanho)
    """
    print("\n" + "="*60)
    print("  DEMO: MAX_TOKENS (Limite de resposta)")
    print("="*60)

    prompt = "Explique o que e machine learning de forma completa."

    limits = [30, 100, 300]

    for limit in limits:
        print(f"\n--- max_tokens={limit} ---")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=limit
        )
        content = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason
        print(f"Finish reason: {finish_reason}")
        print(f"Tokens usados: ~{len(content.split())} palavras")
        print(f"Resposta: {content}")


def demo_stop_sequences():
    """
    STOP: Sequencias que interrompem a geracao

    Util para:
    - Controlar formato de saida
    - Parar em delimitadores especificos
    - Evitar conteudo indesejado apos certo ponto
    """
    print("\n" + "="*60)
    print("  DEMO: STOP SEQUENCES")
    print("="*60)

    # Exemplo 1: Parar apos primeira frase
    print("\n--- Exemplo 1: Parar no ponto final ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Explique recursao em programacao."}
        ],
        temperature=0.3,
        max_tokens=200,
        stop=["."]  # Para na primeira frase
    )
    print(f"Resposta: {resp.choices[0].message.content}")

    # Exemplo 2: Parar em quebra de linha (classificadores)
    print("\n--- Exemplo 2: Classificador com stop em newline ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classifique o sentimento: POSITIVO, NEGATIVO ou NEUTRO"},
            {"role": "user", "content": "Adorei o produto, superou expectativas! Era uma bosta fuck yeah!!!!!!"}
        ],
        temperature=0,
        max_tokens=10,
        stop=["\n"]
    )
    print(f"Classificacao: {resp.choices[0].message.content}")

    # Exemplo 3: Multiplas stop sequences
    print("\n--- Exemplo 3: Multiplas stop sequences ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Liste 5 linguagens de programacao populares, uma por linha."}
        ],
        temperature=0.3,
        max_tokens=100,
        stop=["\n4.", "4."]  # Para antes do 4 item
    )
    print(f"Resposta (limitada a 3):\n{resp.choices[0].message.content}")

    # Exemplo 4: Stop em delimitador customizado
    print("\n--- Exemplo 4: Stop em delimitador customizado ---")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Responda no formato: RESPOSTA: <sua resposta> Gabriel FIM"},
            {"role": "user", "content": "Qual a capital do Brasil?"}
        ],
        temperature=0,
        max_tokens=50,
        stop=["FIM", "\n\n"]
    )
    print(f"Resposta: {resp.choices[0].message.content}")


def demo_combined_production_configs():
    """
    Configuracoes combinadas para cenarios de producao
    """
    print("\n" + "="*60)
    print("  CONFIGS DE PRODUCAO - CENARIOS REAIS")
    print("="*60)

    # Config 1: Chatbot de atendimento
    print("\n--- CHATBOT ATENDIMENTO ---")
    print("Config: temp=0.3, max_tokens=150, presence=0.3")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Voce e um atendente de suporte tecnico. Seja direto e util."},
            {"role": "user", "content": "Meu pedido nao chegou ainda, o que faco?"}
        ],
        temperature=0.3,
        max_tokens=150,
        presence_penalty=0.3
    )
    print(resp.choices[0].message.content)

    # Config 2: Gerador de codigo
    print("\n--- GERADOR DE CODIGO ---")
    print("Config: temp=0, max_tokens=300")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Voce e um programador Python. Retorne apenas codigo, sem explicacoes."},
            {"role": "user", "content": "Funcao que valida CPF brasileiro."}
        ],
        temperature=0,
        max_tokens=300
    )
    print(resp.choices[0].message.content)

    # Config 3: Copywriter criativo
    print("\n--- COPYWRITER CRIATIVO ---")
    print("Config: temp=0.9, presence=0.6, frequency=0.5")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Crie 3 headlines criativas para uma campanha de cafe gourmet."}
        ],
        temperature=0.9,
        max_tokens=150,
        presence_penalty=0.6,
        frequency_penalty=0.5
    )
    print(resp.choices[0].message.content)

    # Config 4: Classificador deterministico
    print("\n--- CLASSIFICADOR ---")
    print("Config: temp=0, max_tokens=5, stop=['\\n']")
    texts = [
        "O sistema travou e perdi meu trabalho",
        "Gostaria de poder exportar em PDF",
        "Voces sao os melhores, parabens!"
    ]
    for text in texts:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classifique: BUG, FEATURE, ELOGIO ou OUTRO. Responda so a categoria."},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=5,
            stop=["\n"]
        )
        print(f"  '{text[:40]}...' -> {resp.choices[0].message.content}")


def demo_cost_optimization():
    """
    Demonstracao de otimizacao de custos com parametros
    """
    print("\n" + "="*60)
    print("  OTIMIZACAO DE CUSTOS")
    print("="*60)

    prompt = "Resuma o conceito de API REST."

    configs = [
        {"max_tokens": 500, "desc": "Sem limite (caro)"},
        {"max_tokens": 100, "desc": "Limitado (economico)"},
        {"max_tokens": 50, "desc": "Muito limitado (mais economico)"},
    ]

    print("\nMesmo prompt, diferentes limites:\n")

    for config in configs:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=config["max_tokens"]
        )

        # Simula calculo de custo (valores aproximados gpt-4o-mini)
        output_tokens = resp.usage.completion_tokens
        input_tokens = resp.usage.prompt_tokens
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)

        print(f"{config['desc']}")
        print(f"  Tokens: {input_tokens} in + {output_tokens} out")
        print(f"  Custo estimado: ${cost:.6f}")
        print(f"  Resposta: {resp.choices[0].message.content[:80]}...")
        print()

# =========================
# TOOL: Weather API (mock real)
# ========================

def get_weather(city: str):
    """
    Exemplo simples.
    Em produção, aqui você chamaria OpenWeather, WeatherAPI, etc.
    """
    fake_weather_db = {
        "sao paulo": "22°C, parcialmente nublado",
        "recife": "28°C, ensolarado",
        "rio de janeiro": "30°C, calor intenso com previsão de chuva",
        "curitiba": "18°C, chuva leve",
        "campina grande": "21°C, sem nuvens",
    }

    return {
        "city": city,
        "weather": fake_weather_db.get(city.lower(), "Clima não encontrado")
    }


# =========================
# 4. TOOL CALLING — WEATHER AGENT
# =========================

def weather_agent(user_message: str):
    print("\n=== WEATHER AGENT ===")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Obtém o clima atual de uma cidade",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "Você é um assistente que informa o clima usando ferramentas quando necessário."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0
    )

    msg = response.choices[0].message

    # Se o modelo decidiu chamar uma ferramenta
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "get_weather":
                args = json.loads(tool_call.function.arguments)
                result = get_weather(args["city"])

                messages.append(msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "get_weather",
                    "content": json.dumps(result)
                })

                # Segunda chamada com o resultado da tool
                final_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0
                )

                print("Resposta final:", final_response.choices[0].message.content)
    else:
        print("Resposta direta:", msg.content)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("="*60)
    print("  ANATOMIA DO CHAT COMPLETIONS - PARAMETROS DE GERACAO")
    print("="*60)

    # Demos basicas originais
    base_request()
    # classifier()
    # structured_output()

    # === NOVOS DEMOS DE PARAMETROS ===

    # 1. Temperature
    # demo_temperature()
    # demo_temperature_use_cases()

    # 2. Top_p
    # demo_top_p()
    # demo_temperature_vs_top_p()

    # 3. Presence Penalty
    # demo_presence_penalty()

    # 4. Frequency Penalty
    # demo_frequency_penalty()
    # demo_presence_vs_frequency()

    # 5. Max Tokens
    # demo_max_tokens()

    # 6. Stop Sequences
    # demo_stop_sequences()

    # 7. Configs de Producao
    # demo_combined_production_configs()

    # 8. Otimizacao de Custos
    # demo_cost_optimization()

    # weather_agent("Como esta o clima em NY hoje considerando que possivelmente é igual de curitiba?")

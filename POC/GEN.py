import os
from dotenv import load_dotenv

# Carrega as variáveis do ficheiro .env
load_dotenv()


if not os.getenv("GOOGLE_API_KEY"):
    print("ERRO: Chave GOOGLE_API_KEY não encontrada. Adicione-a ao .env")
    exit(1)

#incialmente era agno porém foi alterado para phi por conta de erros
from phi.agent import Agent
from phi.model.google import Gemini
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.embedder.google import GeminiEmbedder

#  Configurar o Banco de Dados Vetorial 
vector_db = LanceDb(
    table_name="dados_csv_gemini",
    uri="./lancedb_data_gemini", 
    search_type=SearchType.vector,
    embedder=GeminiEmbedder(
        id="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    ),
)

#  Criar a Base de Conhecimento

knowledge_base = CSVKnowledgeBase(
    path="base_conhecimento_ifood_genai-exemplo.csv",
    vector_db=vector_db,
    num_documents=5, 
)

print("A carregar base de conhecimento...")
knowledge_base.load(recreate=True)


agent = Agent(
    model=Gemini(id="gemini-2.5-flash"), 
    knowledge=knowledge_base,
    search_knowledge=True, 
    show_tool_calls=True, 
    description="Agente de Suporte iFood",
    instructions=[
        "Sempre consulte o csv antes de responder.",
        "Se a resposta não estiver no CSV, diga que não encontrou informação suficiente.",
        "mude um pouco a forma de responder , não copie e cole oque esta na abse de dados"
        
    ],
    markdown=True,
)

print("\n--- Agente de suporte Iniciado ---\n")
agent.print_response("um cliente  quer reembolso, mas o pedido já saiu para a entrega. Ainda é permitido?", stream=True)
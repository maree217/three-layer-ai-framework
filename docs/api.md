# API Reference

Complete API documentation for the Three-Layer AI Framework.

## Layer 1: UX Automation

### RAGChatbot

A chatbot with Retrieval-Augmented Generation capabilities.

```python
from src.layer1.rag_chatbot import RAGChatbot

bot = RAGChatbot(
    knowledge_base: str,
    model: str = "gpt-4",
    embedding_model: str = "text-embedding-ada-002",
    temperature: float = 0.7,
    max_tokens: int = 1000
)
```

#### Parameters

- **knowledge_base** (str): Path to knowledge base directory
- **model** (str, optional): OpenAI model name. Default: "gpt-4"
- **embedding_model** (str, optional): Embedding model. Default: "text-embedding-ada-002"
- **temperature** (float, optional): Sampling temperature. Default: 0.7
- **max_tokens** (int, optional): Maximum response length. Default: 1000

#### Methods

##### `chat(query: str) -> str`

Process a user query and return a response.

```python
response = bot.chat("What is your return policy?")
```

##### `add_documents(documents: List[str]) -> None`

Add documents to the knowledge base.

```python
bot.add_documents([
    "./policies/return-policy.pdf",
    "./faq/common-questions.md"
])
```

##### `clear_history() -> None`

Clear conversation history.

```python
bot.clear_history()
```

## Layer 2: Data Intelligence

### KnowledgeGraph

Build and query enterprise knowledge graphs.

```python
from src.layer2.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph(
    database: str = "neo4j",
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = "password"
)
```

#### Methods

##### `ingest_documents(documents: List[str]) -> None`

Ingest documents into the knowledge graph.

```python
kg.ingest_documents([
    "./data/policies/*.pdf",
    "./data/processes/*.docx"
])
```

##### `query(query_text: str) -> List[Dict]`

Query the graph using natural language.

```python
results = kg.query("Find all processes related to onboarding")
# Returns: [{"entity": "...", "relationship": "...", "score": 0.95}]
```

##### `add_relationship(from_entity: str, to_entity: str, relationship_type: str) -> None`

Manually add a relationship.

```python
kg.add_relationship(
    from_entity="Employee",
    to_entity="Department",
    relationship_type="WORKS_IN"
)
```

### DataPipeline

Real-time data pipeline management.

```python
from src.layer2.data_pipeline import DataPipeline

pipeline = DataPipeline(
    source: str,
    destination: str,
    schedule: str = "realtime"
)
```

#### Methods

##### `add_transformation(name: str, function: Callable) -> None`

Add a transformation function.

```python
def clean_data(df):
    return df.dropna()

pipeline.add_transformation("clean", clean_data)
```

##### `start() -> None`

Start the pipeline.

```python
pipeline.start()
```

##### `stop() -> None`

Stop the pipeline.

```python
pipeline.stop()
```

## Layer 3: Strategic Intelligence

### StrategicForecastingEngine

Generate strategic forecasts and scenarios.

```python
from src.layer3.azure_ai_foundry import StrategicForecastingEngine

engine = StrategicForecastingEngine(
    workspace: str,
    compute_target: str = "cpu-cluster"
)
```

#### Methods

##### `train_forecast(data, horizon: int, frequency: str, metrics: List[str])`

Train a forecasting model.

```python
model = engine.train_forecast(
    data=historical_data,
    horizon=12,  # months
    frequency="M",
    metrics=["revenue", "costs"]
)
```

##### `predict(model, periods: int) -> DataFrame`

Generate predictions.

```python
forecast = engine.predict(model, periods=12)
```

##### `generate_scenarios(base_assumptions: Dict, uncertainty_ranges: Dict) -> List[Dict]`

Generate strategic scenarios.

```python
scenarios = engine.generate_scenarios(
    base_assumptions={"growth": 0.05},
    uncertainty_ranges={"growth": (0.02, 0.08)}
)
```

### DashboardGenerator

Generate executive dashboards and reports.

```python
from src.layer3.executive_dashboard import DashboardGenerator

generator = DashboardGenerator()
```

#### Methods

##### `create_board_report(data_sources, time_period, include_forecast=True)`

Create a board-ready report.

```python
report = generator.create_board_report(
    data_sources=["finance", "sales"],
    time_period="Q4_2024",
    include_forecast=True
)
```

##### `save_pdf(filename: str) -> None`

Save report as PDF.

```python
report.save_pdf("board_report.pdf")
```

## Common Data Types

### ChatMessage

```python
{
    "role": "user" | "assistant" | "system",
    "content": str,
    "timestamp": datetime
}
```

### Forecast

```python
{
    "metric": str,
    "periods": List[datetime],
    "values": List[float],
    "confidence_lower": List[float],
    "confidence_upper": List[float]
}
```

### Scenario

```python
{
    "name": str,
    "assumptions": Dict[str, float],
    "forecast": Dict[str, List[float]],
    "probability": float
}
```

## Error Handling

All methods raise descriptive exceptions:

```python
from src.layer1.exceptions import (
    ChatbotError,
    KnowledgeBaseError,
    ModelError
)

try:
    response = bot.chat("Hello")
except ChatbotError as e:
    print(f"Chatbot error: {e}")
except ModelError as e:
    print(f"Model error: {e}")
```

## Configuration

### Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

# Azure ML
AZURE_ML_WORKSPACE=...
AZURE_ML_SUBSCRIPTION=...

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
```

## Rate Limits

| Service | Limit | Burst |
|---------|-------|-------|
| Azure OpenAI | 10 req/s | 100 req/min |
| Knowledge Graph | 100 req/s | 1000 req/min |
| Forecasting | 1 req/min | 10 req/hour |

## Changelog

### v1.0.0 (2024-09-01)
- Initial release
- Layer 1, 2, 3 core functionality
- Examples and documentation

## Support

For API questions:
- Email: 2maree@gmail.com
- GitHub Issues: [Report an issue](https://github.com/maree217/three-layer-ai-framework/issues)

---

Last updated: 2024-09-02

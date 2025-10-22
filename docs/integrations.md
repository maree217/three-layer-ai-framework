# Integration Guide

## Overview

This guide covers integrating the Three-Layer AI Framework with enterprise systems and Microsoft ecosystem.

## Microsoft 365 Integration

### Microsoft Copilot

Deploy your Layer 1 chatbot as a Copilot plugin.

#### Prerequisites
- Microsoft 365 E3 or E5 license
- Copilot for Microsoft 365 license
- Azure AD app registration

#### Setup

1. **Register Azure AD App**
```bash
az ad app create \
  --display-name "Three-Layer-AI-Copilot" \
  --sign-in-audience AzureADMyOrg
```

2. **Create Plugin Manifest**
```json
{
  "schema_version": "v2",
  "name_for_human": "Enterprise AI Assistant",
  "name_for_model": "enterprise_ai",
  "description_for_human": "AI assistant powered by three-layer framework",
  "description_for_model": "Assists with enterprise queries using RAG and knowledge graph",
  "auth": {
    "type": "oauth",
    "authorization_url": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
    "token_url": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
  },
  "api": {
    "type": "openapi",
    "url": "https://your-api.azurewebsites.net/openapi.json"
  }
}
```

3. **Deploy Plugin**
```python
from src.layer1.copilot_integration import deploy_plugin

deploy_plugin(
    manifest_path="./manifests/copilot-plugin.json",
    api_endpoint="https://your-api.azurewebsites.net"
)
```

### Microsoft Teams

Deploy as a Teams bot or message extension.

#### Setup

```python
from botbuilder.core import BotFrameworkAdapter, TurnContext
from botbuilder.schema import Activity
from src.layer1.rag_chatbot import RAGChatbot

# Initialize bot
bot = RAGChatbot(knowledge_base="./data")

# Create adapter
adapter = BotFrameworkAdapter(settings)

async def messages(req: Request) -> Response:
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    async def turn_call(turn_context: TurnContext):
        response = bot.chat(turn_context.activity.text)
        await turn_context.send_activity(response)

    await adapter.process_activity(activity, auth_header, turn_call)
    return Response(status=200)
```

### SharePoint

Integrate knowledge base with SharePoint documents.

```python
from src.layer2.connectors import SharePointConnector

sp = SharePointConnector(
    site_url="https://company.sharepoint.com/sites/knowledge",
    client_id=os.getenv("SP_CLIENT_ID"),
    client_secret=os.getenv("SP_CLIENT_SECRET")
)

# Sync documents
documents = sp.get_documents(
    library="Documents",
    filter="FileLeafRef endswith '.pdf'"
)

# Add to knowledge base
bot.add_documents(documents)
```

## Power Platform Integration

### Power BI

Create executive dashboards with Layer 3 forecasts.

```python
from src.layer3.powerbi_integration import PowerBIPublisher

publisher = PowerBIPublisher(
    workspace_id=os.getenv("POWERBI_WORKSPACE_ID"),
    app_id=os.getenv("POWERBI_APP_ID"),
    app_secret=os.getenv("POWERBI_APP_SECRET")
)

# Publish forecast data
publisher.publish_dataset(
    data=forecast_df,
    dataset_name="Strategic Forecast",
    table_name="Forecasts"
)

# Refresh dashboard
publisher.trigger_refresh("Executive Dashboard")
```

### Power Automate

Trigger AI workflows from Power Automate.

#### Create HTTP Trigger

```python
from fastapi import FastAPI, Request
from src.layer1.rag_chatbot import RAGChatbot

app = FastAPI()
bot = RAGChatbot(knowledge_base="./data")

@app.post("/api/process-query")
async def process_query(request: Request):
    data = await request.json()
    query = data.get("query")
    response = bot.chat(query)
    return {"response": response}
```

#### Power Automate Flow

```json
{
  "trigger": "When a new email arrives",
  "actions": [
    {
      "type": "HTTP",
      "method": "POST",
      "uri": "https://your-api.azurewebsites.net/api/process-query",
      "body": {
        "query": "@{triggerBody()?['Subject']}"
      }
    },
    {
      "type": "Send email",
      "to": "@{triggerBody()?['From']}",
      "subject": "Re: @{triggerBody()?['Subject']}",
      "body": "@{outputs('HTTP')?['response']}"
    }
  ]
}
```

## CRM Integration

### Dynamics 365

Connect to Dynamics 365 for customer data.

```python
from src.layer2.connectors import Dynamics365Connector

d365 = Dynamics365Connector(
    instance_url="https://org.crm.dynamics.com",
    client_id=os.getenv("D365_CLIENT_ID"),
    client_secret=os.getenv("D365_CLIENT_SECRET"),
    tenant_id=os.getenv("D365_TENANT_ID")
)

# Get customer data
customers = d365.get_records(
    entity="accounts",
    select=["name", "industry", "revenue"],
    filter="revenue gt 1000000"
)

# Build knowledge graph
kg.add_entities(customers)
```

### Salesforce

Integrate with Salesforce data.

```python
from src.layer2.connectors import SalesforceConnector

sf = SalesforceConnector(
    username=os.getenv("SF_USERNAME"),
    password=os.getenv("SF_PASSWORD"),
    security_token=os.getenv("SF_TOKEN")
)

# Extract opportunities
opportunities = sf.query(
    "SELECT Id, Name, Amount, StageName FROM Opportunity WHERE Amount > 100000"
)

# Add to data pipeline
pipeline.ingest(opportunities)
```

## Data Integration

### Azure Synapse

Use Synapse as data warehouse for Layer 2.

```python
from src.layer2.connectors import SynapseConnector

synapse = SynapseConnector(
    workspace=os.getenv("SYNAPSE_WORKSPACE"),
    database=os.getenv("SYNAPSE_DATABASE"),
    authentication="Service Principal"
)

# Load data
df = synapse.query("""
    SELECT
        customer_id,
        SUM(order_amount) as total_revenue,
        COUNT(*) as order_count
    FROM sales.orders
    GROUP BY customer_id
""")

# Use in forecasting
forecast_engine.train(df)
```

### Databricks

Use Databricks for data processing.

```python
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder \
    .profile("DEFAULT") \
    .getOrCreate()

# Process data
df = spark.sql("""
    SELECT * FROM delta.`/mnt/data/orders`
    WHERE order_date >= '2024-01-01'
""")

# Build knowledge graph
kg.ingest_dataframe(df.toPandas())
```

## External Services

### OpenAI

Use OpenAI models directly (alternative to Azure OpenAI).

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Anthropic Claude

Use Claude models for specific use cases.

```python
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Authentication

### Azure AD (Entra ID)

Secure your APIs with Azure AD.

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
        return payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/protected")
async def protected_route(user: str = Depends(get_current_user)):
    return {"message": f"Hello {user}"}
```

## Monitoring

### Application Insights

Monitor your deployment with Application Insights.

```python
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string=os.getenv("APPINSIGHTS_CONNECTION_STRING")
))

# Log events
logger.info("User query processed", extra={
    "custom_dimensions": {
        "user_id": "123",
        "query_length": 50,
        "response_time_ms": 234
    }
})
```

### Custom Metrics

Track business metrics.

```python
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module

stats = stats_module.stats
view_manager = stats.view_manager

# Define metrics
query_measure = measure_module.MeasureInt(
    "queries", "number of queries", "queries"
)

query_view = view_module.View(
    "query count",
    "number of queries",
    [],
    query_measure,
    aggregation_module.CountAggregation()
)

view_manager.register_view(query_view)

# Record metrics
mmap = stats.stats_recorder.new_measurement_map()
mmap.measure_int_put(query_measure, 1)
mmap.record()
```

## Webhook Integration

### Receive Webhooks

```python
from fastapi import FastAPI, Request
from hmac import compare_digest
import hashlib

app = FastAPI()

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = hashlib.sha256(secret.encode() + payload).hexdigest()
    return compare_digest(signature, expected)

@app.post("/webhook/salesforce")
async def salesforce_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("X-Salesforce-Signature")

    if not verify_signature(payload, signature, os.getenv("WEBHOOK_SECRET")):
        return {"error": "Invalid signature"}, 401

    data = await request.json()
    # Process webhook data
    pipeline.ingest(data)

    return {"status": "received"}
```

## Troubleshooting

### Common Issues

**Issue: Authentication fails**
```python
# Check credentials
print(os.getenv("AZURE_CLIENT_ID"))  # Should not be None

# Test connection
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
token = credential.get_token("https://management.azure.com/.default")
print(token.token[:10])  # Should print token prefix
```

**Issue: Rate limiting**
```python
# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_api():
    return openai.chat.completions.create(...)
```

## Further Reading

- [Architecture Guide](./architecture.md)
- [API Reference](./api.md)
- [Best Practices](./best-practices.md)

---

**Questions?** Contact 2maree@gmail.com

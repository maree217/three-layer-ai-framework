# Quick Start Guide

Get your Three-Layer AI Framework up and running in 15 minutes!

## Prerequisites

- Python 3.9 or higher
- Azure account (free tier works)
- Git installed
- Basic command line knowledge

## Step 1: Clone the Repository

```bash
git clone https://github.com/maree217/three-layer-ai-framework
cd three-layer-ai-framework
```

## Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure Environment

```bash
# Copy the example environment file
cp templates/.env.example .env

# Edit .env with your credentials
nano .env  # or use your favorite editor
```

### Required Environment Variables

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure (optional for Layer 2 & 3)
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_TENANT_ID=your_tenant_id
```

### Getting Azure OpenAI Keys

1. Go to [Azure Portal](https://portal.azure.com)
2. Create an Azure OpenAI resource
3. Deploy GPT-4 model
4. Copy endpoint and key to .env

## Step 4: Run Your First Example

```bash
# Run the quickstart example
python examples/quickstart.py
```

You should see output like:
```
✅ Azure OpenAI connection successful!
🤖 Chatbot initialized
📊 Knowledge base loaded (150 documents)

Example query: "What is the three-layer AI framework?"
Response: "The three-layer AI framework is..."
```

## Step 5: Try the Interactive Demo

```bash
# Start the demo dashboard
python -m uvicorn src.layer3.demo_dashboard:app --reload
```

Open your browser to: http://localhost:8000

You'll see:
- Interactive chatbot (Layer 1)
- Knowledge graph visualization (Layer 2)
- Strategic dashboard (Layer 3)

## What's Next?

### Option A: Explore Examples

```bash
# Customer service chatbot
cd examples/customer_service
python customer_service_bot.py

# Housing compliance use case
cd examples/housing_compliance
python compliance_checker.py
```

### Option B: Build Your Own

Start with Layer 1 - create a simple chatbot:

```python
from src.layer1.rag_chatbot import RAGChatbot

# Initialize chatbot
bot = RAGChatbot(
    knowledge_base="./your_documents",
    model="gpt-4"
)

# Use it
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        break
    response = bot.chat(query)
    print(f"Bot: {response}")
```

### Option C: Deploy to Azure

```bash
# Build Docker image
docker build -t three-layer-ai .

# Push to Azure Container Registry
az acr build --registry myregistry --image three-layer-ai:v1 .

# Deploy to Azure Container Apps
az containerapp create \
  --name three-layer-ai \
  --resource-group myResourceGroup \
  --environment myEnvironment \
  --image myregistry.azurecr.io/three-layer-ai:v1
```

## Common Issues

### Issue: "Module not found"
**Solution**: Make sure you've activated the virtual environment and installed requirements.

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Azure OpenAI authentication failed"
**Solution**: Check your .env file has correct credentials.

```bash
# Test your connection
python -c "from openai import AzureOpenAI; print('Connection OK')"
```

### Issue: "Port 8000 already in use"
**Solution**: Use a different port.

```bash
python -m uvicorn src.layer3.demo_dashboard:app --port 8001
```

## Architecture Overview

You just set up:

```
Layer 1: UX Automation
├── RAG Chatbot (examples/quickstart.py)
└── Demo Dashboard (src.layer3.demo_dashboard)

Layer 2: Data Intelligence
├── Knowledge Graph (src/layer2/knowledge_graph.py)
└── Process Mining (ready for your data)

Layer 3: Strategic Systems
├── Forecasting Engine (src/layer3/azure_ai_foundry.py)
└── Executive Dashboard (demo_dashboard)
```

## Learning Path

1. ✅ **You are here**: Quick start completed
2. 📖 [Architecture Guide](./architecture.md) - Understand the design
3. 🎨 [Layer 1 Guide](./layer1-ux-automation.md) - Build user experiences
4. 🔧 [Layer 2 Guide](./layer2-data-intelligence.md) - Integrate your data
5. 🧠 [Layer 3 Guide](./layer3-strategic-systems.md) - Enable strategic decisions
6. 🚀 [Best Practices](./best-practices.md) - Production deployment
7. 🔌 [Integration Guide](./integrations.md) - Connect to enterprise systems

## Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/my-new-feature

# 2. Make your changes
# ... code ...

# 3. Run tests
pytest tests/

# 4. Format code
black src/ examples/
flake8 src/ examples/

# 5. Commit and push
git add .
git commit -m "Add my new feature"
git push origin feature/my-new-feature
```

## Getting Help

- 📖 [Full Documentation](./architecture.md)
- 💬 [GitHub Issues](https://github.com/maree217/three-layer-ai-framework/issues)
- 📧 Email: 2maree@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/rammaree)

## Next Steps

Choose your path:

**For Business Users**:
→ Try the demo dashboard
→ Explore the examples
→ Contact for consultation

**For Developers**:
→ Read the [API Reference](./api.md)
→ Explore the source code
→ Contribute improvements

**For Architects**:
→ Review [Architecture Guide](./architecture.md)
→ Check [Integration Guide](./integrations.md)
→ Plan your deployment

---

🎉 **Congratulations!** You've completed the quick start. Time to build something amazing!

**Questions?** Contact 2maree@gmail.com

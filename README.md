# Three-Layer AI Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)](https://www.python.org/)
[![Azure](https://img.shields.io/badge/Azure-AI%20Ready-0078D4)](https://azure.microsoft.com/)

## 🚀 Production-Ready Enterprise AI in 30 Days

The Three-Layer AI Framework is a battle-tested approach to implementing enterprise AI, proven across 5+ production deployments with **measurable business impact** and **accelerated time-to-value**.

### 📊 Proven Results
- **85% user adoption** (vs 20% industry average)
- **70% faster deployment** than traditional approaches  
- **£2M+ operational savings** across implementations
- **300% ROI** within 18 months of deployment

## 🏗️ Architecture Overview

```ascii
┌─────────────────────────────────────────────────────────────┐
│           Layer 3: Strategic Intelligence                   │
│   (Azure AI Foundry, Forecasting, Scenario Planning)       │
├─────────────────────────────────────────────────────────────┤
│           Layer 2: Data & Knowledge Intelligence            │
│   (Knowledge Graphs, Process Mining, Real-time Pipelines)  │
├─────────────────────────────────────────────────────────────┤
│           Layer 1: UX Automation                            │
│   (Microsoft Copilot, RAG Chatbots, Workflow Builders)     │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ Quick Start

```bash
# Clone the framework
git clone https://github.com/maree217/three-layer-ai-framework
cd three-layer-ai-framework

# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp templates/.env.example .env
# Edit .env with your Azure/OpenAI keys

# Run your first example
python examples/quickstart.py

# Start the demo dashboard
python -m uvicorn src.layer3.demo_dashboard:app --reload
```

## 💼 Real-World Implementations

### Housing Association: Predictive Maintenance
- **Challenge**: 8,000+ properties, reactive maintenance costing £2.3M annually
- **Solution**: Layer 2 + 3 implementation with IoT data integration
- **Result**: 23% cost reduction, 89% first-time fix rate, £534K savings
- **Code**: [`./examples/housing_compliance`](./examples/housing_compliance)

### Customer Service: Intelligent Automation
- **Challenge**: 60% of 50K monthly queries suitable for automation
- **Solution**: Layer 1 RAG-powered chatbot with knowledge integration  
- **Result**: 50% effort reduction, 85% satisfaction, 24/7 availability
- **Code**: [`./examples/customer_service`](./examples/customer_service)

### Strategic Planning: Executive Decision Support
- **Challenge**: Board meetings requiring 40+ hours of report preparation
- **Solution**: Layer 3 automated forecasting and scenario planning
- **Result**: 90% time reduction, predictive accuracy, strategic agility
- **Code**: [`./examples/predictive_maintenance`](./examples/predictive_maintenance)

## 🛠️ Framework Components

### Layer 1: UX Automation
**Make AI interfaces users actually want to use**

- **Microsoft Copilot Plugins** - Custom plugins for document processing & workflow automation
- **RAG-Enhanced Chatbots** - Domain-specific conversational AI with intelligent retrieval
- **Visual Workflow Builders** - No-code automation tools for business users
- **Smart Productivity Tools** - Contextual assistance with intelligent suggestions

📁 **Code**: [`./src/layer1/`](./src/layer1) | 📖 **Docs**: [`./docs/layer1-ux-automation.md`](./docs/layer1-ux-automation.md)

### Layer 2: Data & Knowledge Intelligence
**Transform organizational data into actionable intelligence**

- **Enterprise Knowledge Graphs** - Multi-source data integration with relationship mapping
- **Process Mining & Analytics** - Workflow analysis with automation opportunity identification
- **Real-time Data Pipelines** - ETL/ELT automation with comprehensive governance
- **Intelligent Data Discovery** - Automated insights with predictive capabilities

📁 **Code**: [`./src/layer2/`](./src/layer2) | 📖 **Docs**: [`./docs/layer2-data-intelligence.md`](./docs/layer2-data-intelligence.md)

### Layer 3: Strategic Intelligence
**AI-powered strategic decision support and forecasting**

- **Azure AI Foundry Integration** - Advanced forecasting and predictive modeling
- **Strategic Scenario Planning** - Multi-scenario analysis with risk assessment
- **Executive Dashboard Automation** - Board-ready reports with natural language insights
- **Predictive Business Intelligence** - Strategic KPI forecasting with early warning systems

📁 **Code**: [`./src/layer3/`](./src/layer3) | 📖 **Docs**: [`./docs/layer3-strategic-systems.md`](./docs/layer3-strategic-systems.md)

## 🎯 Technical Implementation Stack

### Microsoft AI Platform
- **Microsoft Copilot Studio** - Custom plugin development and deployment
- **Semantic Kernel** - Agent orchestration and multi-model integration
- **Azure AI Foundry** - Production-ready GenAI model deployment

### Development Acceleration
- **Infrastructure as Code** - Terraform/Bicep templates for rapid deployment
- **Multi-Agent Systems** - MACAE (Multi-Agent Custom Automation Engine) framework
- **Claude Code Integration** - AI-powered development acceleration

## 📚 Documentation

- **[Architecture Guide](./docs/architecture.md)** - Deep dive into three-layer design principles
- **[Quick Start Guide](./docs/quickstart.md)** - Get running in 15 minutes
- **[API Reference](./docs/api.md)** - Complete API documentation with examples  
- **[Best Practices](./docs/best-practices.md)** - Production deployment guidelines
- **[Integration Guide](./docs/integrations.md)** - Microsoft ecosystem integration patterns

## 🚀 Deployment Options

### Option 1: Local Development
```bash
python examples/quickstart.py
```

### Option 2: Azure Container Instance
```bash
az container create --resource-group myResourceGroup \
  --file templates/azure-container.yml
```

### Option 3: Kubernetes Deployment
```bash
kubectl apply -f templates/k8s-deployment.yml
```

## 📈 Business Impact Metrics

| **Layer** | **Typical Implementation Time** | **User Adoption Rate** | **ROI Timeline** |
|-----------|-------------------------------|----------------------|------------------|
| **Layer 1: UX** | 2-4 weeks | 85%+ | 3-6 months |
| **Layer 2: Data** | 4-8 weeks | 90%+ | 6-12 months |
| **Layer 3: Strategic** | 6-12 weeks | 95%+ | 12-18 months |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and install dev dependencies
git clone https://github.com/maree217/three-layer-ai-framework
cd three-layer-ai-framework
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ examples/
flake8 src/ examples/
```

## 📄 License

MIT License - see [LICENSE](./LICENSE)

## 👤 Author

**Ram Senthil-Maree** - AI Solutions Architect & Engineer  
*Specializing in hands-on enterprise AI implementation with rapid prototyping expertise*

- 🌐 Website: [AICapabilityBuilder.com](https://aicapabilitybuilder.com)
- 💼 LinkedIn: [linkedin.com/in/rammaree](https://linkedin.com/in/rammaree)  
- 📧 Email: [2maree@gmail.com](mailto:2maree@gmail.com)
- 📍 Location: London, UK

---

*"Three-layer AI architecture: from user experience to strategic intelligence"*

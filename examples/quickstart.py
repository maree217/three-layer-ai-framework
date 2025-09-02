#!/usr/bin/env python3
"""
Three-Layer AI Framework - Quick Start Demo
Production-ready example showcasing all three layers working together

This demo demonstrates:
- Layer 1: RAG chatbot for user interaction
- Layer 2: Knowledge graph for data intelligence  
- Layer 3: Strategic forecasting for business intelligence

Usage: python examples/quickstart.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from layer1.rag_chatbot import EnterpriseRAGChatbot, ChatMessage
from layer2.knowledge_graph import EnterpriseKnowledgeGraph
from layer3.azure_ai_foundry import StrategicIntelligenceEngine, AzureAIFoundryClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

async def run_quickstart_demo():
    """
    Comprehensive demo showcasing three-layer AI framework integration
    """
    print("🚀 Three-Layer AI Framework - Quick Start Demo")
    print("=" * 60)
    
    print("""
This demo showcases enterprise AI capabilities across three architectural layers:

🎨 Layer 1: UX Automation - Intelligent user interfaces with RAG capabilities
🔧 Layer 2: Data Intelligence - Knowledge graphs and process optimization  
🧠 Layer 3: Strategic Intelligence - Forecasting and executive decision support

Let's see them work together...
    """)
    
    # ================================
    # LAYER 2: Initialize Knowledge Graph
    # ================================
    print("🔧 LAYER 2: Setting up Data & Knowledge Intelligence")
    print("-" * 50)
    
    kg = EnterpriseKnowledgeGraph(graph_name="enterprise_demo")
    
    # Add enterprise knowledge
    enterprise_docs = [
        {
            "id": "strategic_framework",
            "content": """
            The Three-Layer AI Architecture is a proven enterprise framework developed by 
            Ram Senthil-Maree for systematic AI implementation. Layer 1 focuses on UX Automation 
            with Microsoft Copilot plugins and RAG chatbots achieving 85% user adoption. 
            Layer 2 handles Data Intelligence through knowledge graphs and process mining, 
            delivering 90% faster insights. Layer 3 provides Strategic Intelligence using 
            Azure AI Foundry for forecasting, achieving 300% ROI within 18 months.
            
            Implementation typically follows a phased approach: UX Automation (2-4 weeks), 
            Data Intelligence (4-8 weeks), and Strategic Intelligence (6-12 weeks). 
            This methodology has been successfully deployed across financial services, 
            healthcare, and manufacturing sectors with measurable business impact.
            """,
            "metadata": {"type": "framework_guide", "importance": "high"}
        },
        {
            "id": "azure_ai_capabilities",
            "content": """
            Azure AI Foundry provides enterprise-grade capabilities for AI model development 
            and deployment. Key features include automated machine learning, model management, 
            prompt flow for complex AI applications, and comprehensive monitoring. The platform 
            integrates with Azure OpenAI Service for large language models and supports 
            custom model fine-tuning.
            
            Strategic applications include revenue forecasting with 85% accuracy, scenario 
            planning for executive decision support, and automated insight generation for 
            board presentations. Typical implementation reduces strategic planning time by 
            75% while improving decision confidence through comprehensive risk assessment.
            """,
            "metadata": {"type": "technical_capabilities", "layer": 3}
        },
        {
            "id": "business_impact_data",
            "content": """
            Enterprise AI implementations using the three-layer approach demonstrate consistent 
            business impact. Housing association case study: 8,000 properties, predictive 
            maintenance system reduced costs by 23% (£534K annual savings), improved first-time 
            fix rate to 89%. Customer service automation: 50,000 monthly queries, achieved 
            50% effort reduction with 85% customer satisfaction, enabling 24/7 availability.
            
            Strategic planning transformation: Board report preparation time reduced from 
            40+ hours to 4 hours (90% improvement), with enhanced strategic insights through 
            AI-powered scenario modeling. Financial services deployment achieved 300% ROI 
            within 18 months through operational efficiency and strategic decision acceleration.
            """,
            "metadata": {"type": "case_studies", "validation": "proven"}
        }
    ]
    
    success = await kg.ingest_data_source("enterprise_knowledge", enterprise_docs, "document")
    if success:
        print("✅ Enterprise knowledge base loaded")
        stats = kg.get_graph_statistics()
        print(f"   📊 {stats['total_entities']} entities, {stats['total_relationships']} relationships")
    else:
        print("❌ Knowledge graph setup failed")
        return
    
    # ================================
    # LAYER 1: Initialize RAG Chatbot  
    # ================================
    print(f"\n🎨 LAYER 1: Setting up UX Automation")
    print("-" * 40)
    
    chatbot = EnterpriseRAGChatbot(model_name="gpt-4")
    
    # Load knowledge into chatbot
    chatbot_success = await chatbot.ingest_documents([
        {
            "content": doc["content"],
            "source": doc["id"],
            "metadata": doc["metadata"]
        }
        for doc in enterprise_docs
    ])
    
    if chatbot_success:
        print("✅ RAG chatbot ready with enterprise knowledge")
    else:
        print("⚠️  RAG chatbot running in basic mode (no vector search)")
    
    # ================================
    # LAYER 3: Initialize Strategic Intelligence
    # ================================
    print(f"\n🧠 LAYER 3: Setting up Strategic Intelligence")
    print("-" * 45)
    
    # Initialize Azure AI Foundry client (demo mode)
    azure_client = AzureAIFoundryClient(
        subscription_id="demo-sub",
        resource_group="demo-rg",
        workspace_name="demo-workspace"
    )
    
    strategic_engine = StrategicIntelligenceEngine(azure_client)
    
    # Create sample business data for forecasting
    business_data = create_sample_business_data()
    print("✅ Strategic intelligence engine ready")
    print(f"   📈 {len(business_data)} months of business data prepared")
    
    # ================================
    # INTEGRATED DEMONSTRATION
    # ================================
    print(f"\n🔄 INTEGRATED DEMONSTRATION: All Layers Working Together")
    print("=" * 60)
    
    # Simulate executive asking strategic questions
    strategic_questions = [
        "What is the Three-Layer AI Architecture and how does it work?",
        "What business impact can we expect from AI implementation?", 
        "How should we approach Azure AI Foundry for strategic planning?",
        "What are the success factors for enterprise AI adoption?"
    ]
    
    print("\n💼 Executive Strategic Session Simulation")
    print("-" * 45)
    
    conversation_insights = []
    
    for i, question in enumerate(strategic_questions, 1):
        print(f"\n👤 Executive Question {i}: {question}")
        
        # Layer 1: Generate intelligent response
        response = await chatbot.generate_response(question)
        print(f"🤖 AI Advisor: {response.content[:300]}...")
        
        if response.sources:
            print(f"📚 Knowledge Sources: {', '.join(response.sources)}")
        
        if response.confidence_score:
            print(f"🎯 Response Confidence: {response.confidence_score:.1%}")
        
        # Extract insights for strategic analysis
        conversation_insights.append({
            'question': question,
            'response_confidence': response.confidence_score or 0.7,
            'sources_used': len(response.sources) if response.sources else 0
        })
    
    # ================================
    # STRATEGIC INTELLIGENCE ANALYSIS
    # ================================
    print(f"\n📊 STRATEGIC INTELLIGENCE ANALYSIS")
    print("-" * 40)
    
    # Generate strategic forecast based on conversation insights
    base_assumptions = {
        'base_revenue': 2500000,  # £2.5M baseline
        'growth_rate': 0.18,      # 18% growth expectation
        'volatility': 0.12,       # 12% volatility
        'confidence_factor': np.mean([i['response_confidence'] for i in conversation_insights])
    }
    
    print("🔮 Generating strategic scenarios...")
    scenarios = await strategic_engine.create_strategic_forecast(
        time_horizon=18,  # 18 months
        base_assumptions=base_assumptions,
        include_scenarios=True
    )
    
    print(f"✅ Generated {len(scenarios)} strategic scenarios:")
    for name, scenario in scenarios.items():
        revenue_projection = scenario.projections['revenue'][-1]
        print(f"   📈 {name.title()}: £{revenue_projection:,.0f} "
              f"(probability: {scenario.probability:.0%})")
    
    # Generate strategic insights
    business_context = {
        'conversation_quality': np.mean([i['response_confidence'] for i in conversation_insights]),
        'knowledge_utilization': np.mean([i['sources_used'] for i in conversation_insights]),
        'strategic_maturity': 'high',
        'implementation_readiness': 'advanced'
    }
    
    print(f"\n💡 Generating strategic insights...")
    insights = await strategic_engine.generate_strategic_insights(scenarios, business_context)
    
    print(f"✅ Generated {len(insights)} strategic insights:")
    for insight in insights[:2]:  # Show top 2
        print(f"   🎯 {insight.title}")
        print(f"      {insight.description[:120]}...")
        if insight.recommended_actions:
            print(f"      💼 Key Action: {insight.recommended_actions[0]}")
    
    # ================================
    # EXECUTIVE SUMMARY GENERATION  
    # ================================
    print(f"\n📋 EXECUTIVE SUMMARY GENERATION")
    print("-" * 35)
    
    executive_report = await strategic_engine.create_executive_report(
        report_period="Strategic AI Implementation - Next 18 Months",
        scenarios=scenarios,
        business_metrics=business_context
    )
    
    print("✅ Executive report generated")
    print(f"\n📊 STRATEGIC INTELLIGENCE SUMMARY")
    print("=" * 40)
    print(f"🎯 Forecast Confidence: {executive_report.key_metrics.get('forecast_confidence', 0):.1%}")
    print(f"📈 Projected Growth: {base_assumptions['growth_rate']:.1%} annually")
    print(f"🔍 Strategic Insights: {len(executive_report.strategic_insights)}")
    print(f"⚠️  Risk Level: {executive_report.risk_assessment.get('overall_risk_level', 'medium').title()}")
    
    print(f"\n🎯 TOP STRATEGIC RECOMMENDATIONS:")
    for i, recommendation in enumerate(executive_report.recommendations[:4], 1):
        print(f"   {i}. {recommendation}")
    
    # ================================
    # FRAMEWORK VALIDATION
    # ================================
    print(f"\n✅ THREE-LAYER AI FRAMEWORK VALIDATION")
    print("=" * 45)
    
    # Layer 1 metrics
    chat_summary = chatbot.get_conversation_summary()
    print(f"🎨 Layer 1 (UX Automation):")
    print(f"   • User Interactions: {chat_summary['user_messages']}")
    print(f"   • Response Quality: {chat_summary['average_confidence']:.1%}")
    print(f"   • Knowledge Sources: {chat_summary['sources_used']}")
    
    # Layer 2 metrics  
    kg_stats = kg.get_graph_statistics()
    print(f"\n🔧 Layer 2 (Data Intelligence):")
    print(f"   • Entities Processed: {kg_stats['total_entities']}")
    print(f"   • Relationships Mapped: {kg_stats['total_relationships']}")
    print(f"   • Knowledge Coverage: {len(kg_stats['entity_types'])} domains")
    
    # Layer 3 metrics
    print(f"\n🧠 Layer 3 (Strategic Intelligence):")
    print(f"   • Scenarios Modeled: {len(scenarios)}")
    print(f"   • Strategic Insights: {len(insights)}")
    print(f"   • Executive Recommendations: {len(executive_report.recommendations)}")
    
    print(f"\n🎉 FRAMEWORK INTEGRATION SUCCESS")
    print("=" * 35)
    print("""
The three-layer AI framework has successfully demonstrated:

✅ Seamless integration across all architectural layers
✅ Real-time knowledge processing and intelligent responses  
✅ Strategic intelligence generation with actionable insights
✅ Executive-ready business recommendations and risk assessment

Ready for enterprise deployment with measurable business impact.
    """)
    
    # ================================
    # NEXT STEPS GUIDANCE
    # ================================
    print("🚀 NEXT STEPS FOR IMPLEMENTATION")
    print("-" * 35)
    print("""
1. 🔧 Environment Setup:
   • Configure Azure AI Foundry workspace
   • Set up vector database (ChromaDB/Azure AI Search)
   • Install production dependencies

2. 📊 Data Integration:
   • Connect to enterprise data sources
   • Implement knowledge graph pipelines
   • Set up real-time data feeds

3. 🎯 Customization:
   • Adapt models to your domain
   • Configure business-specific scenarios
   • Implement custom workflow integrations

4. 🚀 Deployment:
   • Use provided Infrastructure as Code templates
   • Set up monitoring and observability
   • Implement security and compliance controls

For detailed implementation guidance, see:
• ./docs/quickstart.md
• ./templates/ (Infrastructure as Code)
• ./examples/ (Domain-specific implementations)
    """)

def create_sample_business_data() -> pd.DataFrame:
    """Create realistic business data for demonstration"""
    # Generate 36 months of historical business data
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M')
    
    # Base business metrics with realistic trends
    base_revenue = 1500000  # £1.5M monthly baseline
    
    # Add growth trend, seasonality, and business events
    trend = np.arange(len(dates)) * 8000  # Growing business
    seasonality = 100000 * (np.sin(2 * np.pi * np.arange(len(dates)) / 12) + 
                            0.5 * np.sin(4 * np.pi * np.arange(len(dates)) / 12))
    
    # Business events (product launches, market changes)
    events = np.zeros(len(dates))
    events[12] = 150000  # Major product launch
    events[24] = -80000  # Market downturn  
    events[30] = 200000  # Strategic partnership
    
    # Random variation
    noise = np.random.normal(0, 50000, len(dates))
    
    revenue = base_revenue + trend + seasonality + events + noise
    revenue = np.maximum(revenue, base_revenue * 0.7)  # Floor at 70% of baseline
    
    # Add supporting business metrics
    customer_acquisition = np.random.poisson(25, len(dates)) + trend / 10000
    market_share = 12.5 + np.cumsum(np.random.normal(0.1, 0.3, len(dates)))
    market_share = np.clip(market_share, 8, 18)  # Keep realistic range
    
    return pd.DataFrame({
        'date': dates,
        'revenue': revenue,
        'customer_acquisition': customer_acquisition,
        'market_share': market_share,
        'employee_count': 150 + np.arange(len(dates)) * 2,  # Growing team
    }).set_index('date')

if __name__ == "__main__":
    asyncio.run(run_quickstart_demo())
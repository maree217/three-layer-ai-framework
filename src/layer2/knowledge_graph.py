"""
Layer 2: Data & Knowledge Intelligence - Enterprise Knowledge Graph
Production-ready knowledge graph implementation for organizational data integration

Features:
- Multi-source data integration with relationship mapping
- Automated entity extraction and relationship discovery
- Graph-based RAG with semantic search capabilities
- Process mining and workflow pattern analysis
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re

try:
    import networkx as nx
    import pandas as pd
    import numpy as np
    from neo4j import GraphDatabase
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Optional dependencies not installed. Run: pip install networkx pandas neo4j spacy scikit-learn")

@dataclass
class Entity:
    """Knowledge graph entity with properties"""
    id: str
    label: str
    entity_type: str
    properties: Dict[str, Any]
    confidence_score: float
    source_documents: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass  
class Relationship:
    """Knowledge graph relationship between entities"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence_score: float
    source_documents: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ProcessStep:
    """Business process step for workflow analysis"""
    id: str
    name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    duration_avg: float
    frequency: int
    automation_potential: float
    dependencies: List[str]

class EnterpriseKnowledgeGraph:
    """
    Enterprise-grade knowledge graph for organizational intelligence
    Integrates multiple data sources with automated relationship discovery
    """
    
    def __init__(self, 
                 graph_name: str = "enterprise_graph",
                 use_neo4j: bool = False,
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None):
        
        self.graph_name = graph_name
        self.use_neo4j = use_neo4j
        
        # Initialize graph storage
        if use_neo4j and neo4j_uri:
            self._initialize_neo4j(neo4j_uri, neo4j_user, neo4j_password)
        else:
            self._initialize_networkx()
        
        # Entity and relationship storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # NLP processing
        self._initialize_nlp()
        
        # Process mining data
        self.processes: Dict[str, ProcessStep] = {}
        self.workflow_patterns: List[Dict[str, Any]] = []
    
    def _initialize_neo4j(self, uri: str, user: str, password: str):
        """Initialize Neo4j database connection"""
        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            self.graph_db = self.neo4j_driver
            print("✅ Connected to Neo4j database")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            self._initialize_networkx()
    
    def _initialize_networkx(self):
        """Initialize NetworkX graph for local processing"""
        self.graph = nx.MultiDiGraph()
        self.neo4j_driver = None
        print("✅ Using NetworkX for local graph storage")
    
    def _initialize_nlp(self):
        """Initialize NLP components for entity extraction"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ NLP model loaded successfully")
        except OSError:
            print("❌ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    async def ingest_data_source(self, 
                                source_name: str, 
                                data: List[Dict[str, Any]], 
                                source_type: str = "document") -> bool:
        """
        Ingest data from various sources into knowledge graph
        
        Args:
            source_name: Name of the data source
            data: List of data items to process
            source_type: Type of source (document, database, api, etc.)
        
        Returns:
            bool: Success status
        """
        try:
            if source_type == "document":
                return await self._process_documents(source_name, data)
            elif source_type == "database":
                return await self._process_database_records(source_name, data)
            elif source_type == "process_logs":
                return await self._process_workflow_logs(source_name, data)
            else:
                print(f"❌ Unsupported source type: {source_type}")
                return False
                
        except Exception as e:
            print(f"❌ Data ingestion error: {e}")
            return False
    
    async def _process_documents(self, source_name: str, documents: List[Dict[str, Any]]) -> bool:
        """Process text documents for entity and relationship extraction"""
        if not self.nlp:
            print("❌ NLP model not available")
            return False
        
        for doc in documents:
            content = doc.get('content', '')
            doc_id = doc.get('id', f"{source_name}_{hash(content)}")
            
            # Extract entities from document
            entities = self._extract_entities(content, doc_id)
            
            # Extract relationships between entities  
            relationships = self._extract_relationships(content, entities, doc_id)
            
            # Add to knowledge graph
            for entity in entities:
                await self._add_entity(entity)
            
            for relationship in relationships:
                await self._add_relationship(relationship)
        
        return True
    
    def _extract_entities(self, text: str, source_doc: str) -> List[Entity]:
        """Extract entities from text using NLP"""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entity_id = self._generate_entity_id(ent.text, ent.label_)
            
            entity = Entity(
                id=entity_id,
                label=ent.text,
                entity_type=ent.label_,
                properties={
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "context": text[max(0, ent.start_char-50):ent.end_char+50]
                },
                confidence_score=0.8,  # Default confidence
                source_documents=[source_doc]
            )
            entities.append(entity)
        
        # Extract key terms and concepts
        key_terms = self._extract_key_terms(text)
        for term in key_terms:
            entity_id = self._generate_entity_id(term, "CONCEPT")
            
            entity = Entity(
                id=entity_id,
                label=term,
                entity_type="CONCEPT",
                properties={
                    "extraction_method": "tfidf",
                    "context": text
                },
                confidence_score=0.6,
                source_documents=[source_doc]
            )
            entities.append(entity)
        
        return entities
    
    def _extract_key_terms(self, text: str, max_terms: int = 10) -> List[str]:
        """Extract key terms using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=max_terms,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            top_indices = tfidf_scores.argsort()[-max_terms:][::-1]
            key_terms = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0.1]
            
            return key_terms
            
        except Exception as e:
            print(f"Key term extraction error: {e}")
            return []
    
    def _extract_relationships(self, text: str, entities: List[Entity], source_doc: str) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships = []
        
        # Simple co-occurrence based relationship extraction
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities co-occur within a reasonable distance
                distance = self._calculate_entity_distance(entity1, entity2, text)
                
                if distance < 100:  # Within 100 characters
                    relationship_id = self._generate_relationship_id(entity1.id, entity2.id)
                    
                    relationship = Relationship(
                        id=relationship_id,
                        source_entity_id=entity1.id,
                        target_entity_id=entity2.id,
                        relationship_type="RELATED_TO",
                        properties={
                            "distance": distance,
                            "context": self._get_relationship_context(entity1, entity2, text)
                        },
                        confidence_score=max(0.3, 1.0 - (distance / 100)),
                        source_documents=[source_doc]
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _calculate_entity_distance(self, entity1: Entity, entity2: Entity, text: str) -> float:
        """Calculate distance between entities in text"""
        pos1 = entity1.properties.get('start_char', 0)
        pos2 = entity2.properties.get('start_char', 0)
        return abs(pos1 - pos2)
    
    def _get_relationship_context(self, entity1: Entity, entity2: Entity, text: str) -> str:
        """Get context surrounding both entities"""
        start_pos = min(
            entity1.properties.get('start_char', 0),
            entity2.properties.get('start_char', 0)
        ) - 50
        
        end_pos = max(
            entity1.properties.get('end_char', 0),
            entity2.properties.get('end_char', 0)
        ) + 50
        
        return text[max(0, start_pos):min(len(text), end_pos)]
    
    async def _add_entity(self, entity: Entity):
        """Add entity to knowledge graph"""
        # Check if entity already exists
        if entity.id in self.entities:
            # Merge with existing entity
            existing = self.entities[entity.id]
            existing.source_documents.extend(entity.source_documents)
            existing.source_documents = list(set(existing.source_documents))  # Remove duplicates
            
            # Update confidence score (weighted average)
            total_docs = len(existing.source_documents)
            existing.confidence_score = (
                (existing.confidence_score * (total_docs - len(entity.source_documents)) + 
                 entity.confidence_score * len(entity.source_documents)) / total_docs
            )
        else:
            self.entities[entity.id] = entity
        
        # Add to graph storage
        if hasattr(self, 'graph'):
            self.graph.add_node(entity.id, **entity.to_dict())
    
    async def _add_relationship(self, relationship: Relationship):
        """Add relationship to knowledge graph"""
        # Check if relationship already exists
        if relationship.id in self.relationships:
            # Merge with existing relationship
            existing = self.relationships[relationship.id]
            existing.source_documents.extend(relationship.source_documents)
            existing.source_documents = list(set(existing.source_documents))
            
            # Update confidence score
            total_docs = len(existing.source_documents)
            existing.confidence_score = (
                (existing.confidence_score * (total_docs - len(relationship.source_documents)) + 
                 relationship.confidence_score * len(relationship.source_documents)) / total_docs
            )
        else:
            self.relationships[relationship.id] = relationship
        
        # Add to graph storage
        if hasattr(self, 'graph'):
            self.graph.add_edge(
                relationship.source_entity_id,
                relationship.target_entity_id,
                key=relationship.id,
                **relationship.to_dict()
            )
    
    def _generate_entity_id(self, label: str, entity_type: str) -> str:
        """Generate unique entity ID"""
        content = f"{label.lower()}_{entity_type.lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_relationship_id(self, source_id: str, target_id: str) -> str:
        """Generate unique relationship ID"""
        content = f"{source_id}_{target_id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def _process_workflow_logs(self, source_name: str, logs: List[Dict[str, Any]]) -> bool:
        """Process workflow logs for process mining"""
        try:
            # Extract process steps
            for log_entry in logs:
                step = ProcessStep(
                    id=log_entry.get('step_id', f"step_{len(self.processes)}"),
                    name=log_entry.get('step_name', 'Unknown Step'),
                    description=log_entry.get('description', ''),
                    inputs=log_entry.get('inputs', []),
                    outputs=log_entry.get('outputs', []),
                    duration_avg=log_entry.get('duration', 0.0),
                    frequency=log_entry.get('frequency', 1),
                    automation_potential=self._assess_automation_potential(log_entry),
                    dependencies=log_entry.get('dependencies', [])
                )
                self.processes[step.id] = step
            
            # Analyze workflow patterns
            self._analyze_workflow_patterns()
            
            return True
            
        except Exception as e:
            print(f"Workflow processing error: {e}")
            return False
    
    def _assess_automation_potential(self, log_entry: Dict[str, Any]) -> float:
        """Assess automation potential for a process step"""
        # Simple heuristic based on step properties
        factors = {
            'repetitive': 0.3,
            'rule_based': 0.3,
            'data_processing': 0.2,
            'low_complexity': 0.2
        }
        
        automation_score = 0.0
        
        # Check for automation indicators
        description = log_entry.get('description', '').lower()
        
        if any(keyword in description for keyword in ['copy', 'paste', 'transfer', 'move']):
            automation_score += factors['repetitive']
        
        if any(keyword in description for keyword in ['if', 'when', 'condition', 'rule']):
            automation_score += factors['rule_based']
        
        if any(keyword in description for keyword in ['data', 'file', 'record', 'database']):
            automation_score += factors['data_processing']
        
        if len(description.split()) < 10:  # Simple description suggests low complexity
            automation_score += factors['low_complexity']
        
        return min(automation_score, 1.0)
    
    def _analyze_workflow_patterns(self):
        """Analyze workflow patterns for optimization opportunities"""
        if len(self.processes) < 2:
            return
        
        # Identify bottlenecks
        bottlenecks = []
        for process_id, process in self.processes.items():
            if process.duration_avg > np.percentile([p.duration_avg for p in self.processes.values()], 75):
                bottlenecks.append({
                    'process_id': process_id,
                    'name': process.name,
                    'avg_duration': process.duration_avg,
                    'optimization_type': 'duration_bottleneck'
                })
        
        # Identify high automation potential
        automation_candidates = []
        for process_id, process in self.processes.items():
            if process.automation_potential > 0.6:
                automation_candidates.append({
                    'process_id': process_id,
                    'name': process.name,
                    'automation_potential': process.automation_potential,
                    'optimization_type': 'automation_candidate'
                })
        
        self.workflow_patterns = bottlenecks + automation_candidates
    
    def query_graph(self, 
                   entity_type: Optional[str] = None,
                   relationship_type: Optional[str] = None,
                   source_document: Optional[str] = None,
                   min_confidence: float = 0.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query the knowledge graph with filters
        
        Args:
            entity_type: Filter by entity type
            relationship_type: Filter by relationship type
            source_document: Filter by source document
            min_confidence: Minimum confidence score
        
        Returns:
            Dictionary with filtered entities and relationships
        """
        filtered_entities = []
        filtered_relationships = []
        
        # Filter entities
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if source_document and source_document not in entity.source_documents:
                continue
            if entity.confidence_score < min_confidence:
                continue
            
            filtered_entities.append(entity.to_dict())
        
        # Filter relationships
        for relationship in self.relationships.values():
            if relationship_type and relationship.relationship_type != relationship_type:
                continue
            if source_document and source_document not in relationship.source_documents:
                continue
            if relationship.confidence_score < min_confidence:
                continue
            
            filtered_relationships.append(relationship.to_dict())
        
        return {
            'entities': filtered_entities,
            'relationships': filtered_relationships
        }
    
    def find_entity_connections(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Find all entities connected to a given entity within max_depth"""
        if not hasattr(self, 'graph') or entity_id not in self.graph:
            return {'connections': [], 'paths': []}
        
        connections = []
        paths = []
        
        # Use NetworkX to find connections
        try:
            # Direct connections (depth 1)
            neighbors = list(self.graph.neighbors(entity_id))
            for neighbor in neighbors:
                if neighbor in self.entities:
                    connections.append({
                        'entity': self.entities[neighbor].to_dict(),
                        'depth': 1,
                        'relationship_id': self._find_relationship_id(entity_id, neighbor)
                    })
            
            # Deeper connections if requested
            if max_depth > 1:
                for target in neighbors:
                    try:
                        shortest_paths = nx.all_simple_paths(
                            self.graph, entity_id, target, cutoff=max_depth
                        )
                        for path in shortest_paths:
                            if len(path) <= max_depth + 1:
                                paths.append({
                                    'path': path,
                                    'length': len(path) - 1,
                                    'entities': [self.entities.get(node, {}).get('label', node) for node in path]
                                })
                    except nx.NetworkXNoPath:
                        continue
                        
        except Exception as e:
            print(f"Connection finding error: {e}")
        
        return {
            'connections': connections,
            'paths': paths[:10]  # Limit to 10 paths to avoid overwhelming results
        }
    
    def _find_relationship_id(self, source_id: str, target_id: str) -> Optional[str]:
        """Find relationship ID between two entities"""
        for relationship in self.relationships.values():
            if (relationship.source_entity_id == source_id and 
                relationship.target_entity_id == target_id) or \
               (relationship.source_entity_id == target_id and 
                relationship.target_entity_id == source_id):
                return relationship.id
        return None
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        entity_types = {}
        relationship_types = {}
        
        for entity in self.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        for relationship in self.relationships.values():
            rel_type = relationship.relationship_type
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'entity_types': entity_types,
            'relationship_types': relationship_types,
            'total_processes': len(self.processes),
            'workflow_patterns': len(self.workflow_patterns),
            'avg_entity_confidence': np.mean([e.confidence_score for e in self.entities.values()]) if self.entities else 0,
            'avg_relationship_confidence': np.mean([r.confidence_score for r in self.relationships.values()]) if self.relationships else 0
        }
    
    def export_graph(self, format: str = "json", file_path: Optional[str] = None) -> str:
        """Export knowledge graph in various formats"""
        if format == "json":
            export_data = {
                'entities': [entity.to_dict() for entity in self.entities.values()],
                'relationships': [rel.to_dict() for rel in self.relationships.values()],
                'processes': [asdict(process) for process in self.processes.values()],
                'workflow_patterns': self.workflow_patterns,
                'statistics': self.get_graph_statistics(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            export_json = json.dumps(export_data, indent=2)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(export_json)
                print(f"✅ Graph exported to {file_path}")
            
            return export_json
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Demo function
async def demo_knowledge_graph():
    """Demonstrate knowledge graph capabilities"""
    print("🕸️ Enterprise Knowledge Graph Demo")
    print("=" * 45)
    
    # Initialize knowledge graph
    kg = EnterpriseKnowledgeGraph()
    
    # Sample documents
    sample_docs = [
        {
            "id": "doc_1",
            "content": """
            Microsoft Azure AI Foundry provides comprehensive tools for developing and deploying 
            enterprise AI applications. The platform integrates with Azure OpenAI Service to 
            enable advanced language models and supports fine-tuning for domain-specific use cases.
            
            Key features include model management, prompt engineering tools, and automated 
            evaluation pipelines. Organizations can achieve 60% faster deployment times using 
            the integrated development environment.
            """,
            "metadata": {"type": "technical_documentation", "layer": 3}
        },
        {
            "id": "doc_2", 
            "content": """
            The Three-Layer AI Architecture consists of UX Automation, Data Intelligence, and 
            Strategic Intelligence layers. UX Automation focuses on user-facing applications 
            like Microsoft Copilot plugins and RAG chatbots. Data Intelligence handles knowledge 
            graphs and process mining. Strategic Intelligence implements forecasting and 
            scenario planning using Azure AI Foundry.
            
            Ram Senthil-Maree developed this framework based on 15+ years of enterprise AI 
            implementation experience. The methodology has achieved 85% user adoption rates 
            across multiple industry deployments.
            """,
            "metadata": {"type": "architecture_guide", "author": "ram_senthil_maree"}
        }
    ]
    
    # Ingest documents
    print("📚 Ingesting documents into knowledge graph...")
    success = await kg.ingest_data_source("documentation", sample_docs, "document")
    
    if success:
        print("✅ Documents processed successfully")
    else:
        print("❌ Document processing failed")
        return
    
    # Sample process logs
    process_logs = [
        {
            "step_id": "data_extraction",
            "step_name": "Extract Data from Source Systems",
            "description": "Copy customer data from CRM system to processing database",
            "duration": 45.0,
            "frequency": 100,
            "inputs": ["crm_database"],
            "outputs": ["staging_table"]
        },
        {
            "step_id": "data_validation", 
            "step_name": "Validate Data Quality",
            "description": "Check data completeness and format rules using predefined criteria",
            "duration": 15.0,
            "frequency": 100,
            "inputs": ["staging_table"],
            "outputs": ["validated_data"]
        }
    ]
    
    print("\n🔄 Processing workflow logs...")
    await kg.ingest_data_source("workflow_system", process_logs, "process_logs")
    
    # Display graph statistics
    print(f"\n📊 Knowledge Graph Statistics:")
    stats = kg.get_graph_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Query examples
    print(f"\n🔍 Query Examples:")
    
    # Find all PERSON entities
    persons = kg.query_graph(entity_type="PERSON", min_confidence=0.5)
    print(f"Found {len(persons['entities'])} person entities:")
    for entity in persons['entities'][:3]:  # Show first 3
        print(f"  - {entity['label']} (confidence: {entity['confidence_score']:.2f})")
    
    # Find high automation potential processes
    print(f"\n🤖 High Automation Potential Processes:")
    for process in kg.processes.values():
        if process.automation_potential > 0.6:
            print(f"  - {process.name}: {process.automation_potential:.1%} potential")
    
    # Show workflow patterns
    print(f"\n📈 Workflow Optimization Opportunities:")
    for pattern in kg.workflow_patterns:
        print(f"  - {pattern['name']}: {pattern['optimization_type']}")
    
    # Export graph
    print(f"\n💾 Exporting knowledge graph...")
    export_path = "./knowledge_graph_export.json"
    kg.export_graph("json", export_path)

if __name__ == "__main__":
    asyncio.run(demo_knowledge_graph())
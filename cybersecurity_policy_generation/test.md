## Functional Model Diagram
```mermaid
flowchart TB
  %% theme + elk renderer
  %%{
    init: {
      "themeVariables": { 
        "clusterBkg": "#f3f3f3",
        "clusterBorder": "#999",
        "labelBackground": "#ffffff",
        "edgeLabelBackground": "#ffffff",
        "edgeLabelBorder": "#999",
        "labelBorder": "#999"
      }
    }
  }%%
  %%{init: {"flowchart": {"defaultRenderer": "elk"}}}%%

  %% STYLES
  classDef box fill:#fff,stroke:#333,stroke-width:1px,rx:8px,ry:8px;
  classDef Neo4j fill:#C9E4FF,stroke:#333,stroke-width:1px,rx:8px,ry:8px;
  classDef Conductor fill:#E6E6E6,stroke:#333,stroke-width:1px,rx:8px,ry:8px;
  classDef lightTeal fill:#5aa9c9,stroke:#333,stroke-width:1px,color:#fff,rx:8px,ry:8px;
  classDef Agents fill:#C2F0C2,stroke:#333,stroke-width:1px,rx:8px,ry:8px;
  classDef Client fill:#FFF6BF,stroke:#333,stroke-width:1px,rx:8px,ry:8px;
  classDef Output fill:#FFD9B3,stroke:#333,stroke-width:1px,rx:8px,ry:8px;
  classDef Validation fill:#E4D3F8,stroke:#333,stroke-width:1px,rx:8px,ry:8px;

   %% NODES

  subgraph ONT["Model Prompt Inputs"]
    direction LR
    OA1["Instruction Prompt"]:::Validation
    OA2["Company Ontology "]:::Conductor
    OA3["Ontology Prompt"]:::Agents
  end

  subgraph META["Test Model"]
    direction TB
    BM["Base model<br/>Llama-3-8B-Instruct"]:::lightTeal
  end

  subgraph KG["Neo4j Knowledge Graph"]
    direction TB
    KG1["Policy to Section Mapping"]:::Neo4j
    KG2["Policy to Framework Mapping"]:::Neo4j
    KG3["Ontology Mapped to Policy Phrase/Section"]:::Neo4j
  end
 
  PS1["Policy Phrase"]:::box
  
  SME["SME  Supervised Training<br/>A/B Testing
  "]:::Output
  
  %% FLOWS
  OA1 --> BM
  OA2 --> BM
  OA3 --> BM

  KG1 --> BM
  KG2 --> BM
  KG3 --> BM

  SME --> BM

  BM --> PS1
  

 ```
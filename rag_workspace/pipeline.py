import os
import json
import logging
import yaml
import uuid
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import langextract as lx
from langextract.core.data import Extraction, ExampleData
from langextract.providers.openai import OpenAILanguageModel
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION (from settings.yaml) ---
with open(Path(__file__).resolve().parent / "settings.yaml", "r") as f:
    config = yaml.safe_load(f)

PROVIDER = "vllm"  # Force to vllm based on the environment setup
INPUT_DIR = config.get("input", {}).get("base_dir", "input")
if not os.path.isabs(INPUT_DIR):
    INPUT_DIR = os.path.join(str(Path(__file__).resolve().parent), INPUT_DIR)
OUTPUT_DIR = config.get("output_storage", {}).get("base_dir", "output")
if not os.path.isabs(OUTPUT_DIR):
    OUTPUT_DIR = os.path.join(str(Path(__file__).resolve().parent), OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resolve provider-specific settings from the MS GraphRAG structure
provider_cfg = config.get("completion_models", {}).get("default_completion_model", {})
MODEL_NAME = os.getenv("GRAPHRAG_LLM_MODEL", provider_cfg.get("model", "meta-llama/Meta-Llama-3-8B-Instruct"))
MODEL_URL = provider_cfg.get("api_base", "http://localhost:8000/v1")
API_KEY = "vllm"

def build_model():
    """Build the LangExtract model based on provider config."""
    return OpenAILanguageModel(
        model_id=MODEL_NAME,
        api_key=API_KEY,
        base_url=MODEL_URL,
    )


# Extraction prompt
EXTRACT_PROMPT = """
Extract all entities and relationships from the text.
For each entity found, classify it as one of: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, MEDICAL_CONDITION, MEDICATION, PROCEDURE, or OTHER.
For each entity, provide attributes like description, role, or any relevant details.
"""

# Few-shot examples
EXAMPLES = [
    ExampleData(
        text="John works at Microsoft in Seattle.",
        extractions=[
            Extraction(extraction_class="PERSON", extraction_text="John", description="An employee at Microsoft"),
            Extraction(extraction_class="ORGANIZATION", extraction_text="Microsoft", description="A technology company"),
            Extraction(extraction_class="LOCATION", extraction_text="Seattle", description="A city in Washington state"),
        ],
    ),
    ExampleData(
        text="Dr. Alice prescribed 50mg of Amoxicillin for the patient's severe infection at Mayo Clinic.",
        extractions=[
            Extraction(extraction_class="PERSON", extraction_text="Alice", description="A doctor", attributes={"role": "physician"}),
            Extraction(extraction_class="MEDICATION", extraction_text="Amoxicillin", description="An antibiotic", attributes={"dosage": "50mg"}),
            Extraction(extraction_class="MEDICAL_CONDITION", extraction_text="severe infection", description="The patient's illness"),
            Extraction(extraction_class="ORGANIZATION", extraction_text="Mayo Clinic", description="A medical facility"),
        ],
    ),
    ExampleData(
        text="The Apollo 11 mission successfully landed on the Moon in 1969.",
        extractions=[
            Extraction(extraction_class="EVENT", extraction_text="Apollo 11 mission", description="Spaceflight that landed first humans on the Moon"),
            Extraction(extraction_class="LOCATION", extraction_text="Moon", description="Earth's only natural satellite"),
        ],
    ),
]


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Simple character-based chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def extract_from_chunk(chunk: str, chunk_idx: int, model) -> Dict[str, Any]:
    """Extract entities from a single chunk using LangExtract."""
    try:
        result = lx.extract(
            text_or_documents=chunk,
            prompt_description=EXTRACT_PROMPT,
            examples=EXAMPLES,
            model=model,
        )

        entities = []
        relationships = []

        if result and result.extractions:
            for ext in result.extractions:
                entity = {
                    "name": ext.extraction_text,
                    "type": ext.extraction_class.upper() if ext.extraction_class else "OTHER",
                    "description": ext.description or "",
                    "attributes": ext.attributes or {},
                }
                entities.append(entity)

            # Infer relationships from attributes referencing other entities
            entity_names = {e["name"].upper(): e["name"] for e in entities}
            for ent in entities:
                attrs = ent.get("attributes", {})
                if isinstance(attrs, dict):
                    for key, val in attrs.items():
                        val_str = str(val).upper() if val else ""
                        if val_str in entity_names:
                            relationships.append({
                                "source_entity": ent["name"],
                                "target_entity": entity_names[val_str],
                                "relationship_type": key.upper(),
                                "description": f"{ent['name']} {key} {entity_names[val_str]}",
                            })

        return {
            "chunk_idx": chunk_idx,
            "entities": entities,
            "relationships": relationships,
        }

    except Exception as e:
        logger.error(f"Extraction failed for chunk {chunk_idx}: {e}")
        return {"chunk_idx": chunk_idx, "entities": [], "relationships": []}


def consolidate_graph(chunk_results: List[Dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all chunk results into unified nodes and edges for MS GraphRAG."""
    nodes_dict = {}
    edges_dict = {}

    for result in chunk_results:
        source_id = f"chunk_{result['chunk_idx']}"

        for ent in result["entities"]:
            ent_key = ent["name"].strip().upper()
            if not ent_key:
                continue
            if ent_key not in nodes_dict:
                nodes_dict[ent_key] = {
                    "id": str(uuid.uuid4()),
                    "title": ent["name"],
                    "type": ent["type"],
                    "description": ent["description"],
                    "source_id": source_id,
                    "degree": 0,
                }
            else:
                if ent["description"]:
                    nodes_dict[ent_key]["description"] += f" | {ent['description']}"
                nodes_dict[ent_key]["source_id"] += f",{source_id}"

        for rel in result["relationships"]:
            src_key = rel["source_entity"].strip().upper()
            tgt_key = rel["target_entity"].strip().upper()
            if not src_key or not tgt_key:
                continue
            pair_key = tuple(sorted([src_key, tgt_key]))
            edge_key = f"{pair_key[0]}_{pair_key[1]}_{rel['relationship_type']}"
            if edge_key not in edges_dict:
                edges_dict[edge_key] = {
                    "id": str(uuid.uuid4()),
                    "source": rel["source_entity"],
                    "target": rel["target_entity"],
                    "weight": 1.0,
                    "description": rel["description"],
                    "source_id": source_id,
                }
            else:
                edges_dict[edge_key]["weight"] += 1.0
                edges_dict[edge_key]["description"] += f" | {rel['description']}"
                edges_dict[edge_key]["source_id"] += f",{source_id}"

    # Calculate node degree
    for edge in edges_dict.values():
        src_key = edge["source"].strip().upper()
        tgt_key = edge["target"].strip().upper()
        if src_key in nodes_dict:
            nodes_dict[src_key]["degree"] += int(edge["weight"])
        if tgt_key in nodes_dict:
            nodes_dict[tgt_key]["degree"] += int(edge["weight"])

    nodes_df = pd.DataFrame(list(nodes_dict.values())) if nodes_dict else pd.DataFrame()
    edges_df = pd.DataFrame(list(edges_dict.values())) if edges_dict else pd.DataFrame()

    return nodes_df, edges_df


def main():
    logger.info(f"Starting Custom GraphRAG Pipeline [{PROVIDER}]")
    logger.info(f"Model: {MODEL_NAME} via {MODEL_URL}")

    # 1. Read input documents
    all_text = ""
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(INPUT_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"

    if not all_text.strip():
        logger.error(f"No text found in {INPUT_DIR}/. Add a .txt file and run again.")
        return

    # 2. Chunk Text
    chunks = chunk_text(all_text)
    logger.info(f"Generated {len(chunks)} text chunks.")

    # 3. Build model and extract
    model = build_model()

    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
        result = extract_from_chunk(chunk, i, model)
        entity_count = len(result["entities"])
        rel_count = len(result["relationships"])
        logger.info(f"  -> {entity_count} entities, {rel_count} relationships")
        chunk_results.append(result)

    total_entities = sum(len(r["entities"]) for r in chunk_results)
    logger.info(f"Extracted {total_entities} total entities across all chunks.")

    if total_entities == 0:
        logger.error("Failed to extract any entities.")
        return

    # 4. Consolidate and format for MS GraphRAG
    nodes_df, edges_df = consolidate_graph(chunk_results)
    logger.info(f"Consolidated Graph: {len(nodes_df)} Nodes, {len(edges_df)} Edges.")

    # 5. Export to Parquet (MS GraphRAG format)
    nodes_path = os.path.join(OUTPUT_DIR, "create_final_nodes.parquet")
    edges_path = os.path.join(OUTPUT_DIR, "create_final_relationships.parquet")
    nodes_df.to_parquet(nodes_path)
    edges_df.to_parquet(edges_path)

    # Save JSON copies for easy inspection
    nodes_df.to_json(os.path.join(OUTPUT_DIR, "nodes.json"), orient="records", indent=2)
    edges_df.to_json(os.path.join(OUTPUT_DIR, "edges.json"), orient="records", indent=2)

    logger.info(f"Exported Nodes and Relationships to {OUTPUT_DIR}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()

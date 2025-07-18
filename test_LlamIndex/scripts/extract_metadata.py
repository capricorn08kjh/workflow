from llama_index.core.extractors import LLMMetadataExtractor
from llama_index.core import Settings
import yaml

with open("config/prompts.yaml") as f:
    prompts = yaml.safe_load(f)

extractor = LLMMetadataExtractor(
    llm=Settings.llm,
    prompt_template=prompts["metadata_extraction"]
)
for node in nodes:
    metadata = extractor.extract([node])[0]
    node.metadata.update(metadata)
    with open(f"data/metadata/{node.metadata['doc_id']}.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

# convert_to_jsonl.py
import json
import ijson
import argparse

def json_to_jsonl(src_path, dst_path):
    with open(src_path, 'rb') as f, open(dst_path, 'w', encoding='utf-8') as out:
        # ijson.items()는 대용량 JSON 배열을 스트리밍 방식으로 파싱
        for obj in ijson.items(f, 'item'):
            json.dump(obj, out, ensure_ascii=False)
            out.write('\n')
    print(f"✅ Converted {src_path} → {dst_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert JSON array file → JSONL line file")
    ap.add_argument("--input", required=True, help="Input JSON file (array)")
    ap.add_argument("--output", required=True, help="Output JSONL file path")
    args = ap.parse_args()

    json_to_jsonl(args.input, args.output)
    
    
"""
python create_index.py \
  --data-dir ./data \
  --files-glob "*.jsonl" \
  --keys user.id items[].sku meta.device.os \
  --lmdb kv_index.lmdb \
  --manifest manifest.json
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import html
import json
import argparse

try:
    import pandas as pd  # optional, used if available
except Exception:
    pd = None  # fallback


def render_hier_table_html(
    data: Union[List[Dict[str, Any]], "pd.DataFrame"],
    id_column: str = "row_id",
    delimiter: str = "^",
    label_column_name: str = "항목",
    visible_columns: Optional[List[str]] = None,
    aggregate: Optional[Dict[str, str]] = None,
    initial_expand_level: int = 0,
) -> str:
    """
    Build a collapsible hierarchical HTML table (treegrid-like) from data.
    'data' is a list of dicts or a pandas DataFrame.
    'id_column' contains the hierarchical id like 'memory^dram^ddr5'.
    'aggregate' supports per-column {'col':'sum'|'avg'|'min'|'max'} for parents.
    """
    # Normalize data to list of dicts
    rows: List[Dict[str, Any]]
    if pd is not None and hasattr(data, "to_dict"):
        rows = data.to_dict(orient="records")
    else:
        rows = list(data)  # assume list[dict]

    if not rows:
        return "<p>표시할 데이터가 없습니다.</p>"

    aggregate = aggregate or {}

    # All columns discovery
    all_keys = set()
    for r in rows:
        for k in r.keys():
            all_keys.add(k)

    # Columns to show
    if visible_columns is None:
        cols = [k for k in all_keys if k != id_column]
    else:
        cols = [c for c in visible_columns if c != id_column]

    numeric_agg_cols = set(aggregate.keys())

    # Node structure
    class Node:
        __slots__ = ("id","key","label","parent_id","level","values","children","is_leaf")
        def __init__(self, id_, key, label, parent_id, level):
            self.id = id_
            self.key = key
            self.label = label
            self.parent_id = parent_id
            self.level = level
            self.values: Dict[str, Any] = {}
            self.children: set[str] = set()
            self.is_leaf = False

    def slug(s: str) -> str:
        # DOM-safe id
        return "".join(ch if ch.isalnum() or ch in "-_:.|" else "_" for ch in s)

    nodes: Dict[str, Node] = {}

    def ensure_node(path_list: List[str]) -> Node:
        key = "|".join(path_list)
        id_ = slug(key)
        if id_ not in nodes:
            label = str(path_list[-1]) if path_list else ""
            parent_id = slug("|".join(path_list[:-1])) if len(path_list) > 1 else None
            n = Node(id_=id_, key=key, label=label, parent_id=parent_id, level=len(path_list)-1)
            nodes[id_] = n
            if parent_id and parent_id in nodes:
                nodes[parent_id].children.add(id_)
        return nodes[id_]

    # Insert rows into leaf nodes
    for row in rows:
        raw = row.get(id_column, None)
        if raw is None:
            continue
        parts = [p.strip() for p in str(raw).split(delimiter) if str(p).strip() != ""]
        if not parts:
            continue
        # Create all prefixes
        for i in range(len(parts)):
            node = ensure_node(parts[:i+1])
            if i == len(parts) - 1:
                node.is_leaf = True
                # Store original rows under a temp bucket for merging
                lst = node.values.get("_rows")
                if lst is None:
                    node.values["_rows"] = [row]
                else:
                    lst.append(row)

    # Aggregate values bottom-up
    ordered_by_level = sorted(nodes.values(), key=lambda n: n.level)
    ids_in_order = [n.id for n in ordered_by_level]

    def to_num(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    for i in range(len(ids_in_order)-1, -1, -1):
        n = nodes[ids_in_order[i]]
        vals: Dict[str, Any] = {}

        # Leaf own values
        if n.is_leaf:
            # Merge duplicates: numeric agg cols -> sum; else take last
            for c in cols:
                arr = []
                for r in n.values.get("_rows", []):
                    if c in r and r[c] is not None:
                        arr.append(r[c])
                if not arr:
                    continue
                if c in numeric_agg_cols:
                    s = 0.0
                    for v in arr:
                        try:
                            num = float(v)
                        except Exception:
                            num = 0.0
                        if num == num:  # not NaN
                            s += num
                    vals[c] = s
                else:
                    vals[c] = arr[-1]

        # Merge children according to aggregate rules
        for c, rule in aggregate.items():
            child_vals = []
            for child_id in n.children:
                cv = nodes[child_id].values.get(c, None)
                if cv is not None:
                    child_vals.append(cv)
            if rule == "sum":
                total = 0.0
                for v in child_vals:
                    num = to_num(v)
                    if num == num:
                        total += num
                if c in vals:
                    own = to_num(vals[c])
                    if own == own:
                        total += own
                vals[c] = total
            elif rule == "avg":
                nums = []
                if c in vals:
                    own = to_num(vals[c])
                    if own == own:
                        nums.append(own)
                for v in child_vals:
                    num = to_num(v)
                    if num == num:
                        nums.append(num)
                if nums:
                    vals[c] = sum(nums) / len(nums)
            elif rule == "max":
                cand = []
                if c in vals:
                    own = to_num(vals[c])
                    if own == own:
                        cand.append(own)
                for v in child_vals:
                    num = to_num(v)
                    if num == num:
                        cand.append(num)
                if cand:
                    vals[c] = max(cand)
            elif rule == "min":
                cand = []
                if c in vals:
                    own = to_num(vals[c])
                    if own == own:
                        cand.append(own)
                for v in child_vals:
                    num = to_num(v)
                    if num == num:
                        cand.append(num)
                if cand:
                    vals[c] = min(cand)

        # Save merged
        n.values.update(vals)

    # Build traversal order (DFS with label sort)
    roots = [n for n in nodes.values() if n.parent_id is None]
    roots.sort(key=lambda x: x.label)

    ordered: List[Node] = []
    def visit(node: Node):
        ordered.append(node)
        children = [nodes[cid] for cid in node.children]
        children.sort(key=lambda x: x.label)
        for ch in children:
            visit(ch)
    for r in roots:
        visit(r)

    # HTML helpers
    def fmt_value(v):
        if v is None:
            return ""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            s = f"{v:,.0f}" if float(v).is_integer() else f"{v:,.3f}".rstrip("0").rstrip(".")
            return s
        try:
            f = float(v)
            s = f"{f:,.0f}" if f.is_integer() else f"{f:,.3f}".rstrip("0").rstrip(".")
            return s
        except Exception:
            return html.escape(str(v))

    def esc(s):
        return html.escape(str(s)) if s is not None else ""

    # Build HTML
    style = """
<style>
  table.htree { border-collapse: collapse; width: 100%; }
  .htree th, .htree td { border: 1px solid #ddd; padding: 8px; }
  .htree tr[hidden] { display: none; }
  .htree .toggle { border: none; background: none; cursor: pointer; font: inherit; padding: 0 4px; line-height: 1; }
  .htree td.indent { padding-left: calc(var(--level, 0) * 1.25rem + 8px); white-space: nowrap; }
  .htree .caret::before { content: "▶"; display: inline-block; transform: rotate(0deg); transition: transform .15s; }
  .htree .caret[aria-expanded="true"]::before { transform: rotate(90deg); }
  .htree thead th { position: sticky; top: 0; background: #fff; z-index: 1; }
</style>
"""
    thead = f"""
<thead>
  <tr>
    <th>{esc(label_column_name)}</th>
    {''.join(f'<th>{esc(c)}</th>' for c in cols)}
  </tr>
</thead>
"""
    rows_html = []
    for n in ordered:
        has_children = len(n.children) > 0
        btn = ('<button class="toggle caret" aria-expanded="false" title="펼치기/접기"></button>'
               if has_children else '<span style="display:inline-block;width:1.2em"></span>')
        cells = [f'<td class="indent" style="--level:{n.level}">{btn}{esc(n.label)}</td>']
        for c in cols:
            cells.append(f"<td>{fmt_value(n.values.get(c))}</td>")
        hidden_attr = "" if n.level <= initial_expand_level else " hidden"
        parent_attr = f' data-parent="{html.escape(n.parent_id)}"' if n.parent_id else ""
        aria_level = n.level + 1
        rows_html.append(
            f'<tr id="row-{html.escape(n.id)}" data-id="{html.escape(n.id)}"{parent_attr} '
            f'data-level="{n.level}" aria-level="{aria_level}" aria-expanded="false"{hidden_attr}>'
            + "".join(cells) + "</tr>"
        )

    tbody_html = "<tbody id='htree-body'>" + "".join(rows_html) + "</tbody>"

    script = """
<script>
(function(){
  const tbody = document.getElementById('htree-body');
  if (!tbody) return;
  function setChildrenVisible(parentId, visible) {
    const rows = tbody.querySelectorAll('tr[data-parent="'+CSS.escape(parentId)+'"]');
    rows.forEach(row => {
      row.hidden = !visible;
      if (!visible) {
        row.setAttribute('aria-expanded','false');
        const btn = row.querySelector('.toggle');
        if (btn) btn.setAttribute('aria-expanded','false');
        const id = row.getAttribute('data-id');
        setChildrenVisible(id, false);
      }
    });
  }
  tbody.addEventListener('click', (e) => {
    const btn = e.target.closest('.toggle');
    if (!btn) return;
    const tr = btn.closest('tr');
    const id = tr.getAttribute('data-id');
    const expanded = btn.getAttribute('aria-expanded') === 'true';
    const next = !expanded;
    btn.setAttribute('aria-expanded', String(next));
    tr.setAttribute('aria-expanded', String(next));
    setChildrenVisible(id, next);
  });
})();
</script>
"""
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
{style}
<table class="htree" role="treegrid" aria-label="계층형 표">
  {thead}
  {tbody_html}
</table>
{script}
"""
    return html_doc


def _read_input(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pandas as pd  # try again inside
        if path.suffix.lower() in {".csv"}:
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() in {".json"}:
            try:
                return pd.read_json(path)
            except Exception:
                return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    # Fallback without pandas
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        import csv
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    raise ValueError("지원하지 않는 입력 형식입니다. CSV/JSON(.xlsx는 pandas 필요)")


def _parse_agg(expr: Optional[str]) -> Dict[str, str]:
    if not expr:
        return {}
    out: Dict[str, str] = {}
    for part in expr.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().lower()
    return out


def save_hier_table_html(
    data,
    output_path: Union[str, Path],
    **kwargs
) -> Path:
    html_doc = render_hier_table_html(data, **kwargs)
    output_path = Path(output_path)
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path


def main(argv=None):
    p = argparse.ArgumentParser(description="계층형 접힘 HTML 테이블 생성기")
    p.add_argument("--input", "-i", required=True, help="입력 파일 경로 (CSV/JSON/Excel)")
    p.add_argument("--out", "-o", required=True, help="생성될 HTML 파일 경로")
    p.add_argument("--id-column", default="row_id", help="계층 ID 컬럼명 (default: row_id)")
    p.add_argument("--delimiter", default="^", help="계층 구분자 (default: ^)")
    p.add_argument("--label", default="항목", help="첫 컬럼 제목 (default: 항목)")
    p.add_argument("--columns", nargs="*", help="표시할 컬럼 목록 (공백으로 구분)")
    p.add_argument("--aggregate", help='집계 규칙 예: "예상실적=sum,실제실적=sum"')
    p.add_argument("--expand", type=int, default=0, help="초기 펼침 레벨 (0=루트만)")
    args = p.parse_args(argv)

    data = _read_input(Path(args.input))
    agg = _parse_agg(args.aggregate)

    html_doc = render_hier_table_html(
        data,
        id_column=args.id_column,
        delimiter=args.delimiter,
        label_column_name=args.label,
        visible_columns=args.columns,
        aggregate=agg,
        initial_expand_level=args.expand,
    )
    Path(args.out).write_text(html_doc, encoding="utf-8")
    print(f"Saved HTML to {args.out}")


if __name__ == "__main__":
    main()

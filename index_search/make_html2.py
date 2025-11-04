
from typing import List, Dict, Any, Optional, Union, Iterable
from pathlib import Path
import html
import json
import argparse
import re

try:
    import pandas as pd  # optional
except Exception:
    pd = None


def _normalize_columns(columns: Optional[Union[str, Iterable[str]]]) -> Optional[List[str]]:
    if columns is None:
        return None
    if isinstance(columns, str):
        parts = [p.strip() for p in re.split(r'[,\s]+', columns) if p.strip()]
        return parts if parts else None
    try:
        return [str(c).strip() for c in columns]
    except TypeError:
        s = str(columns)
        parts = [p.strip() for p in re.split(r'[,\s]+', s) if p.strip()]
        return parts if parts else None


def render_hier_table_html(
    data: Union[List[Dict[str, Any]], "pd.DataFrame"],
    id_column: str = "row_id",
    delimiter: str = "^",
    label_column_name: str = "항목",                 # hierarchy column header
    visible_columns: Optional[Union[str, Iterable[str]]] = None,  # columns shown (besides id_column)
    aggregate: Optional[Dict[str, str]] = None,
    initial_expand_level: int = 0,
    text_columns: Optional[Union[str, Iterable[str]]] = None,     # columns to force as text
    leading_columns: Optional[Union[str, Iterable[str]]] = None,  # columns to render BEFORE hierarchy
) -> str:
    # Normalize to list of dicts
    if pd is not None and hasattr(data, "to_dict"):
        rows = data.to_dict(orient="records")
    else:
        rows = list(data)

    if not rows:
        return "<p>표시할 데이터가 없습니다.</p>"

    aggregate = aggregate or {}

    # Discover columns
    all_keys = set()
    for r in rows:
        for k in r.keys():
            all_keys.add(str(k))

    norm_visible = _normalize_columns(visible_columns)
    if norm_visible is None:
        cols_all = [k for k in all_keys if k != id_column]
    else:
        cols_all = [c for c in norm_visible if c != id_column]

    # Split into leading + trailing columns (relative order preserved)
    leading = _normalize_columns(leading_columns) or []
    leading = [c for c in leading if c in cols_all]
    trailing = [c for c in cols_all if c not in leading]

    numeric_agg_cols = set(aggregate.keys())
    text_cols = set(_normalize_columns(text_columns) or [])

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

    # Insert data
    for row in rows:
        raw = row.get(id_column, None)
        if raw is None:
            continue
        parts = [p.strip() for p in str(raw).split(delimiter) if str(p).strip() != ""]
        if not parts:
            continue
        for i in range(len(parts)):
            node = ensure_node(parts[:i+1])
            if i == len(parts) - 1:
                node.is_leaf = True
                node.values.setdefault("_rows", []).append(row)

    # Aggregate bottom-up
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

        if n.is_leaf:
            for c in cols_all:
                arr = [r[c] for r in n.values.get("_rows", []) if c in r and r[c] is not None]
                if not arr:
                    continue
                if c in numeric_agg_cols:
                    s = 0.0
                    for v in arr:
                        try:
                            num = float(v)
                        except Exception:
                            num = 0.0
                        if num == num:
                            s += num
                    vals[c] = s
                else:
                    vals[c] = arr[-1]

        for c, rule in aggregate.items():
            child_vals = [nodes[child_id].values.get(c) for child_id in n.children if nodes[child_id].values.get(c) is not None]
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

        n.values.update(vals)

    # Order traversal
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
    # Header: leading -> hierarchy label -> trailing
    thead = "<thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in leading) \
            + f"<th>{esc(label_column_name)}</th>" \
            + "".join(f"<th>{esc(c)}</th>" for c in trailing) + "</tr></thead>"

    rows_html = []
    for n in ordered:
        has_children = len(n.children) > 0
        btn = ('<button class="toggle caret" aria-expanded="false" title="펼치기/접기"></button>'
               if has_children else '<span style="display:inline-block;width:1.2em"></span>')
        # leading cells
        lead_cells = []
        for c in leading:
            val = n.values.get(c)
            if c in text_cols:
                cell = "" if val is None else html.escape(str(val))
            else:
                cell = fmt_value(val)
            lead_cells.append(f"<td>{cell}</td>")
        # hierarchy label cell
        label_cell = f'<td class="indent" style="--level:{n.level}">{btn}{esc(n.label)}</td>'
        # trailing cells
        trail_cells = []
        for c in trailing:
            val = n.values.get(c)
            if c in text_cols:
                cell = "" if val is None else html.escape(str(val))
            else:
                cell = fmt_value(val)
            trail_cells.append(f"<td>{cell}</td>")

        hidden_attr = "" if n.level <= initial_expand_level else " hidden"
        parent_attr = f' data-parent="{html.escape(n.parent_id)}"' if n.parent_id else ""
        aria_level = n.level + 1

        rows_html.append(
            f'<tr id="row-{html.escape(n.id)}" data-id="{html.escape(n.id)}"{parent_attr} '
            f'data-level="{n.level}" aria-level="{aria_level}" aria-expanded="false"{hidden_attr}>'
            + "".join(lead_cells) + label_cell + "".join(trail_cells) + "</tr>"
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

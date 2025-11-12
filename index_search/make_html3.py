from typing import List, Dict, Any, Optional, Union, Iterable, Callable
import html, re

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
    data: List[Dict[str, Any]],
    id_column: str = "row_id",
    delimiter: str = "^",
    label_column_name: str = "항목",                       # 계층(트리) 컬럼 헤더
    visible_columns: Optional[Union[str, Iterable[str]]] = None,  # 렌더링할 데이터 컬럼
    leading_columns: Optional[Union[str, Iterable[str]]] = None,  # 트리 앞(맨 왼쪽) 컬럼들
    text_columns: Optional[Union[str, Iterable[str]]] = None,     # 숫자 포맷 방지할 컬럼들
    aggregate: Optional[Dict[str, str]] = None,                   # 부모행 집계 규칙
    initial_expand_level: int = 0,                                # 초기 펼침 레벨
    header_groups: Optional[Dict[str, List[str]]] = None,         # 2단 헤더: {'실적':['예상실적','실제실적']}
    alias_map: Optional[Dict[str, str]] = None,                   # 헤더 표시명 매핑
    label_format_hook: Optional[Callable[[str, int, List[str]], str]] = None,  # 계층 라벨 포매터
) -> str:
    rows = list(data)
    if not rows:
        return "<p>표시할 데이터가 없습니다.</p>"

    aggregate = aggregate or {}
    alias_map = alias_map or {}
    header_groups = header_groups or {}

    # 컬럼 수집
    all_keys = set()
    for r in rows:
        for k in r.keys():
            all_keys.add(str(k))

    norm_visible = _normalize_columns(visible_columns)
    if norm_visible is None:
        cols_all = [k for k in all_keys if k != id_column]
    else:
        cols_all = [c for c in norm_visible if c != id_column]

    leading = _normalize_columns(leading_columns) or []
    leading = [c for c in leading if c in cols_all]
    trailing = [c for c in cols_all if c not in leading]

    text_cols = set(_normalize_columns(text_columns) or [])
    numeric_agg_cols = set(aggregate.keys())

    # 노드 구조
    class Node:
        __slots__ = ("id","key","label","parent_id","level","values","children","is_leaf","path")
        def __init__(self, id_, key, label, parent_id, level, path):
            self.id = id_
            self.key = key
            self.label = label
            self.parent_id = parent_id
            self.level = level
            self.values: Dict[str, Any] = {}
            self.children: set[str] = set()
            self.is_leaf = False
            self.path = path

    def slug(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_:.|" else "_" for ch in s)

    nodes: Dict[str, Node] = {}

    def ensure_node(path_list: List[str]) -> Node:
        key = "|".join(path_list)
        id_ = slug(key)
        if id_ not in nodes:
            raw_label = str(path_list[-1]) if path_list else ""
            label = raw_label
            level = len(path_list) - 1
            if label_format_hook is not None:
                try:
                    label = label_format_hook(raw_label, level, path_list[:])
                except Exception:
                    label = raw_label
            parent_id = slug("|".join(path_list[:-1])) if len(path_list) > 1 else None
            n = Node(id_=id_, key=key, label=label, parent_id=parent_id, level=level, path=path_list[:])
            nodes[id_] = n
            if parent_id and parent_id in nodes:
                nodes[parent_id].children.add(id_)
        return nodes[id_]

    # 데이터 삽입
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

    # bottom-up 집계
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

    # 순회 순서
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

    def esc(s): return html.escape(str(s)) if s is not None else ""
    def alias(c): return alias_map.get(c, c)

    # thead (2단 헤더 지원)
    group_by_col = {}
    for gname, cols in (header_groups or {}).items():
        for c in cols:
            group_by_col[c] = gname

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
    have_groups = any(col in group_by_col for col in (trailing or []))
    if have_groups:
        top_cells = []
        for c in leading:
            top_cells.append(f'<th rowspan="2">{esc(alias(c))}</th>')
        top_cells.append(f'<th rowspan="2">{esc(label_column_name)}</th>')
        i = 0
        while i < len(trailing):
            col = trailing[i]
            g = group_by_col.get(col)
            if not g:
                top_cells.append(f'<th rowspan="2">{esc(alias(col))}</th>')
                i += 1
            else:
                j = i; members = []
                while j < len(trailing) and group_by_col.get(trailing[j]) == g:
                    members.append(trailing[j]); j += 1
                top_cells.append(f'<th colspan="{len(members)}">{esc(g)}</th>')
                i = j
        bottom_cells = []
        i = 0
        while i < len(trailing):
            col = trailing[i]; g = group_by_col.get(col)
            if not g: i += 1; continue
            j = i
            while j < len(trailing) and group_by_col.get(trailing[j]) == g:
                bottom_cells.append(f'<th>{esc(alias(trailing[j]))}</th>'); j += 1
            i = j
        thead = "<thead>" + f"<tr>{''.join(top_cells)}</tr>" + f"<tr>{''.join(bottom_cells)}</tr>" + "</thead>"
    else:
        thead = "<thead><tr>" + "".join(f"<th>{esc(alias(c))}</th>" for c in leading) \
                + f"<th>{esc(label_column_name)}</th>" \
                + "".join(f"<th>{esc(alias(c))}</th>" for c in trailing) + "</tr></thead>"

    # tbody
    rows_html = []
    for n in ordered:
        has_children = len(n.children) > 0
        btn = ('<button class="toggle caret" aria-expanded="false" title="펼치기/접기"></button>'
               if has_children else '<span style="display:inline-block;width:1.2em"></span>')
        lead_cells = []
        for c in leading:
            val = n.values.get(c)
            cell = "" if val is None else (esc(val) if c in text_cols else fmt_value(val))
            lead_cells.append(f"<td>{cell}</td>")
        label_cell = f'<td class="indent" style="--level:{n.level}">{btn}{esc(n.label)}</td>'
        trail_cells = []
        for c in trailing:
            val = n.values.get(c)
            cell = "" if val is None else (esc(val) if c in text_cols else fmt_value(val))
            trail_cells.append(f"<td>{cell}</td>")
        hidden_attr = "" if n.level <= initial_expand_level else " hidden"
        parent_attr = f' data-parent="{html.escape(n.parent_id)}"' if n.parent_id else ""
        aria_level = n.level + 1
        rows_html.append(
            f'<tr id="row-{html.escape(n.id)}" data-id="{html.escape(n.id)}"{parent_attr} '
            f'data-level="{n.level}" aria-level="{aria_level}" aria-expanded="false"{hidden_attr}>'
            + "".join(lead_cells) + label_cell + "".join(trail_cells) + "</tr>"
        )

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
    tbody_html = "<tbody id='htree-body'>" + "".join(rows_html) + "</tbody>"
    html_doc = f"<!doctype html>\n<meta charset='utf-8'>\n{style}\n<table class='htree' role='treegrid' aria-label='계층형 표'>\n{thead}\n{tbody_html}\n</table>\n{script}\n"
    return html_doc

# ---------------- Example according to your spec ----------------
def right_of_underscore(s: str) -> str:
    return str(s).rsplit("_", 1)[-1]

# 예시 데이터
demo_rows = [
    {"yyyymm": "202501", "DATA_LABEL": "32_graphic", "예상실적": 120, "실제실적": 110},
    {"yyyymm": "202501", "DATA_LABEL": "64_graphic", "예상실적":  80, "실제실적":  75},
    {"yyyymm": "202502", "DATA_LABEL": "16_compute", "예상실적":  60, "실제실적":  55},
    {"yyyymm": "202502", "DATA_LABEL": "compute",     "예상실적":  20, "실제실적":  18},
]

# 계층키는 DATA_LABEL을 그대로 쓰되, 화면표시는 '_' 오른쪽만
for r in demo_rows:
    r["row_id"] = r["DATA_LABEL"]

alias = {"yyyymm": "월", "예상실적": "예상 실적", "실제실적": "실제 실적"}

html_doc = render_hier_table_html(
    demo_rows,
    id_column="row_id",
    label_column_name="구분(DATA_LABEL)",
    visible_columns=["yyyymm","예상실적","실제실적"],
    leading_columns=["yyyymm"],
    text_columns=["yyyymm"],
    aggregate={"예상실적":"sum","실제실적":"sum"},
    initial_expand_level=0,
    header_groups={"실적": ["예상실적","실제실적"]},
    alias_map=alias,
    label_format_hook=lambda label, level, path: right_of_underscore(label)
)

# 파일 저장
out_path = Path("/mnt/data/hier_table_grouped_demo.html")
out_path.write_text(html_doc, encoding="utf-8")
str(out_path)

완료! 요청하신 스펙 전부 반영해서 **재작성**했고, 동작 데모 HTML도 만들어뒀어요.

- 데모 HTML: [hier_table_grouped_demo.html](sandbox:/mnt/data/hier_table_grouped_demo.html?_chatgptios_conversationID=6909551a-6d60-8324-92c6-52a6db93ad19&_chatgptios_messageID=02aab038-1cac-480e-a4c9-753736251e69)

# 핵심 기능 들어간 파이썬 함수
- **leading_columns**: 트리(계층) 컬럼 **앞(맨 왼쪽)**에 올 컬럼 지정 → `yyyymm` 왼쪽 고정
- **text_columns**: `yyyymm` 같은 문자열을 숫자처럼 포맷하지 않음
- **header_groups**: 2단 헤더(예: 상단 “실적” / 하단 “예상 실적”, “실제 실적”)
- **alias_map**: 데이터프레임 컬럼명 ↔ HTML 헤더명 매핑
- **label_format_hook**: 계층 라벨 가공(예: `DATA_LABEL`의 `_` 오른쪽만 표시)

아래 코드를 그대로 붙여 넣어 사용하세요:

```python
from typing import List, Dict, Any, Optional, Union, Iterable, Callable
import html, re

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
    data: List[Dict[str, Any]],
    id_column: str = "row_id",
    delimiter: str = "^",
    label_column_name: str = "항목",                       # 계층(트리) 컬럼 헤더
    visible_columns: Optional[Union[str, Iterable[str]]] = None,  # 렌더링할 데이터 컬럼
    leading_columns: Optional[Union[str, Iterable[str]]] = None,  # 트리 앞(맨 왼쪽) 컬럼들
    text_columns: Optional[Union[str, Iterable[str]]] = None,     # 숫자 포맷 방지할 컬럼들
    aggregate: Optional[Dict[str, str]] = None,                   # 부모행 집계 규칙
    initial_expand_level: int = 0,                                # 초기 펼침 레벨
    header_groups: Optional[Dict[str, List[str]]] = None,         # 2단 헤더: {'실적':['예상실적','실제실적']}
    alias_map: Optional[Dict[str, str]] = None,                   # 헤더 표시명 매핑
    label_format_hook: Optional[Callable[[str, int, List[str]], str]] = None,  # 계층 라벨 포매터
) -> str:
    rows = list(data)
    if not rows:
        return "<p>표시할 데이터가 없습니다.</p>"

    aggregate = aggregate or {}
    alias_map = alias_map or {}
    header_groups = header_groups or {}

    # 컬럼 수집
    all_keys = set()
    for r in rows:
        for k in r.keys():
            all_keys.add(str(k))

    norm_visible = _normalize_columns(visible_columns)
    if norm_visible is None:
        cols_all = [k for k in all_keys if k != id_column]
    else:
        cols_all = [c for c in norm_visible if c != id_column]

    leading = _normalize_columns(leading_columns) or []
    leading = [c for c in leading if c in cols_all]
    trailing = [c for c in cols_all if c not in leading]

    text_cols = set(_normalize_columns(text_columns) or [])
    numeric_agg_cols = set(aggregate.keys())

    # 노드 구조
    class Node:
        __slots__ = ("id","key","label","parent_id","level","values","children","is_leaf","path")
        def __init__(self, id_, key, label, parent_id, level, path):
            self.id = id_
            self.key = key
            self.label = label
            self.parent_id = parent_id
            self.level = level
            self.values: Dict[str, Any] = {}
            self.children: set[str] = set()
            self.is_leaf = False
            self.path = path

    def slug(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_:.|" else "_" for ch in s)

    nodes: Dict[str, Node] = {}

    def ensure_node(path_list: List[str]) -> Node:
        key = "|".join(path_list)
        id_ = slug(key)
        if id_ not in nodes:
            raw_label = str(path_list[-1]) if path_list else ""
            label = raw_label
            level = len(path_list) - 1
            if label_format_hook is not None:
                try:
                    label = label_format_hook(raw_label, level, path_list[:])
                except Exception:
                    label = raw_label
            parent_id = slug("|".join(path_list[:-1])) if len(path_list) > 1 else None
            n = Node(id_=id_, key=key, label=label, parent_id=parent_id, level=level, path=path_list[:])
            nodes[id_] = n
            if parent_id and parent_id in nodes:
                nodes[parent_id].children.add(id_)
        return nodes[id_]

    # 데이터 삽입
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

    # bottom-up 집계
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

    # 순회 순서
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

    def esc(s): return html.escape(str(s)) if s is not None else ""
    def alias(c): return (alias_map or {}).get(c, c)

    # thead (2단 헤더 지원)
    group_by_col = {}
    for gname, cols in (header_groups or {}).items():
        for c in cols:
            group_by_col[c] = gname

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
    have_groups = any(col in group_by_col for col in (trailing or []))
    if have_groups:
        top_cells = []
        for c in leading:
            top_cells.append(f'<th rowspan="2">{esc(alias(c))}</th>')
        top_cells.append(f'<th rowspan="2">{esc(label_column_name)}</th>')
        i = 0
        while i < len(trailing):
            col = trailing[i]
            g = group_by_col.get(col)
            if not g:
                top_cells.append(f'<th rowspan="2'>{esc(alias(col))}</th>')
                i += 1
            else:
                j = i; members = []
                while j < len(trailing) and group_by_col.get(trailing[j]) == g:
                    members.append(trailing[j]); j += 1
                top_cells.append(f'<th colspan="{len(members)}">{esc(g)}</th>')
                i = j
        bottom_cells = []
        i = 0
        while i < len(trailing):
            col = trailing[i]; g = group_by_col.get(col)
            if not g: i += 1; continue
            j = i
            while j < len(trailing) and group_by_col.get(trailing[j]) == g:
                bottom_cells.append(f'<th>{esc(alias(trailing[j]))}</th>'); j += 1
            i = j
        thead = "<thead>" + f"<tr>{''.join(top_cells)}</tr>" + f"<tr>{''.join(bottom_cells)}</tr>" + "</thead>"
    else:
        thead = "<thead><tr>" + "".join(f"<th>{esc(alias(c))}</th>" for c in leading) \
                + f"<th>{esc(label_column_name)}</th>" \
                + "".join(f"<th>{esc(alias(c))}</th>" for c in trailing) + "</tr></thead>"

    # tbody
    rows_html = []
    for n in ordered:
        has_children = len(n.children) > 0
        btn = ('<button class="toggle caret" aria-expanded="false" title="펼치기/접기"></button>'
               if has_children else '<span style="display:inline-block;width:1.2em"></span>')
        lead_cells = []
        for c in leading:
            val = n.values.get(c)
            cell = "" if val is None else (esc(val) if c in text_cols else fmt_value(val))
            lead_cells.append(f"<td>{cell}</td>")
        label_cell = f'<td class="indent" style="--level:{n.level}">{btn}{esc(n.label)}</td>'
        trail_cells = []
        for c in trailing:
            val = n.values.get(c)
            cell = "" if val is None else (esc(val) if c in text_cols else fmt_value(val))
            trail_cells.append(f"<td>{cell}</td>")
        hidden_attr = "" if n.level <= initial_expand_level else " hidden"
        parent_attr = f' data-parent="{html.escape(n.parent_id)}"' if n.parent_id else ""
        aria_level = n.level + 1
        rows_html.append(
            f'<tr id="row-{html.escape(n.id)}" data-id="{html.escape(n.id)}"{parent_attr} '
            f'data-level="{n.level}" aria-level="{aria_level}" aria-expanded="false"{hidden_attr}>'
            + "".join(lead_cells) + label_cell + "".join(trail_cells) + "</tr>"
        )

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
    tbody_html = "<tbody id='htree-body'>" + "".join(rows_html) + "</tbody>"
    html_doc = f"<!doctype html>\n<meta charset='utf-8'>\n{style}\n<table class='htree' role='treegrid' aria-label='계층형 표'>\n{thead}\n{tbody_html}\n</table>\n{script}\n"
    return html_doc
"""Streamlit app: 评论分析 / 问答助手 / 评测结果.

Run (Windows):  .venv\\Scripts\\streamlit run app.py

Without an API key the app runs in DEMO MODE: the LLM client only answers from
the on-disk cache (outputs/cache/llm), so the sample analysis, the preset
questions and the saved evaluation results all work for anyone cloning the repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.agent import run_agent
from src.aggregation import to_frame
from src.config import EVAL_RESULTS_DIR, ROOT, TAXONOMY, get_settings, has_api_key
from src.llm_client import DemoCacheMiss, LLMClient
from src.prompts import PROMPTS
from src.tools import LATEST_PROMPT, ToolBox

st.set_page_config(page_title="个护美妆评论洞察 Agent", page_icon="🧴", layout="wide")

SENT_ZH = {"positive": "正向", "neutral": "中性", "negative": "负向"}
SENT_COLORS = {"正向": "#2a78d6", "中性": "#8c8b85", "负向": "#eb6834"}  # fixed, by entity
PRESET_QUESTIONS = json.loads((ROOT / "scripts" / "demo_questions.json").read_text(encoding="utf-8"))

DEMO = not has_api_key()


@st.cache_resource
def get_client() -> LLMClient:
    return LLMClient(get_settings(), demo_mode=DEMO)


def sentiment_color_scale() -> alt.Scale:
    return alt.Scale(domain=list(SENT_COLORS), range=list(SENT_COLORS.values()))


# ---------------------------------------------------------------- sidebar --
st.sidebar.title("🧴 评论洞察 Agent")
if DEMO:
    st.sidebar.warning("演示模式：未检测到 LLM_API_KEY，只能读取缓存与已保存的结果。")
else:
    st.sidebar.success(f"在线模式：{get_settings().model}")
page = st.sidebar.radio("页面", ["评论分析", "问答助手", "评测结果"])
version = st.sidebar.selectbox("分类 prompt 版本", sorted(PROMPTS), index=sorted(PROMPTS).index(LATEST_PROMPT))


# ---------------------------------------------------------- page 1 --------
def page_analysis() -> None:
    st.header("评论分析")
    st.caption("流程 = Agent 用的同一套工具：load_reviews → classify_reviews → aggregate_stats → write_report")
    src = st.radio("数据来源", ["示例数据（固定评测集 300 条）", "上传 CSV"], horizontal=True)
    uploaded = None
    if src == "上传 CSV":
        f = st.file_uploader("CSV 需包含 review 列（可选 review_id、label）", type="csv")
        if f is None:
            return
        uploaded = pd.read_csv(f, encoding="utf-8-sig")
        if DEMO:
            st.info("演示模式下无法对新上传的数据调用大模型；请在 .env 中配置 LLM_API_KEY。")
    if not st.button("开始分析", type="primary"):
        if "analysis" not in st.session_state:
            return
    else:
        tb = ToolBox(get_client(), prompt_version=version, uploaded_df=uploaded)
        try:
            with st.spinner("加载与分类中…（相同输入会命中缓存，不重复花钱）"):
                tb.load_reviews("upload" if uploaded is not None else "sample")
                tb.classify_reviews()
                tb.aggregate_stats()
                tb.write_report()
        except DemoCacheMiss as e:
            st.error(str(e))
            return
        except Exception as e:  # show the error, do not crash the page
            st.error(f"分析失败：{e}")
            return
        st.session_state["analysis"] = tb

    tb: ToolBox = st.session_state["analysis"]
    stats = tb.stats
    df = to_frame(tb.reviews, tb.labels)

    c = st.columns(4)
    c[0].metric("评论数", stats["n_total"])
    c[1].metric("分类失败", stats["n_failed"])
    c[2].metric("负向占比", f"{stats['sentiment_shares']['negative']:.1%}")
    c[3].metric("中性占比", f"{stats['sentiment_shares']['neutral']:.1%}")
    if tb.source == "sample":
        st.caption("注意：示例数据是正负各 150 条的平衡评测集，情感占比不代表真实好评率。")

    left, right = st.columns([1, 2])
    sd = pd.DataFrame([{"情感": SENT_ZH[k], "数量": v} for k, v in stats["sentiment_counts"].items()])
    left.subheader("情感分布")
    left.altair_chart(alt.Chart(sd).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=36).encode(
        x=alt.X("情感:N", sort=list(SENT_COLORS), title=None),
        y=alt.Y("数量:Q", title="评论数"),
        color=alt.Color("情感:N", scale=sentiment_color_scale(), legend=None),
        tooltip=["情感", "数量"]).properties(height=300), width="stretch")

    rows = [{"标签": t, "情感": SENT_ZH[s], "数量": n}
            for t, d in stats["tag_by_sentiment"].items() for s, n in d.items() if n]
    td = pd.DataFrame(rows)
    right.subheader("问题标签 × 情感")
    if not td.empty:
        order = sorted(TAXONOMY, key=lambda t: -stats["tag_counts"][t])
        right.altair_chart(alt.Chart(td).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                                                   stroke="#fcfcfb", strokeWidth=1).encode(
            y=alt.Y("标签:N", sort=order, title=None),
            x=alt.X("数量:Q", title="提及次数（一条评论最多 2 个标签）"),
            color=alt.Color("情感:N", scale=sentiment_color_scale(), legend=alt.Legend(orient="top")),
            order=alt.Order("情感:N"),
            tooltip=["标签", "情感", "数量"]).properties(height=340), width="stretch")

    st.subheader("负向评论 Top 问题与原话")
    for x in stats["negative_tags"][:5]:
        with st.expander(f"{x['tag']}：{x['count']} 条，占负向 {x['share_of_negative']:.1%}"):
            for q in x["quotes"]:
                st.markdown(f"- “{q['text']}”  `#{q['review_id']}`")

    st.subheader("逐条结果")
    show = df.assign(情感=df["sentiment"].map(SENT_ZH), 标签=df["tags"].map("、".join))
    cols = ["review_id", "review", "情感", "标签"] + (["label"] if "label" in show and show["label"].notna().any() else [])
    st.dataframe(show[cols], width="stretch", height=300)

    st.subheader("改进报告")
    st.download_button("下载报告（Markdown）", tb.report_md, file_name=f"report_{tb.source}_{tb.prompt_version}.md")
    with st.container(border=True):
        st.markdown(tb.report_md)


# ---------------------------------------------------------- page 2 --------
def render_result(res) -> None:
    st.markdown(res.answer)
    if res.kb_no_answer:
        st.warning("知识库检索无结果：Agent 应如实说明知识库没有答案。")
    if res.retrieved_sources:
        st.caption("检索到的知识库来源：" + "、".join(res.retrieved_sources) +
                   "　｜　回答中引用：" + ("、".join(res.cited_sources) or "无"))
    with st.expander(f"🔧 推理过程：调用了 {len(res.steps)} 次工具"):
        if not res.steps:
            st.write("没有调用工具（直接回答）。")
        for i, s in enumerate(res.steps, 1):
            st.markdown(f"**{i}. `{s.tool}`**  参数：`{json.dumps(s.arguments, ensure_ascii=False)}`")
            r = dict(s.result)
            if "report_markdown" in r:
                r["report_markdown"] = r["report_markdown"][:300] + "…"
            st.json(r, expanded=False)


def page_chat() -> None:
    st.header("问答助手")
    st.caption("手写的 tool-calling 循环（无 Agent 框架）。每个问题独立运行（不带多轮记忆），方便复现与缓存。"
               + (" 演示模式下只能回答下面的预置问题（结果来自缓存）。" if DEMO else ""))
    st.session_state.setdefault("chat", [])
    cols = st.columns(len(PRESET_QUESTIONS))
    clicked = None
    for c, q in zip(cols, PRESET_QUESTIONS):
        if c.button(q, width="stretch"):
            clicked = q
    typed = st.chat_input("问点什么，例如：差评主要集中在哪？硅油有什么作用？")
    question = clicked or typed

    for role, content in st.session_state["chat"]:
        with st.chat_message(role):
            render_result(content) if role == "assistant" else st.markdown(content)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Agent 思考并调用工具中…"):
                    res = run_agent(question, ToolBox(get_client(), prompt_version=LATEST_PROMPT), get_client())
            except DemoCacheMiss as e:
                st.info(f"{e} 演示模式只支持预置问题；配置 API Key 后可自由提问。")
                return
            render_result(res)
        st.session_state["chat"] += [("user", question), ("assistant", res)]


# ---------------------------------------------------------- page 3 --------
def page_eval() -> None:
    st.header("评测结果")
    comp = EVAL_RESULTS_DIR / "comparison.csv"
    if not comp.exists():
        st.info("还没有评测结果。运行：.venv\\Scripts\\python scripts\\evaluate.py --version v1 --stability")
        return
    st.caption("所有数字均读取自 outputs/eval_results/，由 scripts/evaluate.py 实际运行生成。"
               "评测集：300 条（正/负各 150，random_state=42），人工标签为二分类，"
               "因此预测为“中性”一律算错，并单独报告中性率。")
    df = pd.read_csv(comp, encoding="utf-8-sig")
    pretty = df.rename(columns={"version": "版本", "accuracy": "准确率", "macro_f1": "Macro-F1",
                                "neutral_rate": "中性率", "json_failure_rate": "JSON失败率",
                                "accuracy_on_non_neutral": "非中性样本准确率",
                                "stability_identical": "稳定性(两次完全一致)", "api_cost": "单次评测花费"})
    st.dataframe(pretty, width="stretch", hide_index=True)
    ho = EVAL_RESULTS_DIR / "holdout" / "comparison.csv"
    if ho.exists():
        st.markdown("**泛化检查（holdout 集）**：另外 300 条、与评测集不重叠、写 prompt 时从未查看过的评论。"
                    "v2 是看着评测集 badcase 写的，所以 holdout 上的提升才是更可信的数字。")
        hd = pd.read_csv(ho, encoding="utf-8-sig").drop(columns=["stability_identical"])
        st.dataframe(hd.rename(columns={"version": "版本", "accuracy": "准确率", "macro_f1": "Macro-F1",
                                        "neutral_rate": "中性率", "json_failure_rate": "JSON失败率",
                                        "accuracy_on_non_neutral": "非中性样本准确率", "api_cost": "单次评测花费"}),
                     width="stretch", hide_index=True)

    versions = df["version"].tolist()
    cols = st.columns(len(versions))
    for c, v in zip(cols, versions):
        m = json.loads((EVAL_RESULTS_DIR / v / "metrics.json").read_text(encoding="utf-8"))
        cm = m["confusion_matrix"]
        cells = [{"真实": SENT_ZH[t], "预测": {"failed": "失败", **SENT_ZH}[p], "数量": cm["values"][i][j]}
                 for i, t in enumerate(cm["rows_true"]) for j, p in enumerate(cm["cols_pred"])]
        cd = pd.DataFrame(cells)
        base = alt.Chart(cd).encode(x=alt.X("预测:N", sort=["正向", "中性", "负向", "失败"]),
                                    y=alt.Y("真实:N", sort=["正向", "负向"]))
        heat = base.mark_rect(stroke="#fcfcfb", strokeWidth=2).encode(
            color=alt.Color("数量:Q", scale=alt.Scale(range=["#f0efec", "#2a78d6"]), legend=None),
            tooltip=["真实", "预测", "数量"])
        text = base.mark_text(fontSize=16).encode(
            text="数量:Q", color=alt.condition(alt.datum["数量"] > 80, alt.value("white"), alt.value("#0b0b0b")))
        c.subheader(f"{v} 混淆矩阵")
        c.altair_chart((heat + text).properties(height=200), width="stretch")
        if m.get("stability"):
            s = m["stability"]
            c.caption(f"稳定性（关缓存跑两次）：情感+标签完全一致 {s['identical_sentiment_and_tags']:.1%}；"
                      f"仅情感 {s['identical_sentiment']:.1%}；仅标签 {s['identical_tags']:.1%}")

    st.subheader("Badcase 浏览")
    v = st.selectbox("版本", versions, index=len(versions) - 1)
    bc = pd.read_csv(EVAL_RESULTS_DIR / v / "badcases.csv", encoding="utf-8-sig")
    st.dataframe(bc[["review_id", "gold", "pred", "tags", "reason", "review"]],
                 width="stretch", hide_index=True)
    doc = ROOT / "docs" / "prompt_iteration.md"
    if doc.exists():
        with st.expander("Prompt 迭代记录（docs/prompt_iteration.md）"):
            st.markdown(doc.read_text(encoding="utf-8"))


{"评论分析": page_analysis, "问答助手": page_chat, "评测结果": page_eval}[page]()

u = get_client().usage  # shown after the page ran, so it includes this run's calls
st.sidebar.caption(f"本进程累计：API 调用 {u.calls} 次，缓存命中 {u.cache_hits} 次，"
                   f"估算花费 {u.cost(get_settings()):.4f} {get_settings().currency}")

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Análise de crédito", page_icon="💳", layout="wide")

# ---------- Estilo ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stApp { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #EDF1F5; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }

.titulo { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: 2.9rem; line-height: 1.02; letter-spacing: -0.03em; color: #14213D; margin: 0; }
.sub { color: #51607A; font-size: 1.05rem; margin: .6rem 0 1.8rem; max-width: 34rem; }

[data-testid="stForm"] { background: #fff; border: 1px solid #D3DCE6; border-radius: 14px; padding: 1.5rem 1.5rem 1.2rem; }
[data-testid="stForm"] label p { font-weight: 500; font-size: .92rem; color: #2B3A55; }
[data-testid="stForm"] input { border-radius: 8px; }
[data-testid="stFormSubmitButton"] button { background: #14213D; color: #fff; border: 0; border-radius: 8px;
  padding: .7rem 1rem; font-weight: 600; font-size: 1rem; }
[data-testid="stFormSubmitButton"] button:hover { background: #22355F; color: #fff; }
[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid #5FD3A0; outline-offset: 2px; }

.painel { background: #14213D; color: #fff; border-radius: 20px; padding: 1.8rem 1.8rem 1.5rem; min-height: 100%; }
.painel h3 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.15rem; margin: 0 0 .2rem; color: #fff; }
.vazio { color: #A9B6CE; font-size: 1.02rem; line-height: 1.5; padding: 3rem 0; max-width: 20rem; }
.vazio b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.5rem; color: #fff; display: block; margin-bottom: .5rem; font-weight: 700; }

.gauge { width: 100%; max-width: 360px; margin: .4rem auto 0; display: block; }
.gauge .pct { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 44px; fill: #fff; }
.veredito { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.9rem; text-align: center; margin: -.3rem 0 .1rem; }
.ok { color: #5FD3A0; } .no { color: #FF8A7A; }
.leg { text-align: center; color: #A9B6CE; font-size: .9rem; margin-bottom: 1.4rem; }

.fator { border-top: 1px solid #2D3E63; padding: .85rem 0 .7rem; }
.fator .linha { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.fator .nome { font-weight: 600; }
.fator .peso { color: #A9B6CE; font-size: .88rem; white-space: nowrap; }
.barra { height: 6px; background: #2D3E63; border-radius: 3px; margin: .5rem 0 .45rem; }
.barra i { display: block; height: 100%; border-radius: 3px; background: #8FA8D8; }
.cmp { color: #C9D3E6; font-size: .9rem; }
.tag { display: inline-block; font-size: .78rem; font-weight: 600; padding: .05rem .5rem; border-radius: 99px; margin-left: .4rem; }
.tag.ok { background: rgba(95,211,160,.16); } .tag.no { background: rgba(255,138,122,.16); }
.aviso { color: #8997B3; font-size: .82rem; margin-top: 1rem; }
@media (prefers-reduced-motion: no-preference) { .gauge .arco { transition: stroke-dasharray .6s ease-out; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Dados e modelo ----------
NOMES = {
    "idade": "Idade",
    "renda_mensal": "Renda mensal",
    "valor_credito": "Crédito solicitado",
    "score_credito": "Score de crédito",
    "tempo_emprego": "Tempo de emprego",
    "dividas_mensais": "Dívidas mensais",
}
# True = quanto maior, melhor | False = quanto menor, melhor | None = neutro
MELHOR_ALTO = {"idade": None, "renda_mensal": True, "valor_credito": False,
               "score_credito": True, "tempo_emprego": True, "dividas_mensais": False}


@st.cache_resource
def carregar():
    pacote = joblib.load("modelo_credito.joblib")
    df = pd.read_csv("data/credito.csv").fillna(pacote["medianas"])
    ref = df[df["aprovado"] == 1][pacote["features"]].median().to_dict()
    return pacote, ref


def fmt(f, v):
    if f in ("renda_mensal", "valor_credito", "dividas_mensais"):
        return "R$ " + f"{v:,.0f}".replace(",", ".")
    if f == "tempo_emprego":
        return f"{v:.1f} anos".replace(".", ",")
    if f == "idade":
        return f"{v:.0f} anos"
    return f"{v:.0f}"


pacote, ref_aprovados = carregar()
modelo, features = pacote["modelo"], pacote["features"]


def gauge(prob, aprovado):
    cor = "#5FD3A0" if aprovado else "#FF8A7A"
    comp = 282.74
    return f"""<svg class="gauge" viewBox="0 0 220 135" role="img" aria-label="Probabilidade de aprovação: {prob:.0%}">
<path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="#2D3E63" stroke-width="14" stroke-linecap="round"/>
<path class="arco" d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="{cor}" stroke-width="14" stroke-linecap="round" stroke-dasharray="{prob * comp:.1f} {comp}"/>
<line x1="110" y1="6" x2="110" y2="32" stroke="#fff" stroke-width="2" stroke-dasharray="3 3"/>
<text x="110" y="104" text-anchor="middle" class="pct">{prob:.0%}</text>
<text x="110" y="124" text-anchor="middle" fill="#A9B6CE" font-size="9.5">corte de aprovação: 50%</text>
</svg>"""


# ---------- Página ----------
st.markdown('<h1 class="titulo">Análise de crédito</h1>'
            '<p class="sub">Informe o perfil financeiro e veja a decisão do modelo Random Forest, '
            'com os fatores que mais pesaram.</p>', unsafe_allow_html=True)

esq, dir_ = st.columns([5, 6], gap="large")

with esq:
    with st.form("formulario", border=False):
        c1, c2 = st.columns(2)
        idade = c1.number_input("Idade (anos)", 18, 80, 30)
        emprego = c2.number_input("Tempo de emprego (anos)", 0.0, 50.0, 3.0, step=0.5)
        renda = c1.number_input("Renda mensal (R$)", 0.0, 100000.0, 4000.0, step=100.0)
        dividas = c2.number_input("Dívidas mensais (R$)", 0.0, 100000.0, 800.0, step=50.0)
        credito = st.number_input("Crédito solicitado (R$)", 0.0, 1000000.0, 20000.0, step=500.0)
        score = st.slider("Score de crédito", 0, 1000, 600)
        enviar = st.form_submit_button("Analisar perfil", use_container_width=True)

with dir_:
    if not enviar:
        st.markdown('<div class="painel"><div class="vazio"><b>Aguardando um perfil</b>'
                    'Preencha o formulário e clique em Analisar perfil. O resultado aparece aqui.</div></div>',
                    unsafe_allow_html=True)
    else:
        valores = dict(zip(features, [idade, renda, credito, score, emprego, dividas]))
        prob = modelo.predict_proba(pd.DataFrame([valores]))[0][1]
        aprovado = prob >= 0.5
        top3 = sorted(zip(features, modelo.feature_importances_), key=lambda x: -x[1])[:3]
        maior = top3[0][1]

        linhas = ""
        for f, imp in top3:
            v, r = valores[f], ref_aprovados[f]
            tag = ""
            if MELHOR_ALTO[f] is not None:
                bom = (v >= r) if MELHOR_ALTO[f] else (v <= r)
                tag = ('<span class="tag ok">a favor</span>' if bom
                       else '<span class="tag no">contra</span>')
            linhas += (
                f'<div class="fator"><div class="linha"><span class="nome">{NOMES[f]}{tag}</span>'
                f'<span class="peso">{imp:.0%} da decisão</span></div>'
                f'<div class="barra"><i style="width:{imp / maior * 100:.0f}%"></i></div>'
                f'<div class="cmp">Você: {fmt(f, v)} · mediana dos aprovados: {fmt(f, r)}</div></div>'
            )

        st.markdown(
            f'<div class="painel">{gauge(prob, aprovado)}'
            f'<div class="veredito {"ok" if aprovado else "no"}">{"Crédito aprovado" if aprovado else "Crédito reprovado"}</div>'
            f'<div class="leg">Probabilidade estimada de aprovação</div>'
            f'<h3>Os 3 fatores que mais pesaram</h3>{linhas}'
            f'<div class="aviso">Resultado meramente simulado, não representa uma decisão bancária real.</div></div>',
            unsafe_allow_html=True,
        )

with st.expander("Desempenho do modelo nos dados de teste"):
    with open("metricas.json", encoding="utf-8") as f:
        m = json.load(f)["metricas"]
    a, b, c, d = st.columns(4)
    a.metric("Acurácia", f"{m['acuracia']:.0%}")
    b.metric("Precisão", f"{m['precisao']:.0%}")
    c.metric("Recall", f"{m['recall']:.0%}")
    d.metric("F1-score", f"{m['f1']:.0%}")

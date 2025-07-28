from dotenv import load_dotenv

load_dotenv()

"""
Streamlit アプリケーション
========================

このアプリでは、利用者が質問したい内容を入力フォームから送信し、
ラジオボタンで指定した専門家のキャラクターに基づいて大規模言語モデル
(LLM) が回答を生成します。LangChain を利用してシステムメッセージと
ユーザーの質問からプロンプトを構築し、ChatOpenAI モデルに渡します。

使用方法:

1. 画面のテキストエリアに質問内容を入力します。
2. ラジオボタンから希望する専門家を選択します（料理、法律、旅行）。
3. "送信" ボタンを押すと、選択された分野の専門家として LLM が回答を生成し、画面上に表示します。

このコードでは、Lesson8 で取り扱った LangChain のシンプルな使い方を参考に、
`ChatPromptTemplate` と `LLMChain` を組み合わせて使用しています。専門家の
タイプによって異なるシステムメッセージが適用されるため、同じ質問でも違う
視点からの回答を得ることができます。
"""


import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage



# グローバルで LLM インスタンスを初期化
llm = ChatOpenAI(temperature=0)

def get_llm_response(question: str, expert: str) -> str:
    """
    入力テキストと専門家タイプを受け取り、LLM からの回答を返す関数。

    Parameters
    ----------
    question : str
        ユーザーが入力する質問文。
    expert : str
        ラジオボタンで選択された専門家の種類。

    Returns
    -------
    str
        LLM から生成された回答。
    """
    # 専門家ごとにシステムメッセージを定義
    system_prompts = {
        "料理の専門家": (
            "あなたは優秀な料理の専門家です。家庭料理からプロの料理まで幅広い"
            "知識を持ち、利用者の質問に対して料理に関する具体的かつ実践的な"
            "アドバイスやレシピを提供してください。"
        ),
        "法律の専門家": (
            "あなたは経験豊富な法律の専門家です。法律用語をわかりやすく説明し、"
            "利用者の質問に対して日本の法体系を前提とした見解や助言を提供してください。"
        ),
        "旅行アドバイザー": (
            "あなたは世界中の観光地に詳しい旅行アドバイザーです。旅行計画やおすすめの"
            "観光スポット、季節ごとの見どころなど、利用者の質問に合わせて役立つ情報を"
            "提供してください。"
        ),
    }

    # 選択された専門家に応じてシステムメッセージを取得
    system_message = system_prompts.get(
        expert,
        "あなたは有能な専門家です。質問に対して誠実かつ明確に回答してください。",
    )

    # LangChain のプロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{question}"),
        ]
    )

    # LLM とチェーンを初期化
    # モデル名を指定しない場合、環境変数に基づくデフォルト (gpt-3.5-turbo など) が使用されます。
    # 質問をパラメータに渡してモデルを実行
    # テンプレートの変数名が {question} なので、キーも "question" にする
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"question": question})
    return response
    response = chain.run({"question": question})
    return response


def main() -> None:
    """
    Streamlit アプリのエントリポイント。
    ユーザーの入力を受け取り、専門家の種類に応じて LLＭ の回答を表示します。
    """
    st.set_page_config(page_title="専門家に質問できるAIアプリ", page_icon="🤖")
    st.title("専門家に質問できるAIアプリ")
    st.markdown(
        """
        このアプリでは、以下の手順で質問に対する AI の回答を得ることができます。

        1. 下のフォームに質問を入力します。
        2. ラジオボタンから、質問に回答してほしい専門家の種類を選択します。
        3. **送信** ボタンを押すと、選択した専門家として LLＭ が回答を生成し、このページに表示します。

        LangChain を使用し、システムプロンプトを専門家ごとに切り替えることで、
        同じ質問でも異なる視点からの回答が得られる構成になっています。
        """,
        unsafe_allow_html=True,
    )

    # 入力フォーム
    with st.form(key="input_form"):
        user_question = st.text_area(
            "質問を入力してください",
            placeholder="例: 簡単に作れる夕食レシピを教えてください。",
        )
        expert_type = st.radio(
            "回答してほしい専門家を選択してください",
            ("料理の専門家", "法律の専門家", "旅行アドバイザー"),
        )
        submit = st.form_submit_button("送信")

    if submit:
        if not user_question.strip():
            st.warning("質問を入力してください。", icon="⚠️")
        else:
            # LLＭ へ問い合わせ
            with st.spinner("AI が回答を生成しています…"):
                try:
                    answer = get_llm_response(user_question, expert_type)
                    st.markdown("### 回答")
                    st.write(answer)
                except Exception as e:
                    # API キー未設定、ネットワーク障害、LangChain のエラーなどが発生する可能性があります
                    st.error(
                        "回答の生成中にエラーが発生しました。API キーの設定やネットワーク環境をご確認ください。\n"
                        "An error occurred while generating the answer. Please check your API key settings and network environment."
                    )


if __name__ == "__main__":
    main()
    

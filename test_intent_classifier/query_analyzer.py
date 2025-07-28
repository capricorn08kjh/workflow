# query_analyzer.py

import json
from llama_index.llms.openai_like import OpenAILike

class QueryAnalyzer:
    """사용자 쿼리를 분석하여 명확성을 판단하고, 불분명할 경우 질문을 생성하는 클래스"""

    def __init__(self, llm_config: dict):
        self.llm = OpenAILike(
            model=llm_config['model_name'],
            api_base=llm_config['openai_api_base'],
            api_key=llm_config['openai_api_key'],
            temperature=0.1,
            is_chat_model=True
        )

    def _build_prompt(self, user_query: str, table_schemas: str) -> str:
        """쿼리 분석 및 질문 생성을 위한 프롬프트 생성"""
        return f"""
        당신은 사용자 질의를 분석하는 AI 어시스턴트입니다.
        주어진 '사용자 질의'가 '테이블 스키마' 정보를 바탕으로 명확하고 구체적인 Oracle SQL로 변환될 수 있는지 판단해주세요.

        [분석 가이드라인]
        1. 모호한 표현이 있는지 확인하세요. (예: '최근', '인기 있는', '실적이 좋은')
        2. 구체적인 조건이 누락되었는지 확인하세요. (예: '매출 조회' -> 어느 기간? 어느 제품?)
        3. 날짜, 숫자, 특정 이름 등 필수 파라미터가 빠졌는지 확인하세요.

        [결과 포맷]
        - 분석 결과, 쿼리가 명확하고 충분하면: {{"status": "clear"}}
        - 쿼리가 모호하거나 정보가 부족하면, 사용자에게 되물을 친절한 질문을 한국어로 생성하여 아래 형식으로 응답해주세요:
          {{"status": "clarification_needed", "question": "..."}}

        ---
        [테이블 스키마 정보]
        {table_schemas}

        [사용자 질의]
        {user_query}
        ---

        분석 결과를 JSON 형식으로만 반환해주세요.
        """

    def analyze(self, user_query: str, table_schemas: str) -> dict:
        """
        쿼리를 분석하여 명확성을 판단하거나 되물을 질문을 생성합니다.

        Args:
            user_query (str): 사용자의 자연어 질의
            table_schemas (str): 관련된 테이블의 스키마 정보

        Returns:
            dict: 분석 결과 (예: {"status": "clear"} 또는 {"status": "clarification_needed", "question": "..."})
        """
        prompt = self._build_prompt(user_query, table_schemas)
        try:
            response = self.llm.complete(prompt)
            result = json.loads(response.text)
            # 응답 형식 유효성 검사
            if result.get("status") == "clear":
                return {"status": "clear"}
            if result.get("status") == "clarification_needed" and "question" in result:
                return result
            # 형식이 맞지 않으면 불분명한 것으로 간주하고 일반적인 질문 반환
            return {"status": "clarification_needed", "question": "죄송합니다. 질문을 더 구체적으로 말씀해주시겠어요?"}
        except Exception:
            # 예외 발생 시, 항상 명확화가 필요한 것으로 처리
            return {"status": "clarification_needed", "question": "질문을 이해하는데 어려움이 있습니다. 다른 방식으로 질문해주시겠어요?"}

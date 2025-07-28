# intent_classifier.py

import json
from llama_index.llms.openai_like import OpenAILike

class IntentClassifier:
    """사용자 질의의 의도를 분류하는 클래스"""

    def __init__(self, llm_config: dict):
        """
        초기화 메서드
        Args:
            llm_config (dict): LLM 관련 설정 (API 주소, 키, 모델명 등)
        """
        self.llm = OpenAILike(
            model=llm_config['model_name'],
            api_base=llm_config['openai_api_base'],
            api_key=llm_config['openai_api_key'],
            temperature=0.0,
            is_chat_model=True
        )

    def _build_prompt(self, user_query: str, is_clarifying: bool) -> str:
        """의도 분류를 위한 프롬프트 생성"""
        
        if is_clarifying:
            # 시스템이 질문한 상태에서의 사용자 입력은 '추가 정보 제공'으로 간주
            return f"""
            시스템이 사용자에게 명확한 정보(예: 기간, 조건 등)를 요청한 상황입니다.
            사용자 입력: "{user_query}"
            이 입력은 이전에 불완전했던 쿼리를 완성하기 위한 추가 정보입니다.

            이 경우, 의도는 'clarification_response' 입니다.
            JSON 형식으로 응답하세요: {{"intent": "clarification_response"}}
            """

        # 일반적인 상황에서의 의도 분류
        return f"""
        사용자 질의의 의도를 다음 카테고리 중 하나로 분류해주세요:
        - 'data_query': 데이터베이스에서 정보를 조회하려는 명백한 요청 (예: "매출 보여줘", "직원 목록 조회")
        - 'greeting': 간단한 인사말 (예: "안녕", "반가워")
        - 'unknown': 기타 잡담 또는 시스템이 처리할 수 없는 요청

        사용자 질의: "{user_query}"

        가장 적절한 의도 하나를 선택하여 아래 JSON 형식으로만 응답해주세요.
        {{"intent": "..."}}
        """

    def classify(self, user_query: str, conversation_state: str = "AWAITING_QUERY") -> str:
        """
        사용자 질의의 의도를 분류합니다.

        Args:
            user_query (str): 사용자의 자연어 입력
            conversation_state (str): 현재 대화 상태 ('AWAITING_QUERY' 또는 'AWAITING_CLARIFICATION')

        Returns:
            str: 분류된 의도 ('data_query', 'greeting', 'clarification_response', 'unknown')
        """
        is_clarifying = (conversation_state == "AWAITING_CLARIFICATION")
        prompt = self._build_prompt(user_query, is_clarifying)
        
        try:
            response = self.llm.complete(prompt)
            result = json.loads(response.text)
            return result.get("intent", "unknown")
        except (json.JSONDecodeError, KeyError):
            # LLM이 JSON 형식을 따르지 않은 경우, 키워드 기반으로 단순 분류
            if any(keyword in user_query for keyword in ["조회", "찾아줘", "알려줘", "보여줘"]):
                return "data_query"
            return "unknown"

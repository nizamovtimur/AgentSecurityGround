"""
Клиент для взаимодействия с Langflow flow (run/chat).
"""

import os
import uuid
import httpx


class FlowClient:
    """
    Клиент для вызова Langflow flow по FLOW_ID.

    Использует POST /api/v1/run/{flow_id} для отправки сообщений и получения
    ответов. Поддерживает session_id для поддержания контекста диалога.
    """

    def __init__(self, flow_id, base_url=None, api_key=None):
        self.flow_id = str(flow_id).strip() if flow_id else ""
        if not self.flow_id:
            raise ValueError("flow_id cannot be empty")
        url = base_url or os.environ.get("LANGFLOW_URL", "http://localhost:7860")
        self.base_url = str(url).rstrip("/") if url else "http://localhost:7860"
        self.api_key = api_key or os.environ.get("LANGFLOW_API_KEY")
        self.session_id = None

    def chat(self, message, session_id=None):
        """
        Отправляет сообщение во флоу и возвращает ответ ассистента.

        Payload: output_type=chat, input_type=chat, input_value=message.
        Ответ извлекается из outputs[0].outputs[0].messages[0].message.

        Args:
            message: Текст сообщения пользователя
            session_id: ID сессии. Если None — создаётся UUID или используется
                       предыдущий session_id для продолжения диалога

        Returns:
            Текст ответа ассистента (message или text из первого сообщения)

        Raises:
            ValueError: если api_key не задан
            httpx.HTTPStatusError: при ошибке HTTP
        """
        if not self.api_key:
            raise ValueError("LANGFLOW_API_KEY must be set")
        message = str(message) if message is not None else ""

        sid = session_id or self.session_id or str(uuid.uuid4())
        self.session_id = sid

        payload = {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": message,
            "session_id": sid,
        }

        resp = httpx.post(
            "{}/api/v1/run/{}".format(self.base_url, self.flow_id),
            json=payload,
            headers={"x-api-key": self.api_key, "accept": "application/json"},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Langflow run response: outputs[0].outputs[0].messages[0].message
        try:
            out = data.get("outputs", [{}])[0]
            out_list = out.get("outputs", [{}])[0]
            msgs = out_list.get("messages", [{}])
            if msgs:
                msg = msgs[0].get("message") or msgs[0].get("text") or ""
                return str(msg)
        except (IndexError, KeyError, TypeError):
            pass
        return str(data)

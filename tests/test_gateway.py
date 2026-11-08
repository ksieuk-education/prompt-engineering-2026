import json

import httpx
import pytest

from agent_theatre.engine import make_plan
from agent_theatre.llm import Gateway, GenerationError
from agent_theatre.scenario import load_scenario
from agent_theatre.schema import Lookup, ScenePlan


def test_ollama_retries_truncation_and_returns_only_spoken_content():
    requests = []
    def reply(request):
        data = json.loads(request.content)
        requests.append(data)
        if len(requests) == 1:
            return httpx.Response(200, json={"done_reason": "length", "message": {"content": ""}})
        return httpx.Response(200, json={"done_reason": "stop", "message": {
            "thinking": "internal", "content": '{"query":"память"}'}})
    with httpx.Client(transport=httpx.MockTransport(reply)) as http:
        query = Gateway(http, "http://localhost:11434", 12).ask("actor", "system", "prompt", Lookup)
    assert query.query == "память"
    assert len(requests) == 2
    assert requests[1]["options"]["num_predict"] > requests[0]["options"]["num_predict"]


def test_openai_planner_protocol(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    def reply(request):
        data = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert data["text"]["format"]["strict"] is True
        assert data["text"]["format"]["name"] == "Plan"
        assert "scenes" in data["text"]["format"]["schema"]["required"]
        assert data["model"] == "example" and data["store"] is False
        plan = {"scenes": [{"title": str(n), "opening": "Вспышка.", "objective": "Проверить условие."}
                           for n in range(1, 5)]}
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "reasoning"}, {"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(plan)}]}]})
    with httpx.Client(transport=httpx.MockTransport(reply)) as http:
        plan = make_plan(load_scenario("extra"), Gateway(http, "http://localhost", 11), "openai/example")
        assert [s.title for s in plan.scenes] == ["1", "2", "3", "4"]


def test_missing_model_aborts_without_a_fabricated_play():
    def missing(request):
        return httpx.Response(404, json={"error": "model not found"})
    with httpx.Client(transport=httpx.MockTransport(missing)) as http:
        with pytest.raises(GenerationError, match="404"):
            Gateway(http, "http://localhost", 1).ask("absent", "system", "prompt", Lookup)


def test_an_ollama_token_loop_gets_one_fresh_prompt():
    prompts = []
    def reply(request):
        prompts.append(json.loads(request.content)["messages"][-1]["content"])
        if len(prompts) == 1:
            return httpx.Response(500, json={"error": "prediction aborted, token repeat limit reached"})
        return httpx.Response(200, json={"message": {"content": '{"query":"протокол"}'}})
    with httpx.Client(transport=httpx.MockTransport(reply)) as http:
        assert Gateway(http, "http://localhost", 17).ask("actor", "role", "спроси", Lookup).query == "протокол"
    assert len(prompts) == 2 and "другими словами" in prompts[-1]


def test_planner_cannot_open_the_door_before_the_votes():
    with pytest.raises(ValueError, match="ещё закрыта"):
        ScenePlan(title="Выбор", opening="Закрытая дверь медленно открывается.", objective="Проголосовать.")
    ScenePlan(title="Выбор", opening="Дверь не открывается. Слышны шаги.", objective="Проголосовать.")


def test_local_gpt_gets_a_reasoning_budget_and_returns_the_final_json():
    def reply(request):
        data = json.loads(request.content)
        assert data["think"] == "low" and data["options"]["num_predict"] > 100
        return httpx.Response(200, json={"message": {"thinking": "internal", "content": '{"query":"память"}'}})
    with httpx.Client(transport=httpx.MockTransport(reply)) as http:
        answer = Gateway(http, "http://localhost", 404).ask("gpt-oss:20b", "роль", "поищи", Lookup, budget=100)
    assert answer.query == "память"

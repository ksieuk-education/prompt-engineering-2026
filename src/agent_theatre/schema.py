import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator


def nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("нужен непустой текст")
    return value.strip()


Text = Annotated[str, AfterValidator(nonblank)]


class Form(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ScenePlan(Form):
    title: Text
    opening: Text
    objective: Text

    @field_validator("opening")
    @classmethod
    def keep_exit_closed(cls, value: str) -> str:
        # Все opening исполняются до последнего голоса. Развязкой распоряжается resolve().
        motion = r"(?<!не )\b(?:открыва|отпира|распахива|приоткрыва|открыла|открыли|отворя)\w*"
        door = r"\bдвер[ьи]\b"
        if re.search(rf"{door}[^.!?]{{0,35}}{motion}|{motion}[^.!?]{{0,35}}{door}", value, re.I):
            raise ValueError("в начале каждой сцены дверь ещё закрыта, опиши свет или движение актёра")
        return value


class Plan(Form):
    scenes: list[ScenePlan]

    @field_validator("scenes")
    @classmethod
    def four_scenes(cls, scenes: list[ScenePlan]) -> list[ScenePlan]:
        if len(scenes) != 4:
            raise ValueError("пьеса должна состоять ровно из четырёх сцен")
        return scenes


class Lookup(Form):
    query: str


class Intention(Form):
    intent: Text
    to: str
    source: str
    vote: str


class Review(Form):
    model_config = ConfigDict(populate_by_name=True)
    issue: str = Field(description="Категория одной существенной ошибки, либо пустая строка.")
    note: str = Field(alias="поправка", description="Короткая инструкция актёру на русском языке.")


def stage_text_issue(value: str) -> str | None:
    if value and (not re.search(r"[а-яё]", value, re.I)
                  or re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value)):
        return "перепиши текст целиком на русском языке"
    if re.search(r"\.txt\b|\b(?:source|stage|vote|query|line|select|challenge)\s*:", value, re.I):
        return "служебные метки нельзя произносить в пьесе"
    return None


class Delivery(Form):
    model_config = ConfigDict(populate_by_name=True)
    line: Text = Field(alias="реплика", description="Прямая речь одного актёра, без имени перед ней.")
    stage: str = Field(alias="жест", description="Один видимый жест именно этого актёра. Можно оставить пустым.")

    @field_validator("line")
    @classmethod
    def stage_language(cls, value: str) -> str:
        if issue := stage_text_issue(value):
            raise ValueError(issue)
        if re.search(r"(?:^|\n)\s*(?:Гермиона|Чендлер|Тони|Барби|Профессор)(?: [А-ЯЁ][а-яё]+)?\s*:", value):
            raise ValueError("нужна только собственная прямая речь, без имени говорящего")
        return value

    @field_validator("stage")
    @classmethod
    def optional_direction(cls, value: str) -> str:
        # Посторонняя подпись модели не должна лишать актёра корректной реплики.
        return "" if stage_text_issue(value) else value


class Speech(Delivery):
    to: str
    source: str
    vote: str
    edited: bool = False
    quality_issues: list[str] = Field(default_factory=list)

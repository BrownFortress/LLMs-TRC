from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Event():
    e_id: str
    token: str
    tense: str
    pos: str
    offset: slice
    sent_id: int
    sentence: list
    offset_paragraph: list
    paragraph_id: slice

@dataclass
class TemporalFunction():
    t_id: str
    token: str
    type_func: str
    value: str
    offset: slice
    sent_id: int
    sentence: list
    offset_paragraph: list
    paragraph_id: slice

@dataclass
class Relation():
    rel_type: str
    event1: Event
    event2: Event

@dataclass
class Document:
    d_id: str
    title: str
    extra_info: str
    text: list
    paragraphs: list
    tokens: list
    events: Dict[str, Event]
    temporal_relations: Dict[str, List[Relation]]
    time_expressions: List[TemporalFunction]

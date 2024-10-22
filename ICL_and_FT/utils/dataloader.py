from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import override
from sandbox.data_structures import Document, Event
import torch
from dataclasses import dataclass
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import re


def remove_punctuation(test_str):
    # Using filter() and lambda function to filter out punctuation characters
    result = "".join(
        filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str)
    )
    return result


@dataclass
class Prompt:
    embeddings: torch.LongTensor
    spans: list
    mask: list
    complementary_mask: list


@dataclass
class DataRow:
    d_id: str
    label: str
    id_label: int
    context: str
    event1: Event
    event2: Event
    model_input: Prompt
    prediction: str



class Prompter:
    def __init__(
        self, tokenizer, truncate_limit=None, add_spacing=True, pre_context=0, post_context=0, only_context_intra=False
    ) -> None:
        self.tokenizer = tokenizer
        self.truncate_limit = truncate_limit
        self.add_spacing = add_spacing
        self.pre_context = pre_context
        self.post_context = post_context
        self.only_context_intra = only_context_intra

    def _make_prompt(
        self,
        final_output: torch.LongTensor,
        spans: list,
        masks: list,
        complementary_masks: list,
    ) -> Prompt:
        prompt = Prompt(final_output, spans, masks, complementary_masks)
        if (
            self.truncate_limit is not None
            and prompt.embeddings.shape[0] > self.truncate_limit
        ):
            prompt = self._truncate(prompt)
        return prompt

    def _truncate(self, prompt: Prompt) -> Prompt:
        print("TRUNCATE")
        for span in prompt.spans:
            assert span[1] <= self.truncate_limit
        # truncate embeddings and masks up to self.truncate_limit
        emb = prompt.embeddings[: self.truncate_limit]

        masks = []
        for m in prompt.mask:
            masks.append(m[: self.truncate_limit])

        compl_masks = []
        for cm in prompt.complementary_mask:
            compl_masks.append(cm[: self.truncate_limit])

        return Prompt(emb, prompt.spans, masks, compl_masks)

    def plain(self, doc, e1, e2) -> Prompt:

        if e1.sent_id == e2.sent_id:
            context = " ".join(doc.text[e1.sent_id])
        else:
            context = " ".join(doc.text[e1.sent_id]) + " " +  " ".join(doc.text[e2.sent_id])

        final_output = self.tokenizer(
                    context.strip(),
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
        return self._make_prompt(final_output, None, None, None)

    def word_in_sentence(self, doc, e1, e2) -> Prompt:
        spans = []
        masks = []
        complementary_masks = []
        tmp_final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        event_list = [e1, e2]
        # Context management
        added_something = False
        debug = False
        context = ""
        start_id = 0
        if debug:
            print("="*89)
        if e1.sent_id <= e2.sent_id:
            start_id = e1.sent_id
        else:
            start_id = e2.sent_id

        # Add context only to intra sentences
        if self.only_context_intra and e1.sent_id == e2.sent_id and self.pre_context > 0:
            for id_r in range(self.pre_context - 1, 0, -1):
                back_pos = start_id - id_r
                if back_pos >= 0:
                    context += " ".join(doc.text[back_pos]).strip() + " "
            if context != "":
                added_something = True
                encoded_context = self.tokenizer(
                    context.strip(),
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                tmp_final_output = torch.cat([tmp_final_output, encoded_context], dim=0).type(
                    torch.LongTensor
                )

        elif self.pre_context > 0:
            for id_r in range(self.pre_context - 1, 0, -1):
                back_pos = start_id - id_r
                if back_pos >= 0:
                    context += " ".join(doc.text[back_pos]).strip() + " "
            if context != "":
                added_something = True
                encoded_context = self.tokenizer(
                    context.strip(),
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                tmp_final_output = torch.cat([tmp_final_output, encoded_context], dim=0).type(
                    torch.LongTensor
                )

        if e1.sent_id != e2.sent_id:
            # ==================== DIFFERENT SENTENCE =====================================
            if debug:
                print('DIFF!')

            for id_ev, ev in enumerate(event_list):
                sentence = doc.text[ev.sent_id]


                if ev.offset.start > 0:
                    if (id_ev > 0 or added_something) and self.add_spacing:
                        # Here we add a space because for the tokenizer Ġsaid e said are different where Ġ is a whitespace
                        # Not for lama
                        pre_event = self.tokenizer(
                            " " + " ".join(sentence[0 : ev.offset.start]),
                            padding=False,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]
                    else:
                        pre_event = self.tokenizer(
                            " ".join(sentence[0 : ev.offset.start]),
                            padding=False,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]
                else:
                    pre_event = torch.Tensor([])
                if (
                    " ".join(sentence[0 : ev.offset.start]) != " "
                    and self.add_spacing
                ):
                    # Here we add a space because for the tokenizer Ġsaid e said are different where Ġ is a whitespace
                    event = self.tokenizer(
                        " " + " ".join(sentence[ev.offset]),
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                else:
                    event = self.tokenizer(
                        " ".join(sentence[ev.offset]),
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                # Here we add a space because for the tokenizer Ġsaid e said are different where Ġ is a whitespace
                if ev.offset.stop < len(sentence):
                    if self.add_spacing:
                        post_event = self.tokenizer(
                            " " + " ".join(sentence[ev.offset.stop :]),
                            padding=False,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]

                    else:
                        post_event = self.tokenizer(
                            " ".join(sentence[ev.offset.stop :]),
                            padding=False,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]
                    # print(final_output.size(), len(final_output), " ".join(doc.text[ev.sent_id][0:ev.offset]))
                else:
                    post_event = torch.tensor([])
                tmp_final_output = torch.cat([tmp_final_output, pre_event], dim=0).type(
                    torch.LongTensor
                )
                s = len(tmp_final_output)
                tmp_final_output = torch.cat([tmp_final_output, event], dim=0).type(
                    torch.LongTensor
                )
                e = len(tmp_final_output)  # - 1 # We remove the extra space we added
                tmp_final_output = torch.cat([tmp_final_output, post_event], dim=0).type(
                    torch.LongTensor
                )
                context += " ".join(sentence).strip() + " "
                # print(s,e)
                spans.append((s, e))
                if debug:
                    print("In LOOP:", self.tokenizer.convert_ids_to_tokens(tmp_final_output), self.tokenizer.decode(tmp_final_output[s:e]).strip().replace(" ", ""), (s, e), ev.token)
                    print("Decoded:", self.tokenizer.decode(tmp_final_output[s:e]).strip().replace(" ", "") == ev.token)
                assert self.tokenizer.decode(tmp_final_output[s:e]).strip().replace(" ", "") == ev.token.strip().replace(" ", "")

        else:
            # ==================== SAME SENTENCE =====================================
            if debug:
                print('SAME!')

            start = 0
            sentence = doc.text[e1.sent_id]
            context += " ".join(sentence).strip()
            for id_ev, ev in enumerate(event_list):
                if (
                    (id_ev > 0 or added_something)
                    and " ".join(sentence[start : ev.offset.start]).strip() != ""
                    and self.add_spacing
                ):
                    # Here we add a space because for the tokenizer Ġsaid e said are different where Ġ is a whitespace
                    pre_event = self.tokenizer(
                        " " + " ".join(sentence[start : ev.offset.start]).strip(),
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                else:
                    pre_event = self.tokenizer(
                        " ".join(sentence[start : ev.offset.start]),
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                if (
                    " ".join(sentence[start : ev.offset.start]) != "" or id_ev > 0
                ) and self.add_spacing:
                    # Here we add a space because for the tokenizer Ġsaid e said are different where Ġ is a whitespace
                    event = self.tokenizer(
                        " " + " ".join(sentence[ev.offset]),
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                else:
                    event = self.tokenizer(
                        " ".join(sentence[ev.offset]),
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                start = ev.offset.start + 1
                tmp_final_output = torch.cat([tmp_final_output, pre_event], dim=0).type(
                    torch.LongTensor
                )

                s = len(tmp_final_output)
                tmp_final_output = torch.cat([tmp_final_output, event], dim=0).type(
                    torch.LongTensor
                )
                e = len(tmp_final_output)  # - 1 # We remove the extra space we added
                if id_ev + 1 == len(event_list):
                    if " ".join(sentence[ev.offset.stop :]) != "" and self.add_spacing:
                        # Here we add a space because for the tokenizer Ġsaid e said are different where Ġ is a whitespace
                        post_event = self.tokenizer(
                            " " + " ".join(sentence[ev.offset.stop :]),
                            padding=False,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]
                    else:
                        post_event = self.tokenizer(
                            " ".join(sentence[ev.offset.stop :]),
                            padding=False,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]
                    tmp_final_output = torch.cat([tmp_final_output, post_event], dim=0).type(
                    torch.LongTensor
                )
                if debug:
                    print(tmp_final_output)
                    print('LOOP:', self.tokenizer.convert_ids_to_tokens(tmp_final_output), (s, e), ev.token)
                    print('LOOP:', self.tokenizer.convert_ids_to_tokens(tmp_final_output))
                    print('LOOP:', self.tokenizer.decode(tmp_final_output[s:e]), (s, e), ev.token)
                assert self.tokenizer.decode(tmp_final_output[s:e]).strip().replace(" ", "") == ev.token.strip().replace(" ", "")
                spans.append((s, e))

        if context[-1] != " ":
            context += " "

        if self.only_context_intra and e1.sent_id == e2.sent_id and self.post_context > 0:
            for id_r in range(1, self.pre_context + 1):
                foward_pos = start_id + id_r
                if foward_pos < len(doc.text):
                    if self.add_spacing:
                        context += " ".join(doc.text[foward_pos]).strip() + " "
                    else:
                        post = self.tokenizer(
                                " ".join(doc.text[foward_pos]).strip(),
                                padding=False,
                                return_tensors="pt",
                                add_special_tokens=False,
                            )["input_ids"][0]
                        tmp_final_output = torch.cat([tmp_final_output, post], dim=0)
        elif self.post_context > 0:
            for id_r in range(1, self.pre_context + 1):
                foward_pos = start_id + id_r
                if foward_pos < len(doc.text):
                    if self.add_spacing:
                        context += " ".join(doc.text[foward_pos]).strip() + " "
                    else:
                        post = self.tokenizer(
                            " ".join(doc.text[foward_pos]).strip(),
                            padding=False,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]
                        tmp_final_output = torch.cat([tmp_final_output, post], dim=0)


        context = context.strip()
        if  self.add_spacing:
            final_output = self.tokenizer(context, padding=False, return_tensors="pt")[
                "input_ids"
            ][0]
        else:
            eos = torch.LongTensor([self.tokenizer.eos_token_id])
            final_output = torch.cat([tmp_final_output, eos], dim=0)

        for id_s, span in enumerate(spans):
            ev = event_list[id_s]
            if (
                self.tokenizer.decode(final_output[span[0] : span[1]]).strip().replace(" ", "")
                != ev.token.strip().replace(" ", "")
            ):
                tokens = self.tokenizer.convert_ids_to_tokens(final_output)
                tmp_tokens = self.tokenizer.convert_ids_to_tokens(tmp_final_output)
                for id_x, x in enumerate(tmp_tokens):
                    if x != tokens[id_x]:
                        print(x, tokens[id_x], id_x)
                        break

                print(
                    self.tokenizer.convert_ids_to_tokens(final_output),
                    span,'Event:',
                    ev.token,
                    "DEC:",
                    self.tokenizer.convert_ids_to_tokens(
                        final_output[span[0] : span[1]]
                    ),
                )
            assert (
                self.tokenizer.decode(final_output[span[0] : span[1]]).strip().replace(" ", "") == ev.token.strip().replace(" ", "")
            )

        for se in spans:
            mask = torch.zeros(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(-math.inf)
            mask[se[0] : se[1]] = 1
            complementary_mask[se[0] : se[1]] = 0
            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def events_eos_paragraph(self, doc, e1, e2) -> Prompt:
        context1 = doc.paragraphs[e1.paragraph_id]
        context2 = doc.paragraphs[e2.paragraph_id]

        if e1.paragraph_id != e2.paragraph_id:
            if e1.paragraph_id < e2.paragraph_id:
                context = " ".join(context1) + " [SEP] " + " ".join(context2)
            else:
                context = " ".join(context2) + " [SEP] " + " ".join(context1)
        else:
            context = " ".join(context1)

        special_tokens = [" [event1]: ", " [event2]: "]
        event_list = [e1, e2]
        context_embeddings = self.tokenizer(
            context, padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        final_output = torch.cat([final_output, context_embeddings])
        spans = []
        for e_id, ev in enumerate(event_list):
            special_token_encoding = self.tokenizer(
                special_tokens[e_id],
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            final_output = torch.cat([final_output, special_token_encoding])
            start = len(final_output)
            ev_embedding = self.tokenizer(
                ev.token, padding=False, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            final_output = torch.cat([final_output, ev_embedding])
            end = len(final_output)
            spans.append((start, end))

        eos = torch.LongTensor([self.tokenizer.eos_token_id])
        final_output = torch.cat([final_output, eos], dim=0)

        masks, complementary_masks = [], []
        for se in spans:
            mask = torch.zeros(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(-math.inf)
            mask[se[0] : se[1]] = 1  # We multiply this to isolate the event embeddings
            complementary_mask[
                se[0] : se[1]
            ] = 0  # We add this to not bias the maxpooling
            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def events_eos(self, doc, e1, e2) -> Prompt:
        context1 = doc.text[e1.sent_id]
        context2 = doc.text[e2.sent_id]

        if e1.sent_id != e2.sent_id:
            if e1.sent_id < e2.sent_id:
                context = " ".join(context1) + " [SEP] " + " ".join(context2)
            else:
                context = " ".join(context2) + " [SEP] " + " ".join(context1)
        else:
            context = " ".join(context1)

        special_tokens = [" [event1]: ", " [event2]: "]
        event_list = [e1, e2]
        context_embeddings = self.tokenizer(
            context, padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        final_output = torch.cat([final_output, context_embeddings])
        spans = []
        for e_id, ev in enumerate(event_list):
            special_token_encoding = self.tokenizer(
                special_tokens[e_id],
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            final_output = torch.cat([final_output, special_token_encoding])
            start = len(final_output)
            ev_embedding = self.tokenizer(
                ev.token, padding=False, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            final_output = torch.cat([final_output, ev_embedding])
            end = len(final_output)
            spans.append((start, end))

        eos = torch.LongTensor([self.tokenizer.eos_token_id])
        final_output = torch.cat([final_output, eos], dim=0)

        masks, complementary_masks = [], []
        for se in spans:
            mask = torch.zeros(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(-math.inf)
            mask[se[0] : se[1]] = 1  # We multiply this to isolate the event embeddings
            complementary_mask[
                se[0] : se[1]
            ] = 0  # We add this to not bias the maxpooling

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class PromptPrompter(Prompter):
    ev1_sub = "<ev1>"
    ev2_sub = "<ev2>"
    classes_sub = "<classes>"
    context_sub = "<context>"
    examples_sub = "<examples>"
    ev1_q_sub = "event1"
    ev2_q_sub = "event2"

    def __init__(
        self,
        prompt: str,
        classes: List[str],
        tokenizer,
        truncate_limit: Optional[int] = None,
        examples: List[Tuple[Document, Event, Event, str]] = None,
        rel_type_order: List[str] = None,
        highlight_events: str = "in_sentence",
        questions: Optional[Dict[str, str]] = None,
        prompt_type: str = "class",
        ev_in_question: bool = False
    ):
        super(PromptPrompter, self).__init__(tokenizer, truncate_limit=truncate_limit)
        self.prompt = prompt
        self.classes = classes
        self.classes_str = "{" + ", ".join([label for label in self.classes if label != "VAGUE"]) + "}"
        self.examples = examples
        self.example_embs: torch.Tensor = None
        self.highlight_events = highlight_events
        self.questions = questions
        self.prompt_type = prompt_type
        self.ev_in_question = ev_in_question
        self.rel_type_order = rel_type_order
        if self.examples is not None and len(self.examples) > 0:
            assert self.rel_type_order is not None
            self._prepare_examples()

    def _get_start_context_emb(self) -> torch.Tensor:
        embeddings = torch.LongTensor([])
        if self.ev_in_question and self.prompt_type != "class":
            embeddings = self.tokenizer("Given the context: ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        return embeddings

    def _get_end_context_emb(self) -> torch.Tensor:
        embeddings = torch.LongTensor([])
        if self.ev_in_question and self.prompt_type != "class":
            end_context = " Answer the question: "
            if self.prompt_type == "questions_sequential":
                end_context = " Answer the questions: "
            embeddings = self.tokenizer(end_context, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        return embeddings

    def _get_context_emb_ev_in_sentence(self, doc: Document, e1: Event, e2: Event) -> torch.Tensor:
        # <context with marked events>
        event_list = [e1, e2]
        # embeddings = torch.LongTensor([])
        embeddings = self._get_start_context_emb()
        if e1.sent_id != e2.sent_id:
            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event =  self.tokenizer(" ".join(doc.text[ev.sent_id][0:ev.offset.start]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

                event =  self.tokenizer(ev_start + " ".join(doc.text[ev.sent_id][ev.offset]) + ev_end, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                post_event =  self.tokenizer(" ".join(doc.text[ev.sent_id][ev.offset.stop:]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

                embeddings = torch.cat([embeddings, pre_event, event, post_event], dim=0).type(torch.LongTensor)
        else:
            sentence = doc.text[e1.sent_id]
            start = 0

            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event =   self.tokenizer(" ".join(sentence[start:ev.offset.start]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                event =  self.tokenizer(ev_start + " ".join(doc.text[ev.sent_id][ev.offset]) + ev_end, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                start = ev.offset.start + 1

                embeddings = torch.cat([embeddings, pre_event, event], dim=0).type(torch.LongTensor)

                if id_ev+1 == len(event_list):
                    post_event =  self.tokenizer(" ".join(sentence[ev.offset.stop:]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                    embeddings =  torch.cat([embeddings, post_event], dim=0)

        end_emb = self._get_end_context_emb()
        embeddings = torch.cat([embeddings, end_emb], dim=0).type(torch.LongTensor)
        return embeddings

    def _get_context_emb_ev_end(self, doc: Document, e1: Event, e2: Event) -> torch.Tensor:
        # s1 s2 [event1]: e1 [event2]: e2
        embeddings = torch.LongTensor([])

        # context
        context = self.tokenizer(" ".join(doc.text[e1.sent_id]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if e1.sent_id != e2.sent_id:
            second_sentence = self.tokenizer(" ".join(doc.text[e2.sent_id]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            context = torch.cat([context, second_sentence], dim=0).type(torch.LongTensor)
        embeddings = torch.cat([embeddings, context], dim=0).type(torch.LongTensor)

        # events
        event_list = [e1, e2]
        events = torch.LongTensor([])
        for i, ev in enumerate(event_list):
            event =  self.tokenizer(f" [event{i+1}]: " + " ".join(doc.text[ev.sent_id][ev.offset]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            events = torch.cat([events, event], dim=0).type(torch.LongTensor)
        embeddings = torch.cat([embeddings, events], dim=0).type(torch.LongTensor)

        return embeddings

    def _get_context_emb(self, doc: Document, e1: Event, e2: Event) -> torch.Tensor:
        if self.highlight_events == "in_sentence":
            embeddings = self._get_context_emb_ev_in_sentence(doc, e1, e2)
        elif self.highlight_events == "end":
            embeddings = self._get_context_emb_ev_end(doc, e1, e2)
        else:
            raise Exception(f"No strategy for highlight_events = '{self.highlight_events}'")
        return embeddings

    def _prepare_examples_class(self, label, context_emb):
        example_embs = []
        arrow_emb = self.tokenizer(f" ->", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        label_emb = self.tokenizer(f" {label} \n", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        example = torch.cat([context_emb, arrow_emb, label_emb], dim=0).type(torch.LongTensor)
        example_embs.append(example)
        if self.example_embs is None:
            self.example_embs = torch.cat(example_embs, dim=0).type(torch.LongTensor)
        else:
            self.example_embs = torch.cat([self.example_embs] + example_embs, dim=0).type(torch.LongTensor)

    def _prepare_question(self, question: str, e1: Event, e2: Event) -> str:
        if self.ev_in_question:
            question = re.sub(r"event1", f"[event1] {e1.token} [/event1]", question)
            question = re.sub(r"event2", f"[event2] {e2.token} [/event2]", question)
        return question

    def _prepare_examples_questions_single(self, label, context_emb, e1: Event, e2: Event):
        example_embs = {q_label: [] for q_label in self.rel_type_order}
        for q_label in self.rel_type_order:
            question = self.questions[q_label]
            answer = "YES" if label == q_label else "NO"
            question = self._prepare_question(question, e1, e2)
            q_emb = self.tokenizer(f" {question}", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            a_emb = self.tokenizer(f" {answer} \n", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            label_emb = torch.cat([q_emb, a_emb], dim=0).type(torch.LongTensor)
            example = torch.cat([context_emb, label_emb], dim=0).type(torch.LongTensor)
            example_embs[q_label].append(example)
        if self.example_embs is None:
            self.example_embs = {q_label: None for q_label in self.rel_type_order}
            self.example_embs = {
                q_label: torch.cat(example_embs[q_label], dim=0).type(torch.LongTensor)
                for q_label, q_ex_emb in self.example_embs.items()
            }
        else:
            self.example_embs = {
                q_label: torch.cat([q_ex_emb] + example_embs[q_label], dim=0).type(torch.LongTensor)
                for q_label, q_ex_emb in self.example_embs.items()
            }

    def _prepare_examples_questions_all(self, label, context_emb, e1: Event, e2: Event):
        example_embs = []
        for q_label in self.rel_type_order:
            question = self.questions[q_label]
            answer = "YES" if label == q_label else "NO"
            question = self._prepare_question(question, e1, e2)
            q_emb = self.tokenizer(f" {question}", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            a_emb = self.tokenizer(f" {answer} \n", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            label_emb = torch.cat([q_emb, a_emb], dim=0).type(torch.LongTensor)
            example = torch.cat([context_emb, label_emb], dim=0).type(torch.LongTensor)
            example_embs.append(example)
        if self.example_embs is None:
            self.example_embs = torch.cat(example_embs, dim=0).type(torch.LongTensor)
        else:
            self.example_embs = torch.cat([self.example_embs] + example_embs, dim=0).type(torch.LongTensor)

    def _prepare_examples_questions_seq(self, label, context_emb, e1: Event, e2: Event):
        # context q1 a1 q2 a2 ...
        example_embs = [context_emb]
        for q_label in self.rel_type_order:
            question = self.questions[q_label]
            answer = "YES" if label == q_label else "NO"
            question = self._prepare_question(question, e1, e2)
            q_emb = self.tokenizer(f" {question}", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            a_emb = self.tokenizer(f" {answer} ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            label_emb = torch.cat([q_emb, a_emb], dim=0).type(torch.LongTensor)
            example_embs.append(label_emb)
        newline_emb = self.tokenizer(f" \n", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        example_embs.append(newline_emb)
        if self.example_embs is None:
            self.example_embs = torch.cat(example_embs, dim=0).type(torch.LongTensor)
        else:
            self.example_embs = torch.cat([self.example_embs] + example_embs, dim=0).type(torch.LongTensor)

    def _prepare_examples(self):
        assert self.rel_type_order is not None
        # <example> <label> \n <example> <label> \n ...
        example_embs = []
        newline_emb = self.tokenizer("\n", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        for doc, e1, e2, label in self.examples:
            context_emb = self._get_context_emb(doc, e1, e2)
            # <label> vs <question> <YES/NO>
            if self.prompt_type == "class":
                self._prepare_examples_class(label, context_emb)
            elif self.prompt_type == "questions_single":
                self._prepare_examples_questions_single(label, context_emb, e1, e2)
            elif self.prompt_type == "questions_all":
                self._prepare_examples_questions_all(label, context_emb, e1, e2)
            elif self.prompt_type == "questions_sequential":
                self._prepare_examples_questions_seq(label, context_emb, e1, e2)

    def few_shot(self, doc, e1, e2) -> Prompt:
        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        if self.example_embs is not None:
            final_output = torch.cat([final_output, self.example_embs], dim=0).type(torch.LongTensor)
        context_emb = self._get_context_emb(doc, e1, e2)
        arrow_emb = self.tokenizer(f" -> ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        final_output = torch.cat([final_output, context_emb, arrow_emb], dim=0).type(torch.LongTensor)

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def few_shot_prompt(self, doc, e1, e2) -> Prompt:
        prompt = self.few_shot(doc, e1, e2)
        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        sys_prompt_str = B_INST + B_SYS + self.prompt + E_SYS
        sys_prompt_str = re.sub(PromptPrompter.classes_sub, self.classes_str, sys_prompt_str)
        sys_prompt_embeddings = self.tokenizer(sys_prompt_str, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e_inst_emb = self.tokenizer(E_INST, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # embeddings[1:] -> remove old bos token
        final_output = torch.cat([final_output, sys_prompt_embeddings, prompt.embeddings[1:], e_inst_emb])
        return self._make_prompt(final_output, prompt.spans, prompt.mask, prompt.complementary_mask)


    def _get_few_shot_questions_examples_emb(self, doc, e1, e2, q_rel_type) -> torch.Tensor:
        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        if self.example_embs is not None:
            if self.prompt_type == "questions_single":
                example_embs = self.example_embs[q_rel_type]
            else: # if self.prompt_type == "questions_all":
                example_embs = self.example_embs
            final_output = torch.cat([final_output, example_embs], dim=0).type(torch.LongTensor)
        return final_output

    def _get_few_shot_questions_question_emb(self, e1, e2, q_rel_type) -> torch.Tensor:
        question = self.questions[q_rel_type]
        question = self._prepare_question(question, e1, e2)
        question_emb = self.tokenizer(f" {question} ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        return question_emb

    def few_shot_questions(self, doc, e1, e2, q_rel_type) -> Prompt:
        examples_emb = self._get_few_shot_questions_examples_emb(doc, e1, e2, q_rel_type)
        context_emb = self._get_context_emb(doc, e1, e2)
        question_emb = self._get_few_shot_questions_question_emb(e1, e2, q_rel_type)

        final_output = torch.cat([examples_emb, context_emb, question_emb], dim=0).type(torch.LongTensor)

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def few_shot_questions_prompt(self, doc, e1, e2, q_rel_type) -> Prompt:
        examples_emb = self._get_few_shot_questions_examples_emb(doc, e1, e2)
        context_emb = self._get_context_emb(doc, e1, e2)
        question_emb = self._get_few_shot_questions_question_emb(e1, e2, q_rel_type)


        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        sys_prompt_str = B_INST + B_SYS + self.prompt + " \n"
        sys_prompt_embeddings = self.tokenizer(sys_prompt_str, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e_sys_emb = self.tokenizer(E_SYS, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e_inst_emb = self.tokenizer(E_INST + " ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # [1:] -> remove old bos token
        # [:-1] -> ignore last \n in examples
        assert self.tokenizer.decode(examples_emb[-1]) == "\n"
        final_output = torch.cat([final_output, sys_prompt_embeddings, examples_emb[1:-1], e_sys_emb, context_emb, question_emb, e_inst_emb])

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def few_shot_questions_seq(self, doc, e1, e2, include_examples=True) -> dict:
        prompt = {}
        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        if self.example_embs is not None and include_examples:
            # examples same as "questions_all"
            final_output = torch.cat([final_output, self.example_embs], dim=0).type(torch.LongTensor)
        context_emb = self._get_context_emb(doc, e1, e2)

        question_embs = {}
        for q_rel_type in self.rel_type_order:
            question = self.questions[q_rel_type]
            question = self._prepare_question(question, e1, e2)
            question_embs[q_rel_type] = self.tokenizer(f" {question} ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        final_output = torch.cat([final_output, context_emb], dim=0).type(torch.LongTensor)

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        prompt = {
            "input": self._make_prompt(final_output, spans, masks, complementary_masks),
            "questions": question_embs,
        }
        return prompt

    def few_shot_questions_seq_prompt(self, doc, e1, e2) -> dict:
        # B_INST B_SYS <sys_prompt> <examples> E_SYS <prompt>
        prompt = self.few_shot_questions_seq(doc, e1, e2, include_examples=False)
        questions = prompt["questions"]
        prompt = prompt["input"]
        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])

        sys_prompt_str = B_INST + B_SYS + self.prompt
        sys_prompt_embeddings = self.tokenizer(sys_prompt_str, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if self.example_embs is not None:
            newline_emb = self.tokenizer("\n", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            # self.example_embs[:-1] -> ignore last \n in examples
            assert self.tokenizer.decode(self.example_embs[-1]) == "\n"
            sys_prompt_embeddings = torch.cat([sys_prompt_embeddings, newline_emb, self.example_embs[:-1]], dim=0).type(torch.LongTensor)
        e_sys_emb = self.tokenizer(E_SYS, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        sys_prompt_embeddings = torch.cat([sys_prompt_embeddings, e_sys_emb])

        # E_INST when making questions
        final_output = torch.cat([final_output, sys_prompt_embeddings, prompt.embeddings[1:]])
        return {
            "input": self._make_prompt(final_output, prompt.spans, prompt.mask, prompt.complementary_mask),
            "questions": questions
        }


    def system_prompt(self, doc, e1, e2) -> Prompt:
        # As https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/generation.py#L44
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        spans = []
        masks = []
        complementary_masks = []
        event_list = [e1, e2]
        special_tokens = [" [event1]: ", " [event2]: "]


        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])

        # <s> [INST] <<SYS>>\n system prompt <</SYS>>\n\n user prompt [/INST] model answer </s>
        prompt_str = B_INST + B_SYS + self.prompt + E_SYS
        prompt_str = re.sub(PromptPrompter.classes_sub, self.classes_str, prompt_str)
        prompt_embeddings = self.tokenizer(prompt_str, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        final_output = torch.cat([final_output, prompt_embeddings])

        if e1.sent_id != e2.sent_id:
            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event =  self.tokenizer(" ".join(doc.text[ev.sent_id][0:ev.offset.start]) + ev_start, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

                event =  self.tokenizer(" ".join(doc.text[ev.sent_id][ev.offset]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                post_event =  self.tokenizer(ev_end + " ".join(doc.text[ev.sent_id][ev.offset.stop:])  + " " , padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

                final_output = torch.cat([final_output, pre_event], dim=0).type(torch.LongTensor)
                s = len(final_output)
                final_output = torch.cat([final_output, event], dim=0).type(torch.LongTensor)
                e = len(final_output)
                final_output =  torch.cat([final_output, post_event], dim=0).type(torch.LongTensor)

                assert re.sub(r"\s", r"", self.tokenizer.decode(final_output[s:e])) == re.sub(r"\s", r"", ev.token)

                spans.append((s, e))
        else:
            sentence = doc.text[e1.sent_id]
            context = " ".join(sentence)
            start = 0

            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event =   self.tokenizer(" ".join(sentence[start:ev.offset.start]) + ev_start, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                event =  self.tokenizer(" ".join(sentence[ev.offset]), padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                start = ev.offset.start + 1

                final_output = torch.cat([final_output, pre_event], dim=0).type(torch.LongTensor)

                s = len(final_output)
                final_output = torch.cat([final_output, event], dim=0).type(torch.LongTensor)
                e = len(final_output)

                ev_end_emb = self.tokenizer(ev_end, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                final_output = torch.cat([final_output, ev_end_emb], dim=0).type(torch.LongTensor)

                if id_ev+1 == len(event_list):
                    post_event =  self.tokenizer(" ".join(sentence[ev.offset.stop:]) + " ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                    final_output =  torch.cat([final_output, post_event], dim=0)

                assert re.sub(r"\s", r"", self.tokenizer.decode(final_output[s:e])) == re.sub(r"\s", r"", ev.token)
                spans.append((s, e))

        e_inst_emb = self.tokenizer(E_INST, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        final_output = torch.cat([final_output, e_inst_emb])

        for se in spans:
            mask = torch.zeros(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(-math.inf)
            mask[se[0]: se[1]] = 1
            complementary_mask[se[0]: se[1]] = 0

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)


class TextPromptPrompter(Prompter):
    ev1_sub = "<ev1>"
    ev2_sub = "<ev2>"
    classes_sub = "<classes>"
    context_sub = "<context>"
    examples_sub = "<examples>"
    ev1_q_sub = "event1"
    ev2_q_sub = "event2"

    def __init__(
        self,
        prompt: str,
        classes: List[str],
        tokenizer,
        truncate_limit: Optional[int] = None,
        examples: List[Tuple[Document, Event, Event, str]] = None,
        rel_type_order: List[str] = None,
        highlight_events: str = "in_sentence",
        questions: Optional[Dict[str, str]] = None,
        prompt_type: str = "class",
        ev_in_question: bool = False
    ):
        super(TextPromptPrompter, self).__init__(tokenizer, truncate_limit=truncate_limit)
        self.prompt = prompt
        self.classes = classes
        self.classes_str = "{" + ", ".join([label for label in self.classes if label != "VAGUE"]) + "}"
        self.examples = examples
        self.example_embs: torch.Tensor = None
        self.highlight_events = highlight_events
        self.questions = questions
        self.prompt_type = prompt_type
        self.ev_in_question = ev_in_question
        self.rel_type_order = rel_type_order
        if self.examples is not None and len(self.examples) > 0:
            assert self.rel_type_order is not None
            self._prepare_examples()
    @override
    def _make_prompt(
        self,
        final_output: dict,
        spans: list,
        masks: list,
        complementary_masks: list,
    ) -> Prompt:
        return Prompt(final_output, spans, masks, complementary_masks)

    def _get_start_context_emb(self) -> str:
        embeddings = ""
        if self.ev_in_question and self.prompt_type != "class":
            embeddings = "Given the context: "
        return embeddings

    def _get_end_context_emb(self) -> str:
        embeddings = ""
        if self.ev_in_question and self.prompt_type != "class":
            end_context = " Answer the question: "
            if self.prompt_type == "questions_sequential":
                end_context = " Answer the questions: "
            embeddings = end_context
        return embeddings

    def _get_context_emb_ev_in_sentence(self, doc: Document, e1: Event, e2: Event) -> str:
        # <context with marked events>
        event_list = [e1, e2]
        # embeddings = torch.LongTensor([])
        context_emb = self._get_start_context_emb()
        embeddings = ""
        if e1.sent_id != e2.sent_id:
            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event =  " ".join(doc.text[ev.sent_id][0:ev.offset.start])
                event =  ev_start + " ".join(doc.text[ev.sent_id][ev.offset]) + ev_end
                post_event =  " ".join(doc.text[ev.sent_id][ev.offset.stop:]) + " "
                embeddings = "".join([embeddings, pre_event, event, post_event])
        else:
            sentence = doc.text[e1.sent_id]
            start = 0

            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event =   " ".join(sentence[start:ev.offset.start])
                event =  ev_start + " ".join(doc.text[ev.sent_id][ev.offset]) + ev_end
                start = ev.offset.stop

                if id_ev+1 == len(event_list):
                    post_event = " ".join(sentence[ev.offset.stop:]) + " "
                    embeddings =  "".join([embeddings, pre_event, event, post_event])
                else:
                    embeddings = "".join([embeddings, pre_event, event])


        end_emb = self._get_end_context_emb()
        embeddings = "".join([context_emb, embeddings, end_emb])
        return embeddings

    def _get_context_emb_ev_end(self, doc: Document, e1: Event, e2: Event) -> str:
        # s1 s2 [event1]: e1 [event2]: e2
        embeddings = ""

        # context
        context = " ".join(doc.text[e1.sent_id])
        if e1.sent_id != e2.sent_id:
            second_sentence = " ".join(doc.text[e2.sent_id])
            context = "".join([context, second_sentence])
        embeddings = "".join([embeddings, context])

        # events
        event_list = [e1, e2]
        events = torch.LongTensor([])
        for i, ev in enumerate(event_list):
            event =  f" [event{i+1}]: " + " ".join(doc.text[ev.sent_id][ev.offset])
            events = "".join([events, event])
        embeddings = "".join([embeddings, events])

        return embeddings

    def _get_context_emb(self, doc: Document, e1: Event, e2: Event) -> str:
        if self.highlight_events == "in_sentence":
            embeddings = self._get_context_emb_ev_in_sentence(doc, e1, e2)
        elif self.highlight_events == "end":
            embeddings = self._get_context_emb_ev_end(doc, e1, e2)
        else:
            raise Exception(f"No strategy for highlight_events = '{self.highlight_events}'")
        return embeddings

    def _prepare_examples_class(self, label, context_emb):
        example_embs = []
        arrow_emb = " ->"
        label_emb = f" {label} \n"
        example = "".join([context_emb, arrow_emb, label_emb])
        example_embs.append(example)
        if self.example_embs is None:
            self.example_embs = "".join(example_embs)
        else:
            self.example_embs = "".join([self.example_embs] + example_embs)

    def _prepare_question(self, question: str, e1: Event, e2: Event) -> str:
        if self.ev_in_question:
            question = re.sub(r"event1", f"[event1] {e1.token} [/event1]", question)
            question = re.sub(r"event2", f"[event2] {e2.token} [/event2]", question)
        return question

    def _prepare_examples_questions_single(self, label, context_emb, e1: Event, e2: Event):
        example_embs = {q_label: [] for q_label in self.rel_type_order}
        for q_label in self.rel_type_order:
            question = self.questions[q_label]
            answer = "YES" if label == q_label else "NO"
            question = self._prepare_question(question, e1, e2)
            q_emb = f" {question}"
            a_emb = f" {answer} \n"
            label_emb = "".join([q_emb, a_emb])
            example = "".join([context_emb, label_emb])
            example_embs[q_label].append(example)
        if self.example_embs is None:
            self.example_embs = {q_label: None for q_label in self.rel_type_order}
            self.example_embs = {
                q_label: "".join(example_embs[q_label])
                for q_label, q_ex_emb in self.example_embs.items()
            }
        else:
            self.example_embs = {
                q_label: "".join([q_ex_emb] + example_embs[q_label])
                for q_label, q_ex_emb in self.example_embs.items()
            }

    def _prepare_examples_questions_all(self, label, context_emb, e1: Event, e2: Event):
        example_embs = []
        for q_label in self.rel_type_order:
            question = self.questions[q_label]
            answer = "YES" if label == q_label else "NO"
            question = self._prepare_question(question, e1, e2)
            q_emb = f" {question}"
            a_emb = f" {answer} \n"
            label_emb = "".join([q_emb, a_emb])
            example = "".join([context_emb, label_emb])
            example_embs.append(example)
        if self.example_embs is None:
            self.example_embs = "".join(example_embs)
        else:
            self.example_embs = "".join([self.example_embs] + example_embs)

    def _prepare_examples_questions_seq(self, label, context_emb, e1: Event, e2: Event):
        # context q1 a1 q2 a2 ...
        example_embs = [context_emb]
        for q_label in self.rel_type_order:
            question = self.questions[q_label]
            answer = "YES" if label == q_label else "NO"
            question = self._prepare_question(question, e1, e2)
            q_emb = f" {question}"
            a_emb = f" {answer} "
            label_emb = "".join([q_emb, a_emb])
            example_embs.append(label_emb)
        newline_emb = f" \n"
        example_embs.append(newline_emb)
        if self.example_embs is None:
            self.example_embs = "".join(example_embs)
        else:
            self.example_embs = "".join([self.example_embs] + example_embs)

    def _prepare_examples(self):
        assert self.rel_type_order is not None
        # <example> <label> \n <example> <label> \n ...
        for doc, e1, e2, label in self.examples:
            context_emb = self._get_context_emb(doc, e1, e2)
            # <label> vs <question> <YES/NO>
            if self.prompt_type == "class":
                self._prepare_examples_class(label, context_emb)
            elif self.prompt_type == "questions_single":
                self._prepare_examples_questions_single(label, context_emb, e1, e2)
            elif self.prompt_type == "questions_all":
                self._prepare_examples_questions_all(label, context_emb, e1, e2)
            elif self.prompt_type == "questions_sequential":
                self._prepare_examples_questions_seq(label, context_emb, e1, e2)

    def few_shot(self, doc, e1, e2) -> Prompt:
        final_output = torch.LongTensor([])
        final_output = {}
        if self.example_embs is not None:
            final_output["examples"] = self.example_embs
        context_emb = self._get_context_emb(doc, e1, e2)
        arrow_emb = f" -> "
        final_output["context"] = "".join([context_emb, arrow_emb])

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def few_shot_prompt(self, doc, e1, e2) -> Prompt:
        prompt = self.few_shot(doc, e1, e2)
        sys_prompt_str = self.prompt
        prompt.embeddings["sys_prompt"] = sys_prompt_str
        return prompt

    def _get_few_shot_questions_examples_emb(self, doc, e1, e2, q_rel_type) -> str:
        final_output = ""
        if self.example_embs is not None:
            if self.prompt_type == "questions_single":
                example_embs = self.example_embs[q_rel_type]
            else: # if self.prompt_type == "questions_all":
                example_embs = self.example_embs
            final_output = "".join([final_output, example_embs])
        return final_output

    def _get_few_shot_questions_question_emb(self, e1, e2, q_rel_type) -> torch.Tensor:
        question = self.questions[q_rel_type]
        question = self._prepare_question(question, e1, e2)
        question_emb = f" {question} "
        return question_emb

    def few_shot_questions(self, doc, e1, e2, q_rel_type) -> Prompt:
        examples_emb = self._get_few_shot_questions_examples_emb(doc, e1, e2, q_rel_type)
        context_emb = self._get_context_emb(doc, e1, e2)
        question_emb = self._get_few_shot_questions_question_emb(e1, e2, q_rel_type)

        final_output = {}
        final_output["examples"] = examples_emb
        final_output["context"] = "".join([context_emb, question_emb])

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def few_shot_questions_prompt(self, doc, e1, e2, q_rel_type) -> Prompt:
        examples_emb = self._get_few_shot_questions_examples_emb(doc, e1, e2)
        context_emb = self._get_context_emb(doc, e1, e2)
        question_emb = self._get_few_shot_questions_question_emb(e1, e2, q_rel_type)


        final_output = torch.LongTensor([])
        if self.tokenizer.bos_token_id is not None:
            final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        # sys_prompt_str = B_INST + B_SYS + self.prompt + E_SYS
        sys_prompt_str = B_INST + B_SYS + self.prompt + " \n"
        sys_prompt_embeddings = self.tokenizer(sys_prompt_str, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e_sys_emb = self.tokenizer(E_SYS, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e_inst_emb = self.tokenizer(E_INST + " ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # [1:] -> remove old bos token
        # [:-1] -> ignore last \n in examples
        assert self.tokenizer.decode(examples_emb[-1]) == "\n"
        final_output = torch.cat([final_output, sys_prompt_embeddings, examples_emb[1:-1], e_sys_emb, context_emb, question_emb, e_inst_emb])

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def few_shot_questions_seq(self, doc, e1, e2, include_examples=True) -> dict:
        final_output = {}
        if self.example_embs is not None and include_examples:
            final_output["examples"] = self.example_embs
        final_output["context"] = self._get_context_emb(doc, e1, e2)

        question_embs = {}
        for q_rel_type in self.rel_type_order:
            question = self.questions[q_rel_type]
            question = self._prepare_question(question, e1, e2)
            question_embs[q_rel_type] = f" {question} "

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(1)
            complementary_mask = torch.FloatTensor(1).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        prompt = {
            "input": self._make_prompt(final_output, spans, masks, complementary_masks),
            "questions": question_embs,
        }
        return prompt

    def few_shot_questions_seq_prompt(self, doc, e1, e2) -> dict:
        prompt = self.few_shot_questions_seq(doc, e1, e2, include_examples=False)
        questions = prompt["questions"]
        prompt = prompt["input"]

        final_output = {}
        if self.example_embs is not None:
            final_output["examples"] = self.example_embs
        final_output["context"] = prompt.embeddings["context"]

        sys_prompt_str = self.prompt
        final_output["sys_prompt"] = sys_prompt_str

        return {
            "input": self._make_prompt(final_output, prompt.spans, prompt.mask, prompt.complementary_mask),
            "questions": questions
        }


class TempDataset(torch.utils.data.Dataset):
    def __init__(self, documents, prompt_function, lab_dict=None, rows=None, skip_vague=True):
        if rows is None:
            self.rows = []
            self.prompter = prompt_function
            if lab_dict == None:
                self.dict_rel = {}
            else:
                self.dict_rel = {k: v for k, v in lab_dict.items()}
            for doc in documents:
                d_id = doc.d_id
                for rel_type, rels in doc.temporal_relations.items():
                    # if rel_type != "VAGUE":
                    if not (rel_type == "VAGUE" and skip_vague):
                        if rel_type not in self.dict_rel:
                            self.dict_rel[rel_type] = len(self.dict_rel)
                        for rel in rels:
                            event1 = rel.event1
                            event2 = rel.event2
                            self._add_example(d_id, rel_type, doc, rel.event1, rel.event2)
        else:
            self.rows = rows
            self.prompter = prompt_function
            self.dict_rel = lab_dict

    def _add_example(self, d_id: str, rel_type: str, doc: Document, e1: Event, e2: Event):
        self.rows.append(
            DataRow(
                d_id,
                rel_type,
                self.dict_rel[rel_type],
                doc,
                e1,
                e2,
                self.prompter(doc, e1, e2),
                None,
            )
        )

    def __getitem__(self, idx) -> DataRow:
        item = self.rows[idx]
        return item

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def stratified_sample(cls, dataset: "TempDataset", sample_size: float = 0.2) -> "TempDataset":
        labels = [row.label for row in dataset.rows]
        sampled_rows, _, sampled_labels, _ = train_test_split(dataset.rows, labels, train_size=sample_size)
        return TempDataset(None, dataset.prompter, rows=sampled_rows)

class TempQuestionsDataset(TempDataset):
    def __init__(self, documents, prompt_function, lab_dict=None, rows=None, skip_vague=True):
        super(TempQuestionsDataset, self).__init__(documents, prompt_function, lab_dict=lab_dict, rows=rows, skip_vague=skip_vague)

    @override
    def _add_example(self, d_id: str, rel_type: str, doc: Document, e1: Event, e2: Event):
        for q_rel_type in list(self.dict_rel.keys()):
            if q_rel_type == "VAGUE":
                # no explicit question for vague
                continue
            comb_rel_type = f"{rel_type}__{q_rel_type}"
            self.rows.append(
                DataRow(
                    d_id,
                    comb_rel_type,
                    self.dict_rel[rel_type],
                    doc,
                    e1,
                    e2,
                    self.prompter(doc, e1, e2, q_rel_type),
                    None,
                )
            )

def collate_fn(data, tokenizer, device, pad_left=False, questions_seq=False, text_emb=False) -> dict:
    new_item = {"input_ids": [], "attention_mask": [], "labels": [], "data":data}

    labels = []
    inp = []
    questions = []

    comp_masks = []
    masks = []
    max_len = 0
    to_duplicate = []
    events2 = []

    for el_id, elem in enumerate(data):
        labels.append(elem.id_label)
        if not questions_seq:
            inp.append(elem.model_input.embeddings)
            masks.extend(elem.model_input.mask)
            comp_masks.extend(elem.model_input.complementary_mask)
            to_duplicate.append(len(elem.model_input.mask))

            if len(elem.model_input.embeddings) > max_len:
                max_len = len(elem.model_input.embeddings)
        else:
            # also assume batch size 1
            inp.append(elem.model_input["input"].embeddings)
            q_add = {id:q for id, q in elem.model_input["questions"].items()}
            first_key = list(q_add.keys())[0]
            if not text_emb:
                q_add = {id:q.to(device) for id, q in q_add.items()}
            questions.append(q_add)
            masks.extend(elem.model_input["input"].mask)
            comp_masks.extend(elem.model_input["input"].complementary_mask)

            to_duplicate.append(len(elem.model_input["input"].mask))

            if len(elem.model_input["input"].embeddings) > max_len:
                max_len = len(elem.model_input["input"].embeddings)

    padded_input = torch.LongTensor(len(inp),max_len).fill_(tokenizer.pad_token_id)
    attention_masks = torch.LongTensor(len(inp),max_len).fill_(0)

    padded_masks = torch.LongTensor(len(masks),max_len).fill_(0)
    padded_comp_mask = torch.FloatTensor(len(masks), max_len).fill_(-math.inf)


    if not text_emb:
        if not pad_left:
            for i, seq in enumerate(inp):
                end = len(seq)
                padded_input[i, :end] = seq
                attention_masks[i, :end] = 1

            for i, seq in enumerate(masks):
                end = len(seq)
                padded_masks[i, :end] = seq
                assert 0 in comp_masks[i]
                padded_comp_mask[i, :end] = comp_masks[i]
        else:
            for i, seq in enumerate(inp):
                start = max_len - len(seq)
                padded_input[i, start:] = seq
                attention_masks[i, start:] = 1

            for i, seq in enumerate(masks):
                start = max_len - len(seq)
                padded_masks[i, start:] = seq
                assert 0 in comp_masks[i]
                padded_comp_mask[i, start:] = comp_masks[i]
    else:
        padded_input = inp

    if not questions_seq:
        new_item["input_ids"] = padded_input.to(device) if not text_emb else padded_input
    else:
        new_item["input_ids"] = padded_input.to(device) if not text_emb else padded_input
        # no padding, assume batch_size == 1
        new_item["questions_ids"] = questions

    new_item['attention_mask'] = attention_masks.to(device)

    new_item['masks'] = padded_masks.to(device)
    new_item['complementary_mask'] = padded_comp_mask.to(device)

    new_item['labels'] = torch.LongTensor(labels)
    new_item['to_duplicate'] = torch.LongTensor(to_duplicate).to(device)
    new_item['data_rows'] = data

    return new_item

def fine_tune_collate_fn(data: List[DataRow], tokenizer, device, rel_type_order: List[str], prompt_type) -> dict:
    new_item = {"input_ids": [], "attention_mask": [], "labels": [], "data": data}

    labels = []
    inp = []

    comp_masks = []
    masks = []
    max_len = 0
    to_duplicate = []

    for el_id, elem in enumerate(data):
        labels.append(elem.id_label)
        if prompt_type != "questions_sequential":
            input = elem.model_input.embeddings
            if "questions" in prompt_type:
                rel_type, q_rel_type = elem.label.split("__")
                if rel_type == q_rel_type:
                    a_str = "YES"
                else:
                    a_str = "NO"
            else:
                a_str = elem.label
            a_emb = tokenizer(f" {a_str} ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            input = torch.cat([input, a_emb], dim=0).type(torch.LongTensor)
            inp.append(input)
            masks.extend([torch.ones(input.shape) for _ in range(len(elem.model_input.mask))])
            comp_masks.extend([torch.zeros(input.shape) for _ in range(len(elem.model_input.mask))])
            to_duplicate.append(len(elem.model_input.mask))

            if len(input) > max_len:
                max_len = len(input)
        else:
            input = elem.model_input["input"].embeddings
            for q_rel_type in rel_type_order:
                q_emb = elem.model_input["questions"][q_rel_type]
                if q_rel_type == elem.label:
                    a_str = "YES"
                else:
                    a_str = "NO"
                a_emb = tokenizer(f" {a_str} ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                input = torch.cat([input, q_emb, a_emb], dim=0).type(torch.LongTensor)
            inp.append(input)
            masks.extend([torch.ones(input.shape) for _ in range(len(elem.model_input["input"].mask))])
            comp_masks.extend([torch.zeros(input.shape) for _ in range(len(elem.model_input["input"].mask))])

            to_duplicate.append(len(elem.model_input["input"].mask))

            if len(input) > max_len:
                max_len = len(input)

    padded_input = torch.LongTensor(len(inp),max_len).fill_(tokenizer.pad_token_id)
    attention_masks = torch.LongTensor(len(inp),max_len).fill_(0)

    padded_masks = torch.LongTensor(len(masks),max_len).fill_(0)
    padded_comp_mask = torch.FloatTensor(len(masks), max_len).fill_(-math.inf)


    for i, seq in enumerate(inp):
        end = len(seq)
        padded_input[i, :end] = seq
        attention_masks[i, :end] = 1

    for i, seq in enumerate(masks):
        end = len(seq)
        padded_masks[i, :end] = seq
        assert 0 in comp_masks[i]
        padded_comp_mask[i, :end] = comp_masks[i]

    new_item["input_ids"] = padded_input.to(device)

    new_item['attention_mask'] = attention_masks.to(device)

    new_item['masks'] = padded_masks.to(device)
    new_item['complementary_mask'] = padded_comp_mask.to(device)

    new_item['labels'] = torch.LongTensor(labels)
    new_item['to_duplicate'] = torch.LongTensor(to_duplicate).to(device)
    new_item['data_rows'] = data

    return new_item

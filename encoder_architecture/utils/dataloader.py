from typing import List, Optional, Tuple
from typing_extensions import override
from sandbox.data_structures import Document, Event
import torch
from dataclasses import dataclass
import math
from tqdm import tqdm

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


@dataclass
class DataRowFrozen:
    d_id: str
    label: str
    id_label: int
    context: str
    event1: Event
    event2: Event
    model_input: Prompt
    prediction: str
    embeddings: torch.Tensor


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
        pass
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

            # print(sentence, e1.token, e2.token)
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


class PromptPrompter(Prompter):
    ev1_sub = "<ev1>"
    ev2_sub = "<ev2>"
    classes_sub = "<classes>"
    context_sub = "<context>"
    examples_sub = "<examples>"

    def __init__(
        self,
        prompt: str,
        classes: List[str],
        tokenizer,
        truncate_limit: Optional[int] = None,
        examples: List[Tuple[Document, Event, Event, str]] = None,
    ):
        super(PromptPrompter, self).__init__(tokenizer, truncate_limit=truncate_limit)
        self.prompt = prompt
        self.classes = classes
        self.classes_str = (
            "{" + ", ".join([label for label in self.classes if label != "VAGUE"]) + "}"
        )
        self.examples = examples
        self.example_embs: torch.Tensor = None
        if self.examples is not None and len(self.examples) > 0:
            self._prepare_examples()

    def _get_context_emb(self, doc: Document, e1: Event, e2: Event) -> torch.Tensor:
        # TODO [SEP] between sentences
        event_list = [e1, e2]
        embeddings = torch.LongTensor([])
        if e1.sent_id != e2.sent_id:
            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event = self.tokenizer(
                    " ".join(doc.text[ev.sent_id][0 : ev.offset.start]) + ev_start,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]

                event = self.tokenizer(
                    " ".join(doc.text[ev.sent_id][ev.offset]),
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                post_event = self.tokenizer(
                    ev_end + " ".join(doc.text[ev.sent_id][ev.offset.stop :]) + " ",
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]

                embeddings = torch.cat([embeddings, pre_event], dim=0).type(
                    torch.LongTensor
                )
                s = len(embeddings)
                embeddings = torch.cat([embeddings, event], dim=0).type(
                    torch.LongTensor
                )
                # e = len(embeddings) - 1 # We remove the extra space we added
                e = len(embeddings)
                embeddings = torch.cat([embeddings, post_event], dim=0).type(
                    torch.LongTensor
                )

                try:
                    assert re.sub(
                        r"\s", r"", self.tokenizer.decode(embeddings[s:e])
                    ) == re.sub(r"\s", r"", ev.token)
                except:
                    print(f"decoded: {self.tokenizer.decode(embeddings[s:e])}")
                    print(f"token: '{ev.token}'")
                    import pdb

                    pdb.set_trace()
                # assert self.tokenizer.decode(embeddings[s:e]) == ev.token

                # spans.append((s, e))
        else:
            sentence = doc.text[e1.sent_id]
            start = 0

            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event = self.tokenizer(
                    " ".join(sentence[start : ev.offset.start]) + ev_start,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                event = self.tokenizer(
                    " ".join(sentence[ev.offset]),
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                start = ev.offset.start + 1

                embeddings = torch.cat([embeddings, pre_event], dim=0).type(
                    torch.LongTensor
                )

                s = len(embeddings)
                embeddings = torch.cat([embeddings, event], dim=0).type(
                    torch.LongTensor
                )
                # e = len(embeddings) - 1 # We remove the extra space we added
                e = len(embeddings)

                ev_end_emb = self.tokenizer(
                    ev_end, padding=False, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                embeddings = torch.cat([embeddings, ev_end_emb], dim=0).type(
                    torch.LongTensor
                )

                if id_ev + 1 == len(event_list):
                    post_event = self.tokenizer(
                        " ".join(sentence[ev.offset.stop :]) + " ",
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                    embeddings = torch.cat([embeddings, post_event], dim=0)

                try:
                    assert re.sub(
                        r"\s", r"", self.tokenizer.decode(embeddings[s:e])
                    ) == re.sub(r"\s", r"", ev.token)
                except:
                    print(f"decoded: '{self.tokenizer.decode(embeddings[s:e])}'")
                    print(f"token: '{ev.token}'")
                    import pdb

                    pdb.set_trace()
                # assert self.tokenizer.decode(embeddings[s:e]) == ev.token
                # spans.append((s, e))

        arrow_emb = self.tokenizer(
            f" -> ", padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        embeddings = torch.cat([embeddings, arrow_emb], dim=0).type(torch.LongTensor)
        return embeddings

    def _prepare_examples(self):
        example_embs = []
        newline_emb = self.tokenizer(
            "\n", padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        for i, (doc, e1, e2, label) in enumerate(self.examples):
            context_emb = self._get_context_emb(doc, e1, e2)
            label_emb = self.tokenizer(
                f" {label} \n",
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            if i > 0:
                example = torch.cat([context_emb, label_emb], dim=0).type(
                    torch.LongTensor
                )
            else:
                example = torch.cat([newline_emb, context_emb, label_emb], dim=0).type(
                    torch.LongTensor
                )
            example_embs.append(example)
        self.example_embs = torch.cat(example_embs + [newline_emb], dim=0).type(
            torch.LongTensor
        )

    def few_shot(self, doc, e1, e2) -> Prompt:
        final_output = torch.LongTensor([self.tokenizer.bos_token_id])
        if self.prompt:
            pass
        if self.example_embs is not None:
            final_output = torch.cat([final_output, self.example_embs], dim=0).type(
                torch.LongTensor
            )
        context_emb = self._get_context_emb(doc, e1, e2)
        final_output = torch.cat([final_output, context_emb], dim=0).type(
            torch.LongTensor
        )

        masks, complementary_masks = [], []
        spans = []
        # 2 events
        for i in range(2):
            mask = torch.ones(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)

            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)

    def system_prompt(self, doc, e1, e2) -> Prompt:
        # TODO [SEP] between sentences
        # As https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/generation.py#L44
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        spans = []
        masks = []
        complementary_masks = []
        event_list = [e1, e2]
        special_tokens = [" [event1]: ", " [event2]: "]

        final_output = torch.LongTensor([self.tokenizer.bos_token_id])

        # <s> [INST] <<SYS>>\n system prompt <</SYS>>\n\n user prompt [/INST] model answer </s>
        prompt_str = B_INST + B_SYS + self.prompt + E_SYS
        prompt_str = re.sub(PromptPrompter.classes_sub, self.classes_str, prompt_str)
        prompt_embeddings = self.tokenizer(
            prompt_str, padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        final_output = torch.cat([final_output, prompt_embeddings])
        # b_inst_emb = self.tokenizer(" " + B_INST + " ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # final_output = torch.cat([final_output, b_inst_emb])

        if e1.sent_id != e2.sent_id:
            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event = self.tokenizer(
                    " ".join(doc.text[ev.sent_id][0 : ev.offset.start]) + ev_start,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]

                event = self.tokenizer(
                    " ".join(doc.text[ev.sent_id][ev.offset]),
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                post_event = self.tokenizer(
                    ev_end + " ".join(doc.text[ev.sent_id][ev.offset.stop :]) + " ",
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]

                final_output = torch.cat([final_output, pre_event], dim=0).type(
                    torch.LongTensor
                )
                s = len(final_output)
                final_output = torch.cat([final_output, event], dim=0).type(
                    torch.LongTensor
                )
                # e = len(final_output) - 1 # We remove the extra space we added
                e = len(final_output)
                final_output = torch.cat([final_output, post_event], dim=0).type(
                    torch.LongTensor
                )

                try:
                    assert re.sub(
                        r"\s", r"", self.tokenizer.decode(final_output[s:e])
                    ) == re.sub(r"\s", r"", ev.token)
                except:
                    print(f"decoded: {self.tokenizer.decode(final_output[s:e])}")
                    print(f"token: '{ev.token}'")
                    import pdb

                    pdb.set_trace()
                # assert self.tokenizer.decode(final_output[s:e]) == ev.token

                spans.append((s, e))
        else:
            sentence = doc.text[e1.sent_id]
            context = " ".join(sentence)
            start = 0

            for id_ev, ev in enumerate(event_list):
                ev_start = f" [event{id_ev+1}] "
                ev_end = f" [/event{id_ev+1}] "
                pre_event = self.tokenizer(
                    " ".join(sentence[start : ev.offset.start]) + ev_start,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                event = self.tokenizer(
                    " ".join(sentence[ev.offset]),
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"][0]
                start = ev.offset.start + 1

                final_output = torch.cat([final_output, pre_event], dim=0).type(
                    torch.LongTensor
                )

                s = len(final_output)
                final_output = torch.cat([final_output, event], dim=0).type(
                    torch.LongTensor
                )
                # e = len(final_output) - 1 # We remove the extra space we added
                e = len(final_output)

                ev_end_emb = self.tokenizer(
                    ev_end, padding=False, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                final_output = torch.cat([final_output, ev_end_emb], dim=0).type(
                    torch.LongTensor
                )

                if id_ev + 1 == len(event_list):
                    post_event = self.tokenizer(
                        " ".join(sentence[ev.offset.stop :]) + " ",
                        padding=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"][0]
                    final_output = torch.cat([final_output, post_event], dim=0)

                try:
                    assert re.sub(
                        r"\s", r"", self.tokenizer.decode(final_output[s:e])
                    ) == re.sub(r"\s", r"", ev.token)
                except:
                    print(f"decoded: '{self.tokenizer.decode(final_output[s:e])}'")
                    print(f"token: '{ev.token}'")
                    import pdb

                    pdb.set_trace()
                # assert self.tokenizer.decode(final_output[s:e]) == ev.token
                spans.append((s, e))

        e_inst_emb = self.tokenizer(
            E_INST, padding=False, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        final_output = torch.cat([final_output, e_inst_emb])

        for se in spans:
            mask = torch.zeros(len(final_output))
            complementary_mask = torch.FloatTensor(len(final_output)).fill_(-math.inf)
            mask[se[0] : se[1]] = 1
            complementary_mask[se[0] : se[1]] = 0
            masks.append(mask.clone())
            complementary_masks.append(complementary_mask.clone())

        # masks, complementary_masks = [], []
        # # 2 events
        # for i in range(2):
        #     mask = torch.ones(len(final_output))
        #     # complementary_mask = torch.FloatTensor(len(final_output)).fill_(-math.inf)
        #     complementary_mask = torch.FloatTensor(len(final_output)).fill_(0)
        #     # mask[se[0]: se[1]] = 1 # We multiply this to isolate the event embeddings
        #     # complementary_mask[se[0]: se[1]] = 0 # We add this to not bias the maxpooling
        #
        #     masks.append(mask.clone())
        #     complementary_masks.append(complementary_mask.clone())

        return self._make_prompt(final_output, spans, masks, complementary_masks)


class TempDataset(torch.utils.data.Dataset):
    def __init__(self, documents, prompt_function, lab_dict=None, skip_vague=True):
        self.rows = []
        self.prompter = prompt_function
        if lab_dict == None:
            self.dict_rel = {}
        else:
            self.dict_rel = {k: v for k, v in lab_dict.items()}
            
        for doc in documents:
            d_id = doc.d_id
            for rel_type, rels in doc.temporal_relations.items():
                if rel_type != "VAGUE" or not skip_vague:
                    if rel_type not in self.dict_rel:
                        self.dict_rel[rel_type] = len(self.dict_rel)
                    for rel in rels:
                        event1 = rel.event1
                        event2 = rel.event2
                        self.rows.append(
                            DataRow(
                                d_id,
                                rel_type,
                                self.dict_rel[rel_type],
                                doc,
                                event1,
                                event2,
                                self.prompter(doc, event1, event2),
                                None,
                            )
                        )

    def __getitem__(self, idx) -> DataRow:
        item = self.rows[idx]
        return item

    def __len__(self) -> int:
        return len(self.rows)


class TempDatasetFrozen(torch.utils.data.Dataset):
    def __init__(self, documents, prompt_function, model, device, lab_dict=None, skip_vague=True):
        self.rows = []
        self.prompter = prompt_function
        model.eval()
        
        if lab_dict == None:
            self.dict_rel = {}
        else:
            self.dict_rel = {k: v for k, v in lab_dict.items()}
        for doc in tqdm(documents):
            d_id = doc.d_id
            for rel_type, rels in doc.temporal_relations.items():
                if rel_type != "VAGUE" or not skip_vague:
                    if rel_type not in self.dict_rel:
                        self.dict_rel[rel_type] = len(self.dict_rel)
                    for rel in rels:
                        event1 = rel.event1
                        event2 = rel.event2
                        prompt = self.prompter(doc, event1, event2)
                        attention_mask = torch.LongTensor(
                            1, len(prompt.embeddings)
                        ).fill_(1)
                        max_len = len(prompt.embeddings)
                        assert max_len > 2
                        padded_masks = torch.LongTensor(len(prompt.mask), max_len).fill_(0)
                        padded_comp_mask = torch.FloatTensor(len(prompt.mask), max_len).fill_(-math.inf)
                        for i, seq in enumerate(prompt.mask):
                            end = len(seq)
                            padded_masks[i, :end] = seq
                            assert 0 in prompt.complementary_mask[i]
                            padded_comp_mask[i, :end] = prompt.complementary_mask[i]
                        with torch.no_grad():
                            logits = model(
                                prompt.embeddings.unsqueeze(0).to(device),
                                attention_mask.to(device),
                                padded_masks.to(device),
                                padded_comp_mask.to(device),
                                2,
                            )
                        self.rows.append(
                            DataRowFrozen(
                                d_id,
                                rel_type,
                                self.dict_rel[rel_type],
                                doc,
                                event1,
                                event2,
                                self.prompter(doc, event1, event2),
                                None,
                                logits,
                            )
                        )

    def __getitem__(self, idx) -> DataRow:
        item = self.rows[idx]
        return item

    def __len__(self) -> int:
        return len(self.rows)


def collate_fn_frozen(data, device, pad_left=False) -> dict:
    new_item = {"input_ids": [], "attention_mask": [], "labels": [], "data": data}

    labels = []
    inp = []

    for el_id, elem in enumerate(data):
        labels.append(elem.id_label)
        inp.append(elem.embeddings)
  
    new_item["input_ids"] = torch.stack(inp).to(device)
    new_item["labels"] = torch.LongTensor(labels)
   
    # Retro comaptibility
    new_item["to_duplicate"] = None
    new_item["attention_mask"] = None

    new_item["masks"] = None
    new_item["complementary_mask"] = None

    new_item["data_rows"] = [
        DataRow(
            d.d_id,
            d.label,
            d.id_label,
            d.context,
            d.event1,
            d.event2,
            d.model_input,
            d.prediction,
        )
        for d in data
    ]
    return new_item


def collate_fn(data, tokenizer, device, pad_left=False) -> dict:
    new_item = {"input_ids": [], "attention_mask": [], "labels": [], "data": data}

    labels = []
    inp = []

    comp_masks = []
    masks = []
    max_len = 0
    to_duplicate = []

    for el_id, elem in enumerate(data):
        labels.append(elem.id_label)
        inp.append(elem.model_input.embeddings)
        masks.extend(elem.model_input.mask)
        comp_masks.extend(elem.model_input.complementary_mask)

        to_duplicate.append(len(elem.model_input.mask))

        if len(elem.model_input.embeddings) > max_len:
            max_len = len(elem.model_input.embeddings)

    padded_input = torch.LongTensor(len(inp), max_len).fill_(tokenizer.pad_token_id)
    attention_masks = torch.LongTensor(len(inp), max_len).fill_(0)

    padded_masks = torch.LongTensor(len(masks), max_len).fill_(0)
    padded_comp_mask = torch.FloatTensor(len(masks), max_len).fill_(-math.inf)

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

    # input_matrix
    # mask_matrix

    new_item["input_ids"] = padded_input.to(device)
    new_item["attention_mask"] = attention_masks.to(device)

    new_item["masks"] = padded_masks.to(device)
    new_item["complementary_mask"] = padded_comp_mask.to(device)

    new_item["labels"] = torch.LongTensor(labels)
    new_item["to_duplicate"] = torch.LongTensor(to_duplicate).to(device)
    new_item["data_rows"] = data

    return new_item



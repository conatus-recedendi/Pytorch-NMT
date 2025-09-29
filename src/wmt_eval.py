#!/usr/bin/env python3
"""
Python implementation of WMT evaluation script (wmt_eval.perl)
MT evaluation scorer supporting both BLEU and NIST metrics
"""

import sys
import os
import argparse
import math
import re
import time
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import xml.etree.ElementTree as ET


class MTEvaluator:
    def __init__(
        self,
        ref_file: str,
        src_file: str,
        tst_file: str,
        detail: int = 0,
        preserve_case: bool = False,
        split_non_ascii: bool = False,
        brevity_penalty: str = "closest",
        international_tokenization: bool = False,
        metrics_matr_output: bool = False,
        no_smoothing: bool = False,
        method: str = "BOTH",
    ):
        self.ref_file = ref_file
        self.src_file = src_file
        self.tst_file = tst_file
        self.detail = detail
        self.preserve_case = preserve_case
        self.split_non_ascii = split_non_ascii
        self.brevity_penalty = brevity_penalty
        self.international_tokenization = international_tokenization
        self.metrics_matr_output = metrics_matr_output
        self.no_smoothing = no_smoothing
        self.method = method

        self.max_ngram = 9

        # Global variables
        self.src_lang = None
        self.tgt_lang = None
        self.tst_sys = []
        self.ref_sys = []
        self.tst_data = {}
        self.ref_data = {}
        self.src_id = None
        self.ref_id = None
        self.tst_id = None
        self.eval_docs = {}
        self.ngram_info = {}

        # Select functions based on options
        self.bleu_bp_func = (
            self.brevity_penalty_closest
            if brevity_penalty == "closest"
            else self.brevity_penalty_shortest
        )
        self.tokenization_func = (
            self.tokenization_international
            if international_tokenization
            else self.tokenization
        )
        self.bleu_score_func = (
            self.bleu_score_nosmoothing if no_smoothing else self.bleu_score
        )

    def date_time_stamp(self) -> Tuple[str, str]:
        """Get current date and time stamp"""
        now = datetime.now()
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        date = f"{now.year} {months[now.month-1]} {now.day}"
        time = f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
        return date, time

    def extract_sgml_tag_and_span(
        self, name: str, data: str
    ) -> Optional[Tuple[str, str, str]]:
        """Extract SGML tag and its content"""
        pattern = rf"<{name}\s*([^>]*)>(.*?)</{name}\s*>(.*)"
        match = re.search(pattern, data, re.IGNORECASE | re.DOTALL)
        return match.groups() if match else None

    def extract_sgml_tag_attribute(self, name: str, data: str) -> Optional[str]:
        """Extract SGML tag attribute value"""
        pattern = rf"{name}\s*=\s*\"([^\"]*)\""
        match = re.search(pattern, data, re.IGNORECASE)
        return match.group(1) if match else None

    def tokenization(self, norm_text: str) -> str:
        """Default tokenization function"""
        # Language-independent part
        norm_text = re.sub(r"<skipped>", "", norm_text)
        norm_text = re.sub(r"-\n", "", norm_text)
        norm_text = re.sub(r"\n", " ", norm_text)
        norm_text = re.sub(r"&quot;", '"', norm_text)
        norm_text = re.sub(r"&amp;", "&", norm_text)
        norm_text = re.sub(r"&lt;", "<", norm_text)
        norm_text = re.sub(r"&gt;", ">", norm_text)

        # Language-dependent part (Western languages)
        norm_text = f" {norm_text} "
        if not self.preserve_case:
            norm_text = norm_text.lower()

        # Tokenize punctuation
        norm_text = re.sub(r"([\{\-\~\[\-\` \-\&\(\-\+\:\-\@\/])", r" \1 ", norm_text)
        norm_text = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", norm_text)
        norm_text = re.sub(r"([\.,])([^0-9])", r" \1 \2", norm_text)
        norm_text = re.sub(r"([0-9])(-)", r"\1 \2 ", norm_text)
        norm_text = re.sub(r"\s+", " ", norm_text)
        norm_text = norm_text.strip()

        return norm_text

    def tokenization_international(self, norm_text: str) -> str:
        """International tokenization using Unicode categories"""
        norm_text = re.sub(r"<skipped>", "", norm_text)
        norm_text = re.sub(r"\p{Hyphen}\p{Zl}", "", norm_text)
        norm_text = re.sub(r"\p{Zl}", " ", norm_text)

        # Replace entities
        norm_text = re.sub(r"&quot;", '"', norm_text)
        norm_text = re.sub(r"&amp;", "&", norm_text)
        norm_text = re.sub(r"&lt;", "<", norm_text)
        norm_text = re.sub(r"&gt;", ">", norm_text)
        norm_text = re.sub(r"&apos;", "'", norm_text)

        if not self.preserve_case:
            norm_text = norm_text.lower()

        if self.split_non_ascii:
            norm_text = re.sub(r"([^\x00-\x7F])", r" \1 ", norm_text)

        # Tokenize punctuation unless preceded AND followed by digits
        norm_text = re.sub(r"(\P{N})(\p{P})", r"\1 \2 ", norm_text)
        norm_text = re.sub(r"(\p{P})(\P{N})", r" \1 \2", norm_text)
        norm_text = re.sub(r"(\p{S})", r" \1 ", norm_text)

        norm_text = re.sub(r"\s+", " ", norm_text)
        norm_text = norm_text.strip()

        return norm_text

    def get_source_info(self, file_path: str) -> str:
        """Parse source file and extract document information"""
        if file_path.lower().endswith(".xml"):
            return self._parse_xml_source(file_path)
        else:
            return self._parse_sgml_source(file_path)

    def _parse_xml_source(self, file_path: str) -> str:
        """Parse XML source file"""
        tree = ET.parse(file_path)
        root = tree.getroot()

        srcset = root.find("srcset")
        if srcset is None:
            raise ValueError(
                f"Source XML file '{file_path}' does not contain 'srcset' element"
            )

        set_id = srcset.get("setid")
        if not set_id:
            raise ValueError(f"No 'setid' attribute value in '{file_path}'")

        src_lang = srcset.get("srclang")
        if not src_lang:
            raise ValueError(f"No srcset 'srclang' attribute value in '{file_path}'")

        if self.src_lang is not None and src_lang != self.src_lang:
            raise ValueError("Not the same srclang attribute values across sets")
        self.src_lang = src_lang

        for doc_elem in srcset.findall(".//doc"):
            doc_id = doc_elem.get("docid")
            if not doc_id:
                raise ValueError(
                    f"No document 'docid' attribute value in '{file_path}'"
                )

            self.eval_docs[doc_id] = {"SEGS": {}}

            for seg_elem in doc_elem.findall(".//seg"):
                seg_id = seg_elem.get("id")
                if not seg_id:
                    raise ValueError(
                        f"No segment 'id' attribute value in '{file_path}'"
                    )

                seg_data = seg_elem.text or ""
                self.eval_docs[doc_id]["SEGS"][seg_id] = self.tokenization_func(
                    seg_data
                )

        return set_id

    def _parse_sgml_source(self, file_path: str) -> str:
        """Parse SGML source file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        # Extract source set info
        result = self.extract_sgml_tag_and_span("SrcSet", data)
        if not result:
            raise ValueError(
                f"FATAL INPUT ERROR: no 'src_set' tag in src_file '{file_path}'"
            )

        tag, span, remaining_data = result

        set_id = self.extract_sgml_tag_attribute("SetID", tag)
        if not set_id:
            raise ValueError(
                f"FATAL INPUT ERROR: no tag attribute 'SetID' in file '{file_path}'"
            )

        src_lang = self.extract_sgml_tag_attribute("SrcLang", tag)
        if not src_lang:
            raise ValueError(
                f"FATAL INPUT ERROR: no tag attribute 'SrcLang' in file '{file_path}'"
            )

        if self.src_lang is not None and src_lang != self.src_lang:
            raise ValueError(
                f"FATAL INPUT ERROR: SrcLang ('{src_lang}') in file '{file_path}' inconsistent"
            )
        self.src_lang = src_lang

        # Parse documents
        data = span
        while True:
            result = self.extract_sgml_tag_and_span("Doc", data)
            if not result:
                break

            tag, span, data = result

            doc_id = self.extract_sgml_tag_attribute("DocID", tag)
            if not doc_id:
                raise ValueError(
                    f"FATAL INPUT ERROR: no tag attribute 'DocID' in file '{file_path}'"
                )

            if doc_id in self.eval_docs:
                raise ValueError(
                    f"FATAL INPUT ERROR: duplicate 'DocID' in file '{file_path}'"
                )

            self.eval_docs[doc_id] = {"SEGS": {}}

            # Clean up span
            span = re.sub(r"[\s\n\r]+", " ", span)

            # Parse segments
            seg_data = span
            while True:
                seg_result = self.extract_sgml_tag_and_span("Seg", seg_data)
                if not seg_result:
                    break

                seg_tag, seg_span, seg_data = seg_result

                seg_id = self.extract_sgml_tag_attribute("id", seg_tag)
                if not seg_id:
                    raise ValueError(
                        f"FATAL INPUT ERROR: no attribute 'id' in file '{file_path}'"
                    )

                self.eval_docs[doc_id]["SEGS"][seg_id] = self.tokenization_func(
                    seg_span
                )

            if not self.eval_docs[doc_id]["SEGS"]:
                raise ValueError(
                    f"FATAL INPUT ERROR: no segments in document '{doc_id}' in file '{file_path}'"
                )

        if not self.eval_docs:
            raise ValueError(f"FATAL INPUT ERROR: no documents in file '{file_path}'")

        return set_id

    def get_mt_data(self, docs: Dict, set_tag: str, file_path: str) -> str:
        """Parse MT data (reference or test) file"""
        if file_path.lower().endswith(".xml"):
            return self._parse_xml_mt_data(docs, file_path)
        else:
            return self._parse_sgml_mt_data(docs, set_tag, file_path)

    def _parse_xml_mt_data(self, docs: Dict, file_path: str) -> str:
        """Parse XML MT data file"""
        tree = ET.parse(file_path)
        root = tree.getroot()

        set_id = None

        for current_set in root.findall("refset") + root.findall("tstset"):
            set_id = current_set.get("setid")
            if not set_id:
                raise ValueError(f"No 'setid' attribute value in '{file_path}'")

            src_lang = current_set.get("srclang")
            if not src_lang:
                raise ValueError(f"No 'srclang' attribute value in '{file_path}'")

            tgt_lang = current_set.get("trglang")
            if not tgt_lang:
                raise ValueError(f"No 'trglang' attribute value in '{file_path}'")

            if src_lang != self.src_lang:
                raise ValueError("Not the same 'srclang' attribute value across sets")

            if self.tgt_lang is not None and tgt_lang != self.tgt_lang:
                raise ValueError("Not the same 'trglang' attribute value across sets")
            self.tgt_lang = tgt_lang

            if current_set.tag == "tstset":
                sys_id = current_set.get("sysid")
                if not sys_id:
                    raise ValueError(f"No 'sysid' attribute value in '{file_path}'")
            else:
                sys_id = current_set.get("refid")
                if not sys_id:
                    raise ValueError(f"No 'refid' attribute value in '{file_path}'")

            docs[sys_id] = {}

            for doc_elem in current_set.findall(".//doc"):
                doc_id = doc_elem.get("docid")
                if not doc_id:
                    raise ValueError(
                        f"No document 'docid' attribute value in '{file_path}'"
                    )

                docs[sys_id][doc_id] = {"FILE": file_path, "SEGS": {}}

                for seg_elem in doc_elem.findall(".//seg"):
                    seg_id = seg_elem.get("id")
                    if not seg_id:
                        raise ValueError(
                            f"No segment 'id' attribute value in '{file_path}'"
                        )

                    seg_data = seg_elem.text or ""
                    docs[sys_id][doc_id]["SEGS"][seg_id] = self.tokenization_func(
                        seg_data
                    )

        return set_id

    def _parse_sgml_mt_data(self, docs: Dict, set_tag: str, file_path: str) -> str:
        """Parse SGML MT data file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        set_id = None

        while True:
            result = self.extract_sgml_tag_and_span(set_tag, data)
            if not result:
                break

            tag, span, data = result

            set_id = self.extract_sgml_tag_attribute("SetID", tag)
            if not set_id:
                raise ValueError(
                    f"FATAL INPUT ERROR: no tag attribute 'SetID' in file '{file_path}'"
                )

            src_lang = self.extract_sgml_tag_attribute("SrcLang", tag)
            if not src_lang:
                raise ValueError(
                    f"FATAL INPUT ERROR: no tag attribute 'SrcLang' in file '{file_path}'"
                )

            if src_lang != self.src_lang:
                raise ValueError(
                    f"FATAL INPUT ERROR: SrcLang ('{src_lang}') in file '{file_path}' inconsistent"
                )

            tgt_lang = self.extract_sgml_tag_attribute("TrgLang", tag)
            if not tgt_lang:
                raise ValueError(
                    f"FATAL INPUT ERROR: no tag attribute 'TrgLang' in file '{file_path}'"
                )

            if self.tgt_lang is not None and tgt_lang != self.tgt_lang:
                raise ValueError(
                    f"FATAL INPUT ERROR: TrgLang ('{tgt_lang}') in file '{file_path}' inconsistent"
                )
            self.tgt_lang = tgt_lang

            # Parse documents within this set
            mt_data = span
            while True:
                doc_result = self.extract_sgml_tag_and_span("Doc", mt_data)
                if not doc_result:
                    break

                doc_tag, doc_span, mt_data = doc_result

                sys_id = self.extract_sgml_tag_attribute("SysID", doc_tag)
                if not sys_id:
                    raise ValueError(
                        f"FATAL INPUT ERROR: no tag attribute 'SysID' in file '{file_path}'"
                    )

                doc_id = self.extract_sgml_tag_attribute("DocID", doc_tag)
                if not doc_id:
                    raise ValueError(
                        f"FATAL INPUT ERROR: no tag attribute 'DocID' in file '{file_path}'"
                    )

                if sys_id in docs and doc_id in docs[sys_id]:
                    raise ValueError(
                        f"FATAL INPUT ERROR: document '{doc_id}' for system '{sys_id}' in file '{file_path}' previously loaded"
                    )

                if sys_id not in docs:
                    docs[sys_id] = {}

                docs[sys_id][doc_id] = {"FILE": file_path, "SEGS": {}}

                # Clean up span
                doc_span = re.sub(r"[\s\n\r]+", " ", doc_span)

                # Parse segments
                seg_data = doc_span
                while True:
                    seg_result = self.extract_sgml_tag_and_span("Seg", seg_data)
                    if not seg_result:
                        break

                    seg_tag, seg_span, seg_data = seg_result

                    seg_id = self.extract_sgml_tag_attribute("id", seg_tag)
                    if not seg_id:
                        raise ValueError(
                            f"FATAL INPUT ERROR: no tag attribute 'id' in file '{file_path}'"
                        )

                    docs[sys_id][doc_id]["SEGS"][seg_id] = self.tokenization_func(
                        seg_span
                    )

                if not docs[sys_id][doc_id]["SEGS"]:
                    raise ValueError(
                        f"FATAL INPUT ERROR: no segments in document '{doc_id}' in file '{file_path}'"
                    )

        return set_id

    def check_mt_data(self):
        """Check MT data for completeness and correctness"""
        self.tst_sys = sorted(self.tst_data.keys())
        self.ref_sys = sorted(self.ref_data.keys())

        if not (self.src_id == self.tst_id == self.ref_id):
            raise ValueError("Not the same 'setid' attribute values across files")

        # Check that every evaluation document is represented for every system and reference
        for doc_id in sorted(self.eval_docs.keys()):
            nseg_source = len(self.eval_docs[doc_id]["SEGS"])

            for sys_id in self.tst_sys:
                if doc_id not in self.tst_data[sys_id]:
                    raise ValueError(
                        f"FATAL ERROR: no document '{doc_id}' for system '{sys_id}'"
                    )

                nseg = len(self.tst_data[sys_id][doc_id]["SEGS"])
                if nseg != nseg_source:
                    raise ValueError(
                        f"FATAL ERROR: translated documents must contain the same # of segments as the source"
                    )

            for sys_id in self.ref_sys:
                if doc_id not in self.ref_data[sys_id]:
                    raise ValueError(
                        f"FATAL ERROR: no document '{doc_id}' for reference '{sys_id}'"
                    )

                nseg = len(self.ref_data[sys_id][doc_id]["SEGS"])
                if nseg != nseg_source:
                    raise ValueError(
                        f"FATAL ERROR: translated documents must contain the same # of segments as the source"
                    )

    def words_to_ngrams(self, words: List[str]) -> Dict[str, int]:
        """Convert list of words to n-gram count dictionary"""
        count = defaultdict(int)

        for i in range(len(words)):
            for j in range(min(self.max_ngram, len(words) - i)):
                ngram = " ".join(words[i : i + j + 1])
                count[ngram] += 1

        return dict(count)

    def compute_ngram_info(self):
        """Compute n-gram information for NIST scoring"""
        ngram_count = defaultdict(int)
        tot_wrds = 0

        for ref_id in self.ref_data:
            for doc_id in self.ref_data[ref_id]:
                for seg_id in self.ref_data[ref_id][doc_id]["SEGS"]:
                    words = self.ref_data[ref_id][doc_id]["SEGS"][seg_id].split()
                    tot_wrds += len(words)
                    ngrams = self.words_to_ngrams(words)
                    for ngram, count in ngrams.items():
                        ngram_count[ngram] += count

        for ngram, count in ngram_count.items():
            words = ngram.split()
            if len(words) > 1:
                mgram = " ".join(words[:-1])
                mgram_count = ngram_count.get(mgram, 0)
                self.ngram_info[ngram] = -math.log2(
                    count / mgram_count if mgram_count > 0 else count / tot_wrds
                )
            else:
                self.ngram_info[ngram] = -math.log2(count / tot_wrds)

    def brevity_penalty_shortest(
        self, current_length: int, reference_length: int, candidate_length: int
    ) -> int:
        """Return shortest reference length"""
        return min(reference_length, current_length)

    def brevity_penalty_closest(
        self, current_length: int, reference_length: int, candidate_length: int
    ) -> int:
        """Return closest reference length to candidate"""
        if abs(candidate_length - reference_length) <= abs(
            candidate_length - current_length
        ):
            if abs(candidate_length - reference_length) == abs(
                candidate_length - current_length
            ):
                return min(current_length, reference_length)
            else:
                return reference_length
        return current_length

    def score_segment(self, tst_seg: str, ref_segs: List[str]) -> Tuple:
        """Score individual segment"""
        match_count = [0] * (self.max_ngram + 1)
        tst_count = [0] * (self.max_ngram + 1)
        ref_count = [0] * (self.max_ngram + 1)
        tst_info = [0.0] * (self.max_ngram + 1)
        ref_info = [0.0] * (self.max_ngram + 1)

        # Get n-gram counts for test segment
        tst_words = tst_seg.split()
        tst_ngrams = self.words_to_ngrams(tst_words)

        for j in range(1, self.max_ngram + 1):
            tst_count[j] = max(0, len(tst_words) - j + 1) if j <= len(tst_words) else 0

        # Get n-gram counts for reference segments
        ref_ngrams_max = {}
        ref_length = None

        for ref_seg in ref_segs:
            ref_words = ref_seg.split()
            ref_ngrams = self.words_to_ngrams(ref_words)

            for ngram, count in ref_ngrams.items():
                words = ngram.split()
                if ngram in self.ngram_info:
                    ref_info[len(words)] += self.ngram_info[ngram]
                ref_ngrams_max[ngram] = max(ref_ngrams_max.get(ngram, 0), count)

            for j in range(1, self.max_ngram + 1):
                ref_count[j] += (
                    max(0, len(ref_words) - j + 1) if j <= len(ref_words) else 0
                )

            if ref_length is None:
                ref_length = len(ref_words)
            else:
                ref_length = self.bleu_bp_func(
                    ref_length, len(ref_words), len(tst_words)
                )

        # Accumulate scoring stats for matching n-grams
        for ngram, count in tst_ngrams.items():
            if ngram in ref_ngrams_max:
                words = ngram.split()
                match_count[len(words)] += min(count, ref_ngrams_max[ngram])
                if ngram in self.ngram_info:
                    tst_info[len(words)] += self.ngram_info[ngram] * min(
                        count, ref_ngrams_max[ngram]
                    )

        return ref_length, match_count, tst_count, ref_count, tst_info, ref_info

    def bleu_score_nosmoothing(
        self,
        ref_length: int,
        matching_ngrams: List[int],
        tst_ngrams: List[int],
        sys: str,
        score_mt: Dict,
    ) -> float:
        """Calculate BLEU score without smoothing"""
        score = 0.0

        for j in range(1, self.max_ngram + 1):
            if matching_ngrams[j] == 0:
                score_mt[j][sys]["cum"] = 0
            else:
                len_score = (
                    min(0, 1 - ref_length / tst_ngrams[1]) if tst_ngrams[1] > 0 else 0
                )
                score += (
                    math.log(matching_ngrams[j] / tst_ngrams[j])
                    if tst_ngrams[j] > 0
                    else 0
                )
                score_mt[j][sys]["cum"] = math.exp(score / j + len_score)

                iscore = (
                    math.log(matching_ngrams[j] / tst_ngrams[j])
                    if tst_ngrams[j] > 0
                    else 0
                )
                score_mt[j][sys]["ind"] = math.exp(iscore)

        return score_mt[4][sys]["cum"]

    def bleu_score(
        self,
        ref_length: int,
        matching_ngrams: List[int],
        tst_ngrams: List[int],
        sys: str,
        score_mt: Dict,
    ) -> float:
        """Calculate BLEU score with smoothing"""
        score = 0.0
        exp_len_score = (
            math.exp(min(0, 1 - ref_length / tst_ngrams[1])) if tst_ngrams[1] > 0 else 0
        )
        smooth = 1

        for j in range(1, self.max_ngram + 1):
            if tst_ngrams[j] == 0:
                iscore = 0
            elif matching_ngrams[j] == 0:
                smooth *= 2
                iscore = math.log(1 / (smooth * tst_ngrams[j]))
            else:
                iscore = math.log(matching_ngrams[j] / tst_ngrams[j])

            score_mt[j][sys]["ind"] = math.exp(iscore)
            score += iscore
            score_mt[j][sys]["cum"] = math.exp(score / j) * exp_len_score

        return score_mt[4][sys]["cum"]

    def nist_length_penalty(self, ratio: float) -> float:
        """Calculate NIST length penalty"""
        if ratio >= 1:
            return 1
        if ratio <= 0:
            return 0

        ratio_x = 1.5
        score_x = 0.5
        beta = -math.log(score_x) / (math.log(ratio_x) * math.log(ratio_x))
        return math.exp(-beta * math.log(ratio) * math.log(ratio))

    def nist_score(
        self,
        nsys: int,
        matching_ngrams: List[int],
        tst_ngrams: List[int],
        ref_ngrams: List[int],
        tst_info: List[float],
        ref_info: List[float],
        sys: str,
        score_mt: Dict,
    ) -> float:
        """Calculate NIST score"""
        score = 0.0

        for n in range(1, self.max_ngram + 1):
            score += tst_info[n] / max(tst_ngrams[n], 1)
            length_penalty = (
                self.nist_length_penalty(tst_ngrams[1] / (ref_ngrams[1] / nsys))
                if ref_ngrams[1] > 0
                else 0
            )
            score_mt[n][sys]["cum"] = score * length_penalty

            iscore = tst_info[n] / max(tst_ngrams[n], 1)
            score_mt[n][sys]["ind"] = iscore * length_penalty

        return score_mt[5][sys]["cum"]

    def score_document(self, sys: str, doc: str, overall_score: Dict) -> Tuple:
        """Score document by aggregating segment scores"""
        cum_ref_length = 0
        cum_match = [0] * (self.max_ngram + 1)
        cum_tst_cnt = [0] * (self.max_ngram + 1)
        cum_ref_cnt = [0] * (self.max_ngram + 1)
        cum_tst_info = [0.0] * (self.max_ngram + 1)
        cum_ref_info = [0.0] * (self.max_ngram + 1)

        for seg_id in sorted(self.tst_data[sys][doc]["SEGS"].keys(), key=int):
            ref_segments = []
            for ref_id in self.ref_sys:
                ref_segments.append(self.ref_data[ref_id][doc]["SEGS"][seg_id])
                if self.detail >= 3:
                    print(
                        f"ref '{ref_id}', seg {seg_id}: {self.ref_data[ref_id][doc]['SEGS'][seg_id]}"
                    )

            if self.detail >= 3:
                print(
                    f"sys '{sys}', seg {seg_id}: {self.tst_data[sys][doc]['SEGS'][seg_id]}"
                )

            ref_length, match_cnt, tst_cnt, ref_cnt, tst_info, ref_info = (
                self.score_segment(
                    self.tst_data[sys][doc]["SEGS"][seg_id], ref_segments
                )
            )

            # Calculate segment-level scores
            if self.method in ["BLEU", "BOTH"]:
                doc_mt = defaultdict(lambda: defaultdict(dict))
                seg_score = self.bleu_score_func(
                    ref_length, match_cnt, tst_cnt, sys, doc_mt
                )

                if "documents" not in overall_score[sys]:
                    overall_score[sys]["documents"] = {}
                if doc not in overall_score[sys]["documents"]:
                    overall_score[sys]["documents"][doc] = {"segments": {}}

                overall_score[sys]["documents"][doc]["segments"][seg_id] = {
                    "score": seg_score
                }

                if self.detail >= 2:
                    print(
                        f'  BLEU score using 4-grams = {seg_score:.4f} for system "{sys}" on segment {seg_id} of document "{doc}" ({tst_cnt[1]} words)'
                    )

            if self.method in ["NIST", "BOTH"]:
                doc_mt = defaultdict(lambda: defaultdict(dict))
                seg_score = self.nist_score(
                    len(self.ref_sys),
                    match_cnt,
                    tst_cnt,
                    ref_cnt,
                    tst_info,
                    ref_info,
                    sys,
                    doc_mt,
                )

                if "documents" not in overall_score[sys]:
                    overall_score[sys]["documents"] = {}
                if doc not in overall_score[sys]["documents"]:
                    overall_score[sys]["documents"][doc] = {"segments": {}}

                overall_score[sys]["documents"][doc]["segments"][seg_id] = {
                    "score": seg_score
                }

                if self.detail >= 2:
                    print(
                        f'  NIST score using 5-grams = {seg_score:.4f} for system "{sys}" on segment {seg_id} of document "{doc}" ({tst_cnt[1]} words)'
                    )

            cum_ref_length += ref_length
            for j in range(1, self.max_ngram + 1):
                cum_match[j] += match_cnt[j]
                cum_tst_cnt[j] += tst_cnt[j]
                cum_ref_cnt[j] += ref_cnt[j]
                cum_tst_info[j] += tst_info[j]
                cum_ref_info[j] += ref_info[j]

        return (
            cum_ref_length,
            cum_match,
            cum_tst_cnt,
            cum_ref_cnt,
            cum_tst_info,
            cum_ref_info,
        )

    def score_system(self, sys: str, score_mt: Dict, overall_score: Dict):
        """Score system by aggregating document scores"""
        cum_ref_length = 0
        cum_match = [0] * (self.max_ngram + 1)
        cum_tst_cnt = [0] * (self.max_ngram + 1)
        cum_ref_cnt = [0] * (self.max_ngram + 1)
        cum_tst_info = [0.0] * (self.max_ngram + 1)
        cum_ref_info = [0.0] * (self.max_ngram + 1)

        for doc in sorted(self.eval_docs.keys()):
            ref_length, match_cnt, tst_cnt, ref_cnt, tst_info, ref_info = (
                self.score_document(sys, doc, overall_score)
            )

            # Calculate document-level scores
            if self.method == "NIST":
                doc_mt = defaultdict(lambda: defaultdict(dict))
                doc_score = self.nist_score(
                    len(self.ref_sys),
                    match_cnt,
                    tst_cnt,
                    ref_cnt,
                    tst_info,
                    ref_info,
                    sys,
                    doc_mt,
                )
                overall_score[sys]["documents"][doc]["score"] = doc_score

                if self.detail >= 1:
                    print(
                        f"NIST score using   5-grams = {doc_score:.4f} for system \"{sys}\" on document \"{doc}\" ({len(self.tst_data[sys][doc]['SEGS'])} segments, {tst_cnt[1]} words)"
                    )

            if self.method == "BLEU":
                doc_mt = defaultdict(lambda: defaultdict(dict))
                doc_score = self.bleu_score_func(
                    ref_length, match_cnt, tst_cnt, sys, doc_mt
                )
                overall_score[sys]["documents"][doc]["score"] = doc_score

                if self.detail >= 1:
                    print(
                        f"BLEU score using   4-grams = {doc_score:.4f} for system \"{sys}\" on document \"{doc}\" ({len(self.tst_data[sys][doc]['SEGS'])} segments, {tst_cnt[1]} words)"
                    )

            cum_ref_length += ref_length
            for j in range(1, self.max_ngram + 1):
                cum_match[j] += match_cnt[j]
                cum_tst_cnt[j] += tst_cnt[j]
                cum_ref_cnt[j] += ref_cnt[j]
                cum_tst_info[j] += tst_info[j]
                cum_ref_info[j] += ref_info[j]

        # Calculate system-level scores
        if self.method == "BLEU":
            overall_score[sys]["score"] = self.bleu_score_func(
                cum_ref_length, cum_match, cum_tst_cnt, sys, score_mt
            )

        if self.method == "NIST":
            overall_score[sys]["score"] = self.nist_score(
                len(self.ref_sys),
                cum_match,
                cum_tst_cnt,
                cum_ref_cnt,
                cum_tst_info,
                cum_ref_info,
                sys,
                score_mt,
            )

    def printout_report(self, nist_mt: Dict, bleu_mt: Dict):
        """Print evaluation report"""
        if self.method == "BOTH":
            for sys in sorted(self.tst_sys):
                print(
                    f"NIST score = {nist_mt[5][sys]['cum']:.4f}  BLEU score = {bleu_mt[4][sys]['cum']:.4f} for system \"{sys}\""
                )
        elif self.method == "NIST":
            for sys in sorted(self.tst_sys):
                print(f"NIST score = {nist_mt[5][sys]['cum']:.4f} for system \"{sys}\"")
        elif self.method == "BLEU":
            for sys in sorted(self.tst_sys):
                print(
                    f"\nBLEU score = {bleu_mt[4][sys]['cum']:.4f} for system \"{sys}\""
                )

        print(
            "\n# ------------------------------------------------------------------------\n"
        )
        print("Individual N-gram scoring")
        print(
            "        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram"
        )
        print(
            "        ------   ------   ------   ------   ------   ------   ------   ------   ------"
        )

        if self.method in ["BOTH", "NIST"]:
            for sys in sorted(self.tst_sys):
                print(f" NIST:", end="")
                for i in range(1, self.max_ngram + 1):
                    print(f"  {nist_mt[i][sys]['ind']:.4f} ", end="")
                print(f' "{sys}"')
            print()

        if self.method in ["BOTH", "BLEU"]:
            for sys in sorted(self.tst_sys):
                print(f" BLEU:", end="")
                for i in range(1, self.max_ngram + 1):
                    print(f"  {bleu_mt[i][sys]['ind']:.4f} ", end="")
                print(f' "{sys}"')

        print(
            "\n# ------------------------------------------------------------------------"
        )
        print("Cumulative N-gram scoring")
        print(
            "        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram"
        )
        print(
            "        ------   ------   ------   ------   ------   ------   ------   ------   ------"
        )

        if self.method in ["BOTH", "NIST"]:
            for sys in sorted(self.tst_sys):
                print(f" NIST:", end="")
                for i in range(1, self.max_ngram + 1):
                    print(f"  {nist_mt[i][sys]['cum']:.4f} ", end="")
                print(f' "{sys}"')
        print()

        if self.method in ["BOTH", "BLEU"]:
            for sys in sorted(self.tst_sys):
                print(f" BLEU:", end="")
                for i in range(1, self.max_ngram + 1):
                    print(f"  {bleu_mt[i][sys]['cum']:.4f} ", end="")
                print(f' "{sys}"')

    def output_metrics_matr(self, prefix: str, overall: Dict):
        """Output MetricsMATR files"""
        file_name_sys = f"{prefix}-sys.scr"
        file_name_doc = f"{prefix}-doc.scr"
        file_name_seg = f"{prefix}-seg.scr"

        with open(file_name_sys, "w") as f_sys, open(file_name_doc, "w") as f_doc, open(
            file_name_seg, "w"
        ) as f_seg:

            for sys in sorted(overall.keys()):
                score_sys = overall[sys]["score"]
                f_sys.write(f"{self.tst_id}\t{sys}\t{score_sys}\n")

                for doc in sorted(overall[sys]["documents"].keys()):
                    score_doc = overall[sys]["documents"][doc]["score"]
                    f_doc.write(f"{self.tst_id}\t{sys}\t{doc}\t{score_doc}\n")

                    for seg in sorted(
                        overall[sys]["documents"][doc]["segments"].keys(), key=int
                    ):
                        score_seg = overall[sys]["documents"][doc]["segments"][seg][
                            "score"
                        ]
                        f_seg.write(
                            f"{self.tst_id}\t{sys}\t{doc}\t{seg}\t{score_seg}\n"
                        )

    def evaluate(self):
        """Main evaluation function"""
        date, time = self.date_time_stamp()
        print(f"MT evaluation scorer began on {date} at {time}")
        print(f"command line: {' '.join(sys.argv)}")

        # Get source document IDs
        self.src_id = self.get_source_info(self.src_file)

        # Get reference translations
        self.ref_id = self.get_mt_data(self.ref_data, "RefSet", self.ref_file)

        # Compute n-gram information
        self.compute_ngram_info()

        # Get translations to evaluate
        self.tst_id = self.get_mt_data(self.tst_data, "TstSet", self.tst_file)

        # Check data for completeness and correctness
        self.check_mt_data()

        # Initialize scoring dictionaries
        nist_mt = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        nist_overall = defaultdict(dict)
        bleu_mt = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        bleu_overall = defaultdict(dict)

        # Evaluate
        print(f"  Evaluation of {self.src_lang}-to-{self.tgt_lang} translation using:")
        cum_seg = sum(len(self.eval_docs[doc]["SEGS"]) for doc in self.eval_docs)
        print(
            f'    src set "{self.src_id}" ({len(self.eval_docs)} docs, {cum_seg} segs)'
        )
        print(f'    ref set "{self.ref_id}" ({len(self.ref_data)} refs)')
        print(f'    tst set "{self.tst_id}" ({len(self.tst_data)} systems)\n')

        for sys in sorted(self.tst_sys):
            for n in range(1, self.max_ngram + 1):
                nist_mt[n][sys]["cum"] = 0
                nist_mt[n][sys]["ind"] = 0
                bleu_mt[n][sys]["cum"] = 0
                bleu_mt[n][sys]["ind"] = 0

            if self.method in ["BOTH", "NIST"]:
                self.method = "NIST"
                self.score_system(sys, nist_mt, nist_overall)

            if self.method in ["BOTH", "BLEU"]:
                self.method = "BLEU"
                self.score_system(sys, bleu_mt, bleu_overall)

        # Print report
        self.printout_report(nist_mt, bleu_mt)

        # Output MetricsMATR files if requested
        if self.metrics_matr_output:
            if self.method in ["BOTH", "NIST"]:
                self.output_metrics_matr("NIST", nist_overall)
            if self.method in ["BOTH", "BLEU"]:
                self.output_metrics_matr("BLEU", bleu_overall)

        date, time = self.date_time_stamp()
        print(f"MT evaluation scorer ended on {date} at {time}")


def main():
    parser = argparse.ArgumentParser(
        description="MT evaluation scorer for BLEU and NIST metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "-r", "--ref-file", required=True, help="Reference translations file"
    )
    parser.add_argument("-s", "--src-file", required=True, help="Source documents file")
    parser.add_argument(
        "-t", "--tst-file", required=True, help="Test translations file"
    )

    # Optional arguments
    parser.add_argument(
        "-d",
        "--detail",
        type=int,
        default=0,
        help="Detail level: 0=system, 1=document, 2=segment, 3=ngram",
    )
    parser.add_argument(
        "-c",
        "--preserve-case",
        action="store_true",
        help="Preserve upper-case characters",
    )
    parser.add_argument(
        "-b", "--bleu-only", action="store_true", help="Generate BLEU scores only"
    )
    parser.add_argument(
        "-n", "--nist-only", action="store_true", help="Generate NIST scores only"
    )
    parser.add_argument(
        "-e",
        "--split-non-ascii",
        action="store_true",
        help="Enclose non-ASCII characters between spaces",
    )
    parser.add_argument(
        "--brevity-penalty",
        choices=["closest", "shortest"],
        default="closest",
        help="Brevity penalty method",
    )
    parser.add_argument(
        "--international-tokenization",
        action="store_true",
        help="Use Unicode-based tokenization",
    )
    parser.add_argument(
        "--metricsMATR-output",
        action="store_true",
        help="Create MetricsMATR output files",
    )
    parser.add_argument(
        "--no-smoothing", action="store_true", help="Disable BLEU score smoothing"
    )

    args = parser.parse_args()

    # Determine method
    method = "BOTH"
    if args.bleu_only:
        method = "BLEU"
    elif args.nist_only:
        method = "NIST"

    # Create evaluator and run
    evaluator = MTEvaluator(
        ref_file=args.ref_file,
        src_file=args.src_file,
        tst_file=args.tst_file,
        detail=args.detail,
        preserve_case=args.preserve_case,
        split_non_ascii=args.split_non_ascii,
        brevity_penalty=args.brevity_penalty,
        international_tokenization=args.international_tokenization,
        metrics_matr_output=args.metricsMATR_output,
        no_smoothing=args.no_smoothing,
        method=method,
    )

    try:
        evaluator.evaluate()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pandas as pd
import requests
from datasets import Dataset, load_dataset
from sacremoses import MosesDetokenizer
from tqdm import tqdm


class BaseDatasetAdapter(ABC):
    """Base dataset adapter class to standardise use of datasets."""

    OPTIONS = []  # must set

    def __init__(
        self,
        dataset_name,
        config_name=None,
        split="train",
        adapter_dict=None,
        load_dataset_options=None,
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.adapter_dict = adapter_dict
        if adapter_dict is None:
            self.prompt_options = self.OPTIONS
        else:
            if set(self.OPTIONS) != set(adapter_dict.values()):
                raise ValueError(
                    f"Adapter dict values must be the same as options: {', '.join(self.OPTIONS)}."
                )
            self.prompt_options = sorted(list(adapter_dict.keys()))
        if load_dataset_options is None:
            self.raw_data = load_dataset(dataset_name, config_name, split=split)
        else:
            self.raw_data = load_dataset(
                dataset_name, config_name, split=split, **load_dataset_options
            )
        self._formatted_data = None

    @property
    def dataset(self) -> Dataset:
        if not isinstance(self._formatted_data, Dataset):
            self._formatted_data = self._get_format_dataset()

        return self._formatted_data

    @abstractmethod
    def _get_format_dataset(self) -> Dataset:
        """Returns a Dataset with (at least) columns:
        ["id", "input", "options", "gold_label"].
        """
        pass


class Banking77Adapter(BaseDatasetAdapter):
    OPTIONS = [
        "activate_my_card",
        "age_limit",
        "apple_pay_or_google_pay",
        "atm_support",
        "automatic_top_up",
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "beneficiary_not_allowed",
        "cancel_transfer",
        "card_about_to_expire",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_not_working",
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "card_swallowed",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "change_pin",
        "compromised_card",
        "contactless_not_working",
        "country_support",
        "declined_card_payment",
        "declined_cash_withdrawal",
        "declined_transfer",
        "direct_debit_payment_not_recognised",
        "disposable_card_limits",
        "edit_personal_details",
        "exchange_charge",
        "exchange_rate",
        "exchange_via_app",
        "extra_charge_on_statement",
        "failed_transfer",
        "fiat_currency_support",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "order_physical_card",
        "passcode_forgotten",
        "pending_card_payment",
        "pending_cash_withdrawal",
        "pending_top_up",
        "pending_transfer",
        "pin_blocked",
        "receiving_money",
        "refund_not_showing_up",
        "request_refund",
        "reverted_card_payment",
        "supported_cards_and_currencies",
        "terminate_account",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_limits",
        "top_up_reverted",
        "topping_up_by_card",
        "transaction_charged_twice",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "verify_top_up",
        "virtual_card_not_working",
        "visa_or_mastercard",
        "why_verify_identity",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__(
            "PolyAI/banking77",
            "default",
            split=split,
            adapter_dict=adapter_dict,
            # load_dataset_options={"revision": "refs/convert/parquet"},
            load_dataset_options={"revision": "refs/pr/8"},
        )
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

        # assert that options match the cleaned version of the dataset labels
        assert self.OPTIONS == [
            name.rstrip("?").lower() for name in self.raw_data.features["label"].names
        ]

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": row["text"],
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["label"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class AGNewsAdapter(BaseDatasetAdapter):
    OPTIONS = ["World", "Sports", "Business", "Sci/Tech"]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__(
            "fancyzhx/ag_news", None, split=split, adapter_dict=adapter_dict
        )
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

        assert self.OPTIONS == self.raw_data.features["label"].names

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": row["text"],
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["label"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class SST5Adapter(BaseDatasetAdapter):
    OPTIONS = ["very negative", "negative", "neutral", "positive", "very positive"]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__("SetFit/sst5", None, split=split, adapter_dict=adapter_dict)
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

    def _process_dataset_row(self, row):
        # label is an int 0..4 (0=very negative ... 4=very positive)
        return {
            "id": row["id"],
            "input": row["text"],
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["label"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class YahooAdapter(BaseDatasetAdapter):
    OPTIONS = [
        "Society & Culture",
        "Science & Mathematics",
        "Health",
        "Education & Reference",
        "Computers & Internet",
        "Sports",
        "Business & Finance",
        "Entertainment & Music",
        "Family & Relationships",
        "Politics & Government",
    ]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__(
            "community-datasets/yahoo_answers_topics",
            None,
            split=split,
            adapter_dict=adapter_dict,
        )

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": f"Question title: {row['question_title']}\n\nQuestion content:\n{row['question_content']}\n\nBest answer:\n{row['best_answer']}",
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["topic"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class DBpediaAdapter(BaseDatasetAdapter):
    OPTIONS = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork",
    ]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__(
            "fancyzhx/dbpedia_14", "dbpedia_14", split=split, adapter_dict=adapter_dict
        )
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

        assert self.OPTIONS == self.raw_data.features["label"].names

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": " ".join([row["title"], row["content"]]),
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["label"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class TwentyNGAdapter(BaseDatasetAdapter):
    OPTIONS = [
        "alt.atheism",
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
        "misc.forsale",
        "rec.autos",
        "rec.motorcycles",
        "rec.sport.baseball",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "soc.religion.christian",
        "talk.politics.guns",
        "talk.politics.mideast",
        "talk.politics.misc",
        "talk.religion.misc",
    ]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__(
            "SetFit/20_newsgroups", None, split=split, adapter_dict=adapter_dict
        )
        # drop empty-text rows
        self.raw_data = self.raw_data.filter(lambda row: row["text"].strip() != "")
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": row["text"],
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["label"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class TREC50Adapter(BaseDatasetAdapter):
    OPTIONS = [
        "ABBR:abb",
        "ABBR:exp",
        "ENTY:animal",
        "ENTY:body",
        "ENTY:color",
        "ENTY:cremat",
        "ENTY:currency",
        "ENTY:dismed",
        "ENTY:event",
        "ENTY:food",
        "ENTY:instru",
        "ENTY:lang",
        "ENTY:letter",
        "ENTY:other",
        "ENTY:plant",
        "ENTY:product",
        "ENTY:religion",
        "ENTY:sport",
        "ENTY:substance",
        "ENTY:symbol",
        "ENTY:techmeth",
        "ENTY:termeq",
        "ENTY:veh",
        "ENTY:word",
        "DESC:def",
        "DESC:desc",
        "DESC:manner",
        "DESC:reason",
        "HUM:gr",
        "HUM:ind",
        "HUM:title",
        "HUM:desc",
        "LOC:city",
        "LOC:country",
        "LOC:mount",
        "LOC:other",
        "LOC:state",
        "NUM:code",
        "NUM:count",
        "NUM:date",
        "NUM:dist",
        "NUM:money",
        "NUM:ord",
        "NUM:other",
        "NUM:period",
        "NUM:perc",
        "NUM:speed",
        "NUM:temp",
        "NUM:volsize",
        "NUM:weight",
    ]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__("CogComp/trec", None, split=split, adapter_dict=adapter_dict)
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

        # fine_label is the 50-class task (coarse_label is the 6-class TREC-6)
        assert self.OPTIONS == self.raw_data.features["fine_label"].names

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": row["text"],
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["fine_label"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class CLINC150Adapter(BaseDatasetAdapter):
    OPTIONS = [
        "restaurant_reviews",
        "nutrition_info",
        "account_blocked",
        "oil_change_how",
        "time",
        "weather",
        "redeem_rewards",
        "interest_rate",
        "gas_type",
        "accept_reservations",
        "smart_home",
        "user_name",
        "report_lost_card",
        "repeat",
        "whisper_mode",
        "what_are_your_hobbies",
        "order",
        "jump_start",
        "schedule_meeting",
        "meeting_schedule",
        "freeze_account",
        "what_song",
        "meaning_of_life",
        "restaurant_reservation",
        "traffic",
        "make_call",
        "text",
        "bill_balance",
        "improve_credit_score",
        "change_language",
        "no",
        "measurement_conversion",
        "timer",
        "flip_coin",
        "do_you_have_pets",
        "balance",
        "tell_joke",
        "last_maintenance",
        "exchange_rate",
        "uber",
        "car_rental",
        "credit_limit",
        "oos",
        "shopping_list",
        "expiration_date",
        "routing",
        "meal_suggestion",
        "tire_change",
        "todo_list",
        "card_declined",
        "rewards_balance",
        "change_accent",
        "vaccines",
        "reminder_update",
        "food_last",
        "change_ai_name",
        "bill_due",
        "who_do_you_work_for",
        "share_location",
        "international_visa",
        "calendar",
        "translate",
        "carry_on",
        "book_flight",
        "insurance_change",
        "todo_list_update",
        "timezone",
        "cancel_reservation",
        "transactions",
        "credit_score",
        "report_fraud",
        "spending_history",
        "directions",
        "spelling",
        "insurance",
        "what_is_your_name",
        "reminder",
        "where_are_you_from",
        "distance",
        "payday",
        "flight_status",
        "find_phone",
        "greeting",
        "alarm",
        "order_status",
        "confirm_reservation",
        "cook_time",
        "damaged_card",
        "reset_settings",
        "pin_change",
        "replacement_card_duration",
        "new_card",
        "roll_dice",
        "income",
        "taxes",
        "date",
        "who_made_you",
        "pto_request",
        "tire_pressure",
        "how_old_are_you",
        "rollover_401k",
        "pto_request_status",
        "how_busy",
        "application_status",
        "recipe",
        "calendar_update",
        "play_music",
        "yes",
        "direct_deposit",
        "credit_limit_change",
        "gas",
        "pay_bill",
        "ingredients_list",
        "lost_luggage",
        "goodbye",
        "what_can_i_ask_you",
        "book_hotel",
        "are_you_a_bot",
        "next_song",
        "change_speed",
        "plug_type",
        "maybe",
        "w2",
        "oil_change_when",
        "thank_you",
        "shopping_list_update",
        "pto_balance",
        "order_checks",
        "travel_alert",
        "fun_fact",
        "sync_device",
        "schedule_maintenance",
        "apr",
        "transfer",
        "ingredient_substitution",
        "calories",
        "current_location",
        "international_fees",
        "calculator",
        "definition",
        "next_holiday",
        "update_playlist",
        "mpg",
        "min_payment",
        "change_user_name",
        "restaurant_suggestion",
        "travel_notification",
        "cancel",
        "pto_used",
        "travel_suggestion",
        "change_volume",
    ]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__(
            "clinc/clinc_oos", "plus", split=split, adapter_dict=adapter_dict
        )
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

        assert self.OPTIONS == self.raw_data.features["intent"].names

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": row["text"],
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["intent"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


class SNLIAdapter(BaseDatasetAdapter):
    OPTIONS = ["entailment", "neutral", "contradiction"]

    def __init__(self, split="train", adapter_dict=None, **kwargs):
        super().__init__(
            "stanfordnlp/snli", None, split=split, adapter_dict=adapter_dict
        )
        # drop pairs with no gold label (label == -1)
        self.raw_data = self.raw_data.filter(lambda row: row["label"] != -1)
        self.raw_data = self.raw_data.add_column("id", list(range(len(self.raw_data))))

        assert self.OPTIONS == self.raw_data.features["label"].names

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "gold_label": self.OPTIONS[row["label"]],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "prompt_options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)


# MMLU-Pro
class MMLUProAdapter(BaseDatasetAdapter):
    def __init__(self, split="train"):
        super().__init__("TIGER-Lab/MMLU-Pro", "default", split=split)

    def _process_dataset_row(self, row):
        return {
            "id": row["question_id"],
            "input": row["question"],
            "options": row["options"],
            "gold_label": row["options"][row["answer_index"]],
            "category": row["category"],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "options", "gold_label", "category"]
        return mapped_ds.select_columns(ordered_cols)


# SciFact


def download_data_file(dl_url, local_file, dl_text):
    """Download data file from online source, given URL to that file.

    Args:
        dl_url (str): URL for the file to download.
        local_file (Path): path to download the file to.
        dl_text (str): text to display to the user when downloading,
            formatted as Downloading {dl_text}...
    """
    # ensure dir exists
    local_file.parent.mkdir(parents=True, exist_ok=True)

    # download
    response = requests.get(dl_url, stream=True)
    response.raise_for_status()

    dl_size = int(response.headers.get("content-length", 0))

    with open(local_file, "wb") as f, tqdm(
        desc=f"Downloading {dl_text}...",
        total=dl_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)


def normalize_wiki_id(wiki_id: Optional[str]) -> Optional[str]:
    """Normalise the wiki_id string. Between the wiki dataset and
    FEVER splits, there is a mismatch in special character encoding.
    This is particularly with accented characters.

    Args:
        wiki_id (str): wikipedia page id (or formatted title).

    Returns:
        str: normalised unicode wiki id.
    """
    if wiki_id is None:
        return None
    return unicodedata.normalize("NFC", wiki_id)


def clean_brackets(text: str) -> str:
    """Replace all Penn Treebank escaped brackets with their string character.

    Args:
        text (str): text to replace brackets in.

    Returns:
        str: cleaned string.
    """
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("-LCB-", "{").replace("-RCB-", "}")
    return text


def wiki_id_to_title(wiki_id: str) -> str:
    """Extract a plain-text title from the wiki id which could
    contain underscores and Penn Treebank escaped brackets.

    Args:
        wiki_id (str): wikipedia page id (or formatted title).
    """
    return clean_brackets(wiki_id).replace("_", " ")


def detokenize_fever_wiki(text: str) -> str:
    """Detokenise the wiki dataset for better parsing by LLMs.

    Args:
        text (str): Moses tokenised text.

    Returns:
        str: detokenised plaintext.
    """
    text = clean_brackets(text)

    md = MosesDetokenizer(lang="en")
    tokens = text.split()
    return md.detokenize(tokens)


# Fever
class FeverAdapter(BaseDatasetAdapter):
    SPLITS = {
        "train": {
            "local_path": Path("./data/fever_train.jsonl"),
            "dl_path": "https://fever.ai/download/fever/train.jsonl",
        },
        "dev": {
            "local_path": Path("./data/fever_dev.jsonl"),
            "dl_path": "https://fever.ai/download/fever/paper_dev.jsonl",
        },
        "test": {
            "local_path": Path("./data/fever_test.jsonl"),
            "dl_path": "https://fever.ai/download/fever/paper_test.jsonl",
        },
    }
    OPTIONS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    WIKI_DL_URL = "https://fever.ai/download/fever/wiki-pages.zip"
    WIKI_ZIP_PATH = Path("./data/wiki-pages.zip")
    WIKI_PATH = Path("./data/wiki-pages")

    # TODO: clean this constructor up -- fit with Base
    def __init__(self, split="train", adapter_dict=None, **kwargs):
        self.dataset_name = "fever/fever"
        self.config_name = "v1.0"
        self.split = split
        self.adapter_dict = adapter_dict
        if adapter_dict is None:
            self.prompt_options = self.OPTIONS
        else:
            if set(self.OPTIONS) != set(adapter_dict.values()):
                raise ValueError(
                    f"Adapter dict values must be the same as options: {', '.join(self.OPTIONS)}."
                )
            self.prompt_options = sorted(list(adapter_dict.keys()))
        self.raw_data = self._load_dataset(split=split)
        self._formatted_data = None

    def _extract_wiki_ids(self, df: pd.DataFrame):
        """Extract wiki ids from the structured dataframe of FEVER splits."""
        return list(
            {
                normalize_wiki_id(item[2])
                for evidence_lists in df["evidence"]
                for evidence_set in evidence_lists
                for item in evidence_set
                if item[2] is not None
            }
        )

    def get_wiki_lookup_dict(self):
        """Get wiki lookup dict. First key is the wiki_id. This then returns
        another dictionary where each line is numbered -- the key is the line number
        and the value is the text from that line.
        """
        wiki_df = self.load_wiki_pages_df()

        wiki_dict = {}
        for _, row in wiki_df.iterrows():
            lines = str(row["lines"]).split("\n")
            data_dict = {}
            for line in lines:
                # tab separated values where index 0 is line number
                # index 1 is sentence
                # index 2+ are links (irrelevant for this)
                data = line.split("\t")
                try:
                    data_dict[int(data[0])] = detokenize_fever_wiki(data[1])
                except ValueError:
                    pass

            row_id = row["id"]
            assert isinstance(row_id, str)
            wiki_dict[normalize_wiki_id(row_id)] = data_dict

        return wiki_dict

    def _extract_row_wiki_evidence(self, row: pd.Series, wiki_dict: dict):
        """Extract wiki evidence given the row and wiki dictionary.

        Args:
            row (pd.Series): row of FEVER dataset split.
            wiki_dict (dict): wiki data as a dictionary (first key is
                the wiki_id, key in the returned dict is line number).

        Returns:
            list: 3d list, 1st dimension is list of all different blocks of
                evidence; 2nd dimension contains individual pieces of evidence;
                3rd dimension contains wiki_id/title and text.
        """
        evidence = row["evidence"]

        output_evidence = []
        for evidence_set in evidence:
            output_pieces = []
            for evidence_piece in evidence_set:
                wiki_id = normalize_wiki_id(evidence_piece[2])
                sentence_num = evidence_piece[3]
                if wiki_id is None or sentence_num is None:
                    continue
                try:
                    output_pieces.append(
                        (
                            wiki_id_to_title(wiki_id),
                            wiki_dict[wiki_id][int(sentence_num)],
                        )
                    )
                except KeyError:
                    print(f"Could not find {wiki_id}.")

            output_evidence.append(output_pieces)
        return output_evidence

    def _add_wiki_evidence(
        self, split_df: pd.DataFrame, wiki_dict: dict
    ) -> pd.DataFrame:
        """Add evidence from wiki dataset to fever dataset.

        Args:
            split_df (pd.DataFrame): dataframe of the fever dataset split.
            wiki_dict (dict): wiki data as a dictionary (first key is
                the wiki_id, key in the returned dict is line number).

        Returns:
            pd.DataFrame: dataframe of the fever dataset split with added
                wiki pages evidence.
        """
        split_df["evidence"] = split_df.apply(
            lambda row: self._extract_row_wiki_evidence(row, wiki_dict), axis=1
        )

        return split_df

    def load_wiki_pages_df(self):
        """Load wiki pages and filter based on the data split (only retrieve
        the wiki pages required).
        """
        # check whether reduced version already exists
        reduced_wiki_file = self.WIKI_PATH / "reduced.jsonl"
        if not reduced_wiki_file.exists():
            if not self.WIKI_ZIP_PATH.exists():
                print(
                    f"File not found: {reduced_wiki_file}. Downloading from {self.WIKI_DL_URL}..."
                )

                # download zip file
                download_data_file(
                    self.WIKI_DL_URL, self.WIKI_ZIP_PATH, "FEVER wiki-pages.zip"
                )

                # unzip file
                print("Unzipping wiki-pages.zip...")
                with ZipFile(self.WIKI_ZIP_PATH, "r") as zip_ref:
                    zip_ref.extractall(self.WIKI_PATH)
                print("Finished.")

            # TODO: could stream to file rather than load df in memory
            print("Collecting required wiki-pages dataset IDs...")
            # collect wiki page links from each split
            # TODO: make second wrapper function that just returns df
            wiki_ids = list(
                {
                    normalize_wiki_id(wiki_id)
                    for split in self.SPLITS
                    for wiki_id in self._extract_wiki_ids(
                        self._load_dataset(split, raw_df=True)
                    )
                }
            )
            print("Done.")

            print("Creating reduced wiki-pages jsonl...")
            # loop through wiki jsonls and extract data points using id
            wiki_files = [f for f in (self.WIKI_PATH / "wiki-pages").glob("*.jsonl")]

            filtered_dfs = []
            for file_path in tqdm(wiki_files, desc="Filtering Wiki Chunks"):
                # load chunk
                temp_df = pd.read_json(file_path, lines=True, encoding="utf-8")

                temp_df["id"] = temp_df["id"].apply(normalize_wiki_id)

                # filter ids
                match_df = temp_df[temp_df["id"].isin(wiki_ids)]

                if not match_df.empty:
                    filtered_dfs.append(match_df)

            print("Concatenating and saving to jsonl file...")
            # concat dfs
            wiki_df = pd.concat(filtered_dfs, ignore_index=True)

            wiki_df.to_json(
                reduced_wiki_file,
                orient="records",
                lines=True,
                index=False,
                force_ascii=False,
            )
            print("Done.")
        else:
            wiki_df = pd.read_json(reduced_wiki_file, lines=True, encoding="utf-8")

        return wiki_df

    def _load_dataset(self, split, raw_df=False):
        """Load dataset split. Download if not found in data directory.

        Args:
            split (str): which split to load (train, dev, test).
            raw_df (bool): whether to return the raw DataFrame rather than dataset.

        Returns:
            pd.DataFrame or Dataset: dataset as a pandas DataFrame or HuggingFace Dataset.
        """
        if split not in self.SPLITS:
            raise ValueError(
                f"Split {split} not supported. Choose from {list(self.SPLITS.keys())}"
            )

        config = self.SPLITS[split]
        local_file = config["local_path"]
        dl_url = config["dl_path"]

        # check if local file in splits exists
        # if not exists, download the dataset
        if not local_file.exists():
            print(f"File not found: {local_file}. Downloading from {dl_url}...")

            # download data
            download_data_file(dl_url, local_file, f"FEVER ({split.upper()} split)")

        df = pd.read_json(local_file, lines=True, encoding="utf-8")

        if raw_df:
            return df

        # link wiki-pages into evidence
        wiki_dict = self.get_wiki_lookup_dict()
        df = self._add_wiki_evidence(df, wiki_dict)

        # return the dataset
        return Dataset.from_pandas(df)

    def _process_dataset_row(self, row):
        return {
            "id": row["id"],
            "input": row["claim"],
            "context": row["evidence"],
            "options": self.OPTIONS,
            "prompt_options": self.prompt_options,
            "verifiable": row["verifiable"],
            "gold_label": row["label"],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = [
            "id",
            "input",
            "context",
            "options",
            "prompt_options",
            "verifiable",
            "gold_label",
        ]
        return mapped_ds.select_columns(ordered_cols)


# PubMedQA
class PubMedQAAdapter(BaseDatasetAdapter):
    def __init__(self, split="train"):
        super().__init__("qiaojin/PubMedQA", "pqa_labeled", split=split)

    def _process_dataset_row(self, row):
        return {
            "id": row["pubid"],
            "input": row["question"],
            "context": {
                "contexts": row["context"]["contexts"],
                "labels": row["context"]["labels"],
            },
            "options": ["yes", "no", "maybe"],
            "gold_label": row["final_decision"],
        }

    def _get_format_dataset(self):
        mapped_ds = self.raw_data.map(
            self._process_dataset_row,
            remove_columns=self.raw_data.column_names,
            load_from_cache_file=False,
        )

        ordered_cols = ["id", "input", "context", "options", "gold_label"]
        return mapped_ds.select_columns(ordered_cols)

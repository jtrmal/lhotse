"""
"""


import logging
import shutil
import string
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

_DEFAULT_URL = (
    "https://codeload.github.com/revdotcom/speech-datasets/zip/refs/heads/main"
)

DEV_SUBSETS = (
    "AFI_en-us_multi-spontaneous_healthcare-retail_v01_Dataset",
    "JPT_en-us_multi-spontaneous_banking_v01_Dataset_1_Dataset",
    "JPT_en-us_multi-spontaneous_insurance_v01_Dataset_1_Dataset",
)


TSV_TRANSCRIPTION_ID = 0
TSV_CHANNEL = 1
TSV_BEGIN_TIME = 2
TSV_END_TIME = 3
TSV_TRANSCRIPT = 4
TSV_DURATION = 5
TSV_RECORDING_ID = 6

TSV_LEFTCHANNELSPEAKERID = 15
TSV_LEFTCHANNELSPEAKERGENDER = 16
TSV_LEFTCHANNELSPEAKERAGE = 17
TSV_LEFTCHANNELSPEAKERLIVINGCOUNTRY = 18
TSV_LEFTCHANNELSPEAKERACCENT = 19
TSV_LEFTCHANNELROLE = 12

TSV_RIGHTCHANNELSPEAKERID = 20
TSV_RIGHTCHANNELSPEAKERGENDER = 21
TSV_RIGHTCHANNELSPEAKERAGE = 22
TSV_RIGHTCHANNELSPEAKERLIVINGCOUNTRY = 23
TSV_RIGHTCHANNELSPEAKERACCENT = 24
TSV_RIGHTCHANNELROLE = 13


def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text


def parse_tsv_file(filename: Pathlike):
    with open(filename) as f:
        transcript = list()
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            line = line.split("\t")
            transcript.append(line)
        return transcript


def convert_time(time: str):
    import datetime

    fields = time.split(":")
    time = datetime.timedelta(
        hours=int(fields[0]), minutes=int(fields[1]), seconds=float(fields[2])
    )
    return time.total_seconds()


def convert_duration(time: str):
    import datetime

    fields = time.split(":")
    time = datetime.timedelta(hours=0, minutes=0, seconds=float(fields[2]))
    return time.total_seconds()


def prepare_uniphore_dev(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_text: bool = False,
) -> Union[RecordingSet, SupervisionSet]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Pathlike, the path of the data dir. The structure is
        expected to mimic the structure in the github repository, notably
        the mp3 files will be searched for in [corpus_dir]/media and transcriptions
        in the directory [corpus_dir]/transcripts/nlp_references
    :param output_dir: Pathlike, the path where to write the manifests.
    :param normalize_text: Bool, if True, normalize the text.
    :return: (recordings, supervisions) pair

    .. caution::
        The `normalize_text` option removes all punctuation and converts all upper case
        to lower case. This includes removing possibly important punctuations such as
        dashes and apostrophes.
    """

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = dict()

    for subset in DEV_SUBSETS:
        logging.info(f"Processing subset: {subset}")
        subset_dir = corpus_dir / subset
        audio_dir = subset_dir / "Audio"
        audio_files = list(audio_dir.glob("*.wav"))

        audio_files.sort()
        recording_set = RecordingSet.from_recordings(
            Recording.from_file(p) for p in audio_files
        )

        transcript_file = subset_dir / "combined.tsv"

        transcript = parse_tsv_file(transcript_file)

        supervision_segments = list()

        if subset.startswith("AFI"):
            TSV_BEGIN_TIME = 2
            TSV_END_TIME = 3
            TSV_TRANSCRIPT = 4
            TSV_DURATION = 5
            TSV_RECORDING_ID = 6

            TSV_LEFTCHANNELSPEAKERID = 15
            TSV_LEFTCHANNELSPEAKERGENDER = 16
            TSV_LEFTCHANNELSPEAKERAGE = 17
            TSV_LEFTCHANNELSPEAKERLIVINGCOUNTRY = 18
            TSV_LEFTCHANNELSPEAKERACCENT = 19
            TSV_LEFTCHANNELROLE = 12

            TSV_RIGHTCHANNELSPEAKERID = 20
            TSV_RIGHTCHANNELSPEAKERGENDER = 21
            TSV_RIGHTCHANNELSPEAKERAGE = 22
            TSV_RIGHTCHANNELSPEAKERLIVINGCOUNTRY = 23
            TSV_RIGHTCHANNELSPEAKERACCENT = 24
            TSV_RIGHTCHANNELROLE = 13
            tsv_offset = 0
        else:
            TSV_BEGIN_TIME = 3
            TSV_END_TIME = 4
            TSV_DURATION = 5
            TSV_TRANSCRIPT = 6
            TSV_RECORDING_ID = 7

            TSV_LEFTCHANNELSPEAKERID = 13
            TSV_LEFTCHANNELNATIVE = 14
            TSV_LEFTCHANNELROLE = 15
            TSV_LEFTCHANNELSPEAKERAGE = 16
            TSV_LEFTCHANNELSPEAKERGENDER = 17
            TSV_LEFTCHANNELSPEAKERLIVINGCOUNTRY = 18
            TSV_LEFTCHANNELSPEAKERACCENT = 19

            TSV_RIGHTCHANNELSPEAKERID = 20
            TSV_RIGHTCHANNELNATIVE = 21
            TSV_RIGHTCHANNELROLE = 22
            TSV_RIGHTCHANNELSPEAKERAGE = 23
            TSV_RIGHTCHANNELSPEAKERGENDER = 24
            TSV_RIGHTCHANNELSPEAKERLIVINGCOUNTRY = 25
            TSV_RIGHTCHANNELSPEAKERACCENT = 26
            tsv_offset = 1

        for line in transcript:
            text = line[TSV_TRANSCRIPT]
            if normalize_text:
                text = normalize(text)

            channel = line[TSV_CHANNEL]
            assert channel in [
                "1",
                "2",
            ], f"Channel ID should be either 1 or 2, got {channel}"

            if channel == "1":
                speaker_id = line[TSV_LEFTCHANNELSPEAKERID]
                speaker_gender = line[TSV_LEFTCHANNELSPEAKERGENDER]
                speaker_age = line[TSV_LEFTCHANNELSPEAKERAGE]
                speaker_living_country = line[TSV_LEFTCHANNELSPEAKERLIVINGCOUNTRY]
                speaker_accent = line[TSV_LEFTCHANNELSPEAKERACCENT]
                speaker_role = line[TSV_LEFTCHANNELROLE]
            else:
                speaker_id = line[TSV_RIGHTCHANNELSPEAKERID]
                speaker_gender = line[TSV_RIGHTCHANNELSPEAKERGENDER]
                speaker_age = line[TSV_RIGHTCHANNELSPEAKERAGE]
                speaker_living_country = line[TSV_RIGHTCHANNELSPEAKERLIVINGCOUNTRY]
                speaker_accent = line[TSV_RIGHTCHANNELSPEAKERACCENT]
                speaker_role = line[TSV_RIGHTCHANNELROLE]

            s = SupervisionSegment(
                id=line[TSV_TRANSCRIPTION_ID],
                recording_id=line[TSV_RECORDING_ID],
                start=convert_time(line[TSV_BEGIN_TIME]),
                duration=convert_duration(line[TSV_DURATION]),
                channel=int(line[TSV_CHANNEL]) - 1,
                language="en-us",
                text=text,
                speaker=speaker_id,
                gender=speaker_gender,
                custom={
                    "accent": speaker_accent,
                    "role": speaker_role,
                    "living_country": speaker_living_country,
                    "age": speaker_age,
                },
            )
            supervision_segments.append(s)
        supervision_set = SupervisionSet.from_segments(supervision_segments)

        validate_recordings_and_supervisions(recording_set, supervision_set)
        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"uniphore_supervisions_{subset}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"uniphore_recordings_{subset}.jsonl.gz")

        manifests[subset] = {
            "supervisions": supervision_set,
            "recordings": recording_set,
        }

    return manifests

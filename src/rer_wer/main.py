import pandas as pd
import logging
import time
from pydra import Workflow, Submitter, mark
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures.model import HFModel
from senselab.audio.tasks.speech_to_text.speech_to_text_evaluation import calculate_wer
from senselab.audio.tasks.preprocessing.preprocessing import downmix_audios_to_mono, resample_audios
import string

def preprocess_text(text):
    """Preprocess text by converting to lowercase and removing punctuation."""
    return text.lower().translate(str.maketrans('', '', string.punctuation))

@mark.task
def read_audio(file_path):
    """Read an audio file and return an Audio object."""
    return Audio.from_filepath(file_path)

@mark.task
def preprocess_audio(audio, resample_rate=16000):
    """Preprocess an audio file by downmixing and resampling."""
    return resample_audios(downmix_audios_to_mono([audio]), resample_rate)

@mark.task
def transcribe_audios_task(audios, model):
    """Transcribe audio files using a speech-to-text model."""
    return transcribe_audios(audios, model)

@mark.task
def compute_wer(reference, hypothesis):
    """Compute the Word Error Rate (WER) between reference and hypothesis texts."""
    return calculate_wer(reference, hypothesis)

@mark.task
def extract_metadata(row, column):
    """Extract metadata from a DataFrame row based on the specified column."""
    return row[column]

@mark.task
def extract_transcription_text(transcripts):
    """Extract and preprocess transcription text from a list of ScriptLine objects."""
    text = transcripts[0].text
    return preprocess_text(text)

@mark.task
def extract_transcription_text_from_tuple(row):
    """Extract and preprocess transcription text from a tuple."""
    text = row[0].text
    return preprocess_text(text)

@mark.task
def extract_expected_text_from_tuple(row):
    """Extract expected text from a tuple."""
    return preprocess_text(row[1])

def prepare_data_workflow(df, model, plugin='cf'):
    wf = Workflow(name="data_preparation_workflow", input_spec=["x"])
    wf.split("x", x=df.to_dict(orient="records"))

    wf.add(extract_metadata(name="extract_path_task", row=wf.lzin.x, column='absolute_path'))
    wf.add(extract_metadata(name="extract_expected_text_task", row=wf.lzin.x, column='expected'))
    wf.add(extract_metadata(name="extract_school_task", row=wf.lzin.x, column='school'))
    wf.add(extract_metadata(name="extract_grade_task", row=wf.lzin.x, column='grade'))
    wf.add(extract_metadata(name="extract_score_task", row=wf.lzin.x, column='score'))
    wf.add(extract_metadata(name="extract_id_task", row=wf.lzin.x, column='identifier'))
    wf.add(read_audio(name="read_audio", file_path=wf.extract_path_task.lzout.out))
    wf.add(preprocess_audio(name="preprocess_audio_task", audio=wf.read_audio.lzout.out))

    wf.set_output(
        [
            ("file", wf.extract_path_task.lzout.out),
            ("school", wf.extract_school_task.lzout.out),
            ("grade", wf.extract_grade_task.lzout.out),
            ("score", wf.extract_score_task.lzout.out),
            ("identifier", wf.extract_id_task.lzout.out),
            ("expected_text", wf.extract_expected_text_task.lzout.out),
            ("audio", wf.read_audio.lzout.out),
            ("preprocessed_audio", wf.preprocess_audio_task.lzout.out),
        ]
    )

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    
    files = []
    schools = []
    grades = []
    scores = []
    identifiers = []
    expected_texts = []
    audios = []
    preprocessed_audios = []
    for res in results:
        files.append(res.output.file)
        schools.append(res.output.school)
        grades.append(res.output.grade)
        scores.append(res.output.score)
        identifiers.append(res.output.identifier)
        expected_texts.append(res.output.expected_text)
        audios.append(res.output.audio)
        preprocessed_audios.append(res.output.preprocessed_audio[0])
    
    return files, schools, grades, scores, identifiers, expected_texts, audios, preprocessed_audios

def transcribe_workflow(preprocessed_audios, model, plugin='cf'):
    transcription_wf = Workflow(
        name="audio_transcription",
        input_spec=["preprocessed_audios"],
        preprocessed_audios=preprocessed_audios
    )

    transcription_wf.add(
        transcribe_audios_task(
            name="transcribe_audios",
            audios=transcription_wf.lzin.preprocessed_audios,
            model=model
        )
    )

    transcription_wf.set_output(
        {
            "transcriptions": transcription_wf.transcribe_audios.lzout.out
        }
    )

    with Submitter(plugin=plugin) as sub:
        sub(transcription_wf)

    results = transcription_wf.result()
    return results.output.transcriptions

def wer_workflow(transcriptions_and_expected_texts, plugin='cf'):
    wer_wf = Workflow(
        name="wer_wf",
        input_spec=["transcriptions_and_expected_texts"],
    )
    wer_wf.split(
        splitter="transcriptions_and_expected_texts",
        transcriptions_and_expected_texts=transcriptions_and_expected_texts,
    )

    wer_wf.add(extract_transcription_text_from_tuple(name="extract_transcription_text_task", 
                                      row=wer_wf.lzin.transcriptions_and_expected_texts))
    wer_wf.add(extract_expected_text_from_tuple(name="extract_expected_text_from_tuple_task",
                                                row=wer_wf.lzin.transcriptions_and_expected_texts))
    wer_wf.add(compute_wer(name="compute_wer",
                       reference=wer_wf.extract_expected_text_from_tuple_task.lzout.out,
                       hypothesis=wer_wf.extract_transcription_text_task.lzout.out))
    
    wer_wf.set_output(
            [
                ("transcription_text", wer_wf.extract_transcription_text_task.lzout.out),
                ("wer", wer_wf.compute_wer.lzout.out),
            ])

    with Submitter(plugin=plugin) as sub:
        sub(wer_wf)

    results = wer_wf.result()
    
    transcription_texts = []
    wers = []
    for res in results:
        transcription_texts.append(res.output.transcription_text)
        wers.append(res.output.wer)
    
    return transcription_texts, wers

def run():
    #########################################################################
    # VARIABLES

    # Define the input CSV file
    csv_file = "/Users/fabiocat/Documents/git/rer_wer/data/input.csv"
    # Define the speech to text model
    model_uri = "openai/whisper-tiny"
    # Define the output CSV file
    output_file = "/Users/fabiocat/Documents/git/rer_wer/data/output.csv"

    #########################################################################

    #########################################################################
    # SETUP LOGGER

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    #########################################################################

    #########################################################################
    # WORKFLOW EXECUTION

    # Read the CSV file
    start_time = time.time()
    logger.info("Reading the CSV file...")
    df = pd.read_csv(csv_file)
    df = df[df['score'] > 0.5]
    logger.info(f"CSV file read in {time.time() - start_time:.2f} seconds")

    # Define the transcription model
    model = HFModel(path_or_uri=model_uri, revision="main")

    # Prepare data workflow
    start_time = time.time()
    logger.info("Starting data preparation workflow...")
    files, schools, grades, scores, identifiers, expected_texts, audios, preprocessed_audios = prepare_data_workflow(df, model)
    logger.info(f"Data preparation workflow completed in {time.time() - start_time:.2f} seconds")

    # Transcription workflow
    start_time = time.time()
    logger.info("Starting transcription workflow...")
    transcriptions = transcribe_workflow(preprocessed_audios, model)
    logger.info(f"Transcription workflow completed in {time.time() - start_time:.2f} seconds")

    # WER workflow
    start_time = time.time()
    logger.info("Starting WER workflow...")
    transcriptions_and_expected_texts = list(zip(transcriptions, expected_texts))
    transcription_texts, wers = wer_workflow(transcriptions_and_expected_texts)
    logger.info(f"WER workflow completed in {time.time() - start_time:.2f} seconds")

    # Write the output CSV file
    start_time = time.time()
    logger.info("Writing the output CSV file...")
    output_df = pd.DataFrame({"file": files, 
                              "school": schools, 
                              "grade": grades, 
                              "score": scores, 
                              "identifier": identifiers, 
                              "expected_text": expected_texts, 
                              "transcription": transcription_texts, 
                              "wer": wers})
    output_df.to_csv(output_file, index=False)
    logger.info(f"Output CSV file written in {time.time() - start_time:.2f} seconds")

# Execute the workflow
if __name__ == "__main__":
    run()

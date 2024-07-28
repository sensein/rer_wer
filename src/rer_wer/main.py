import pandas as pd
import logging
import time
from pydra import Workflow, Submitter, mark
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.audio.tasks.speech_to_text.speech_to_text_evaluation import calculate_wer
from senselab.audio.tasks.preprocessing.preprocessing import downmix_audios_to_mono, resample_audios
import string

def preprocess_text(text: str):
    """Preprocess text by converting to lowercase and removing punctuation."""
    return str(text).lower().translate(str.maketrans('', '', string.punctuation))

@mark.task
def read_audio(file_path):
    """Read an audio file and return an Audio object."""
    return Audio.from_filepath(file_path)

@mark.task
def preprocess_audio(audio, resample_rate=16000):
    """Preprocess an audio file by downmixing and resampling."""
    return resample_audios(downmix_audios_to_mono([audio]), resample_rate)

@mark.task
def transcribe_audios_task(audios, model, language, device):
    """Transcribe audio files using a speech-to-text model."""
    return transcribe_audios(audios, model, language, device)

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

    wf.add(extract_metadata(name="extract_path_task", row=wf.lzin.x, column='absolute_path', cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/1"))
    wf.add(extract_metadata(name="extract_expected_text_task", row=wf.lzin.x, column='expected', cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/2"))
    wf.add(extract_metadata(name="extract_school_task", row=wf.lzin.x, column='school', cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/3"))
    wf.add(extract_metadata(name="extract_grade_task", row=wf.lzin.x, column='grade', cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/4"))
    wf.add(extract_metadata(name="extract_score_task", row=wf.lzin.x, column='score', cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/5"))
    wf.add(extract_metadata(name="extract_id_task", row=wf.lzin.x, column='identifier', cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/6"))
    wf.add(read_audio(name="read_audio", file_path=wf.extract_path_task.lzout.out, cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/7"))
    wf.add(preprocess_audio(name="preprocess_audio_task", audio=wf.read_audio.lzout.out, cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/8"))

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

def transcribe_workflow(preprocessed_audios, model, language, device, plugin='cf'):
    transcription_wf = Workflow(
        name="audio_transcription",
        input_spec=["preprocessed_audios"],
        preprocessed_audios=preprocessed_audios
    )

    transcription_wf.add(
        transcribe_audios_task(
            name="transcribe_audios",
            audios=transcription_wf.lzin.preprocessed_audios,
            model=model,
            language=language,
            device=device,
            cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/9"
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
                                      row=wer_wf.lzin.transcriptions_and_expected_texts,
                                      cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/10"))
    wer_wf.add(extract_expected_text_from_tuple(name="extract_expected_text_from_tuple_task",
                                                row=wer_wf.lzin.transcriptions_and_expected_texts,
                                                cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/11"))
    wer_wf.add(compute_wer(name="compute_wer",
                       reference=wer_wf.extract_expected_text_from_tuple_task.lzout.out,
                       hypothesis=wer_wf.extract_transcription_text_task.lzout.out,
                       cache_dir="/om2/scratch/Sun/fabiocat/rer_wer/src/rer_wer/cache/12"))
    
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

def process_batch(logger, batch_df, model, language, device):
    start_time = time.time()
    logger.info("Starting data preparation workflow for batch...")
    files, schools, grades, scores, identifiers, expected_texts, audios, preprocessed_audios = prepare_data_workflow(batch_df, model)
    logger.info(f"Data preparation workflow completed in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    logger.info("Starting transcription workflow for batch...")
    transcriptions = transcribe_workflow(preprocessed_audios, model, language, device)
    logger.info(f"Transcription workflow completed in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    logger.info("Starting WER workflow for batch...")
    transcriptions_and_expected_texts = list(zip(transcriptions, expected_texts))
    transcription_texts, wers = wer_workflow(transcriptions_and_expected_texts)
    logger.info(f"WER workflow completed in {time.time() - start_time:.2f} seconds")

    output_df = pd.DataFrame({"file": files,
                              "school": schools,
                              "grade": grades,
                              "score": scores,
                              "identifier": identifiers,
                              "expected_text": expected_texts,
                              "transcription": transcription_texts,
                              "wer": wers})
    return output_df

def run():
    #########################################################################
    # VARIABLES

    # Define the input CSV file
    csv_file = "/om2/scratch/Sun/fabiocat/rer_wer/data/wer_table_profiling_id.csv"
    # Define the speech to text model
    model_uri = "openai/whisper-large-v3"
    # Define the output CSV file
    output_file = "/om2/scratch/Sun/fabiocat/rer_wer/data/output.csv"
    batch_size = 1000

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
    total_rows = len(df)
    logger.info(f"CSV file read in {time.time() - start_time:.2f} seconds, total rows: {total_rows}")


    # Define the transcription model
    model = HFModel(path_or_uri=model_uri, revision="main")
    # Define the device used for transcription
    device = DeviceType.CUDA
    # Define the language used for transcription
    language = Language(language_code="English")

    # Process in batches
    for start_row in range(0, total_rows, batch_size):
        end_row = min(start_row + batch_size, total_rows)
        batch_df = df.iloc[start_row:end_row]
        logger.info(f"Processing batch from row {start_row} to {end_row}")
        batch_start_time = time.time()
        # Process batch
        batch_output_df = process_batch(logger, batch_df, model, language, device)

        # Append results to the output file
        if start_row == 0:
            batch_output_df.to_csv(output_file, index=False)
        else:
            batch_output_df.to_csv(output_file, mode='a', header=False, index=False)

        logger.info(f"Batch from row {start_row} to {end_row} processed and saved in {time.time() - batch_start_time:.2f}")

    logger.info(f"The entire execution was completed in {time.time() - start_time:.2f} seconds")

# Execute the workflow
if __name__ == "__main__":
    run()


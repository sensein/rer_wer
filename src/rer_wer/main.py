import pandas as pd
from pydra import Workflow, Submitter, mark
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures.model import HFModel
from senselab.audio.tasks.speech_to_text.speech_to_text_evaluation import calculate_wer
from typing import List
from senselab.utils.data_structures.script_line import ScriptLine
import string

def run():
    #########################################################################
    # VARIABLES

    # Define the input CSV file
    csv_file = "/om2/scratch/Mon/meral/wer_table_profiling.csv"
    # Define the speech to text model
    model_uri = "openai/whisper-large-v3"
    # Define the output CSV file
    output_file = "/net/vast-storage.ib.cluster/scratch/scratch/Mon/fabiocat/rer_wer/data/output.csv"

    #########################################################################

    #########################################################################
    # UTILITIES
    # Define a task to read the audio file and return an Audio object
    @mark.task
    def read_audio(file_path: str):
        return Audio.from_filepath(file_path)

    # Define a task to transcribe audio files
    @mark.task
    def transcribe_audio(audio, model):
        return transcribe_audios([audio], model)

    # Define a task to compute WER
    @mark.task
    def compute_wer(reference: str, hypothesis: str):
        return calculate_wer(reference, hypothesis)

    @mark.task
    def extract_path(row):
        file_path = row['absolute_path']
        return file_path

    @mark.task
    def extract_expected_text(row):
        expected_text = row['expected']
        # Convert to lowercase
        expected_text = expected_text.lower()
        # Remove punctuation
        expected_text = expected_text.translate(str.maketrans('', '', string.punctuation))
        return expected_text
    
    @mark.task
    def extract_transcription_text(transcripts: List[ScriptLine]):
        text = transcripts[0].text
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    @mark.task
    def extract_school(row):
        school = row['school']
        return school

    @mark.task
    def extract_grade(row):
        grade = row['grade']
        return grade

    @mark.task
    def extract_score(row):
        score = row['score']
        return score


    #########################################################################

    #########################################################################
    # WORKFLOW
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Define the transcription model
    model = HFModel(path_or_uri=model_uri, revision="main")

    # Create the workflow
    wf = Workflow(name="audio_transcription_workflow", input_spec=["x"])
    wf.split("x", x=df.to_dict(orient="records"))
    wf.add(extract_path(name="extract_path_task", row=wf.lzin.x))
    wf.add(extract_expected_text(name="extract_expected_text", row=wf.lzin.x))
    wf.add(extract_school(name="extract_school_task", row=wf.lzin.x))
    wf.add(extract_grade(name="extract_grade_task", row=wf.lzin.x))
    wf.add(extract_score(name="extract_score_task", row=wf.lzin.x))
    wf.add(read_audio(name="read_audio", 
                      file_path=wf.extract_path_task.lzout.out))
    wf.add(transcribe_audio(name="transcribe_audio_task",
                            audio=wf.read_audio.lzout.out, 
                            model=model))
    wf.add(extract_transcription_text(name="extract_transcription_text_task", 
                                      transcripts=wf.transcribe_audio_task.lzout.out))
    wf.add(compute_wer(name="compute_wer",
                       reference=wf.extract_expected_text.lzout.out,
                       hypothesis=wf.extract_transcription_text_task.lzout.out))
    
    wf.set_output(
            [
                ("file", wf.extract_path_task.lzout.out),
                ("school", wf.extract_school_task.lzout.out),
                ("grade", wf.extract_grade_task.lzout.out),
                ("score", wf.extract_score_task.lzout.out),
                ("expected_text", wf.extract_expected_text.lzout.out),
                # ("audio", wf.read_audio.lzout.out),
                # ("transcription", wf.transcribe_audio_task.lzout.out),
                ("transcription_text", wf.extract_transcription_text_task.lzout.out),
                ("wer", wf.compute_wer.lzout.out)
            ]
        )

    with Submitter(plugin='cf') as sub:
        sub(wf)

    outputs = wf.result()

    df_output = pd.DataFrame(columns=[])
    for output in outputs:
        new_line = pd.Series(
            {
                "file": output.output.file,
                "school": output.output.school,
                "grade": output.output.grade,
                "score": output.output.score,
                "expected_text": output.output.expected_text,
                "transcription_text": output.output.transcription_text,
                "wer": output.output.wer,
            }
        )
        df_output = pd.concat([df_output, new_line.to_frame().T], ignore_index=True)

    df_output.to_csv(output_file, index=False)
    print("All done.")

# Execute the workflow
if __name__ == "__main__":
    run()

import os

from data_loaders.load_i2b2_data_updated import load_i2b2_absolute_data
from evaluation.metrics import compute_metrics
from llm_testing.absolute_time_predictor import ZeroShotPromptingModel
from llm_testing.llm_models import local_llm
from llm_testing.llm_models.chatGPT import OpenAIModel
from llm_testing.llm_models.gemini import GeminiModel
from llm_testing.multiple_times_predictor import EventTimePredictor


def transform_llm_prediction_to_evaluation_format(prediction, admission_time):
    if prediction["start_minutes"] == None:
        start_time = admission_time
    else:
        start_time = int(prediction["start_minutes"])
    start_lower = start_time - 60 # TODO change later if we evaluate the spans as well
    start_upper = start_time + 60
    if prediction["end_minutes"] == None:
        end_time = admission_time
    else:
        end_time = int(prediction["end_minutes"])
    end_lower = end_time - 60 # TODO change later if we evaluate the spans as well
    end_upper = end_time + 60

    if "duration_minutes" in prediction:
        duration_time = int(prediction["duration_minutes"])
    else:
        duration_time = 60
    duration_lower = duration_time - 60 # TODO change later if we evaluate the spans as well
    duration_upper = duration_time + 60

    prediction = ((start_time, start_lower, start_upper),
                  (end_time, end_lower, end_upper),
                  (duration_time, duration_lower, duration_upper))
    return prediction

def transform_row_to_evaluation_format(row):
    start_time = row["start_time_minutes"]
    start_lower = row["start_lower_minutes"]
    start_upper = row["start_upper_minutes"]
    end_time = row["end_time_minutes"]
    end_lower = row["end_lower_minutes"]
    end_upper = row["end_upper_minutes"]
    duration_time = row["duration_minutes"]
    duration_lower = row["duration_lower_minutes"]
    duration_upper = row["duration_upper_minutes"]


    prediction = ((start_time, start_lower, start_upper),
                  (end_time, end_lower, end_upper),
                  (duration_time, duration_lower, duration_upper))
    return prediction

def log_text(line, log_file="llm_evaluation.log"):
    with open(log_file, "a") as log:
        log.write(line)
        log.write("\n")

def evaluate_llm_prompting(predictor, model_id):
    dataframe_test = load_i2b2_absolute_data(test_split=True)
    dataframe_test = dataframe_test.iloc[:100]
    print(f"Testing data size: {len(dataframe_test)}")

    predictions_starts = []
    predictions_ends = []
    predictions_durations = []
    gold_starts = []
    gold_ends = []
    gold_durations = []
    for row in dataframe_test.iloc:
        prediction = predictor.predict(row)
        s, e, d = transform_llm_prediction_to_evaluation_format(prediction, row["admission_date_minutes"])
        predictions_starts.append(s)
        predictions_ends.append(e)
        predictions_durations.append(d)

        s,e,d = transform_row_to_evaluation_format(row)
        gold_starts.append(s)
        gold_ends.append(e)
        gold_durations.append(d)
        eval_metrics_partial = compute_metrics(predictions_starts, predictions_ends, predictions_durations, gold_starts,
                                       gold_ends,
                                       gold_durations)
        print(f"{model_id} --- Partial metrics: {eval_metrics_partial}")
        log_text(f"{model_id} --- Partial metrics: {eval_metrics_partial}", log_file=f"{model_id}_evaluation.log")

    eval_metrics = compute_metrics(predictions_starts, predictions_ends, predictions_durations, gold_starts, gold_ends,
                                   gold_durations)
    print(f"{model_id} --- Evaluation metrics: {eval_metrics}")
    log_text(f"{model_id} --- Evaluation metrics: {eval_metrics}", log_file=f"{model_id}_evaluation.log")

    pass

def evaluate_llm_prompting_batch_model(predictor, model_id):
    dataframe_test = load_i2b2_absolute_data(test_split=True)
    dataframe_test = dataframe_test.iloc[:100]
    print(f"Testing data size: {len(dataframe_test)}")

    predictions_starts = []
    predictions_ends = []
    predictions_durations = []
    gold_starts = []
    gold_ends = []
    gold_durations = []
    for document_id, predictions in predictor.predict(dataframe_test):
        print(document_id, predictions)
        for prediction in predictions:
            pred_start = prediction["predicted_start_time_minutes"]
            pred_end = prediction["predicted_end_time_minutes"]
            predictions_starts.append((pred_start, pred_start + 60, pred_start - 60))
            predictions_ends.append((pred_end, pred_end + 60, pred_end - 60))
            predictions_durations.append((pred_end - pred_start, pred_end - pred_start - 60, pred_end - pred_start + 60))
            gold_starts.append((prediction["start_time_minutes"], prediction["start_upper_minutes"], prediction["start_lower_minutes"]))
            gold_ends.append((prediction["end_time_minutes"], prediction["end_upper_minutes"], prediction["end_lower_minutes"]))
            gold_durations.append((prediction["duration_minutes"], prediction["duration_upper_minutes"], prediction["duration_lower_minutes"]))
        eval_metrics_partial = compute_metrics(predictions_starts, predictions_ends, predictions_durations, gold_starts,
                                               gold_ends, gold_durations)
        print(f"{model_id} --- Partial metrics: {eval_metrics_partial}")
        log_text(f"{model_id} --- Partial metrics: {eval_metrics_partial}", log_file=f"{model_id}_evaluation.log")

    eval_metrics_partial = compute_metrics(predictions_starts, predictions_ends, predictions_durations, gold_starts,
                                           gold_ends, gold_durations)
    print(f"{model_id} --- Final metrics: {eval_metrics_partial}")
    log_text(f"{model_id} --- Final metrics: {eval_metrics_partial}", log_file=f"{model_id}_evaluation.log")


def evaluate_all_llms():
    # api_key = os.getenv("GEMINI_API_KEY")
    # model = GeminiModel(api_key=api_key)
    # predictor = ZeroShotPromptingModel(model, use_structured_response=False)
    # evaluate_llm_prompting(predictor, model_id="gemini_individual_plain")
    # predictor = ZeroShotPromptingModel(model, use_structured_response=True)
    # evaluate_llm_prompting(predictor, model_id="gemini_individual_structured")
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAIModel(api_key=api_key)
    # predictor = ZeroShotPromptingModel(model, use_structured_response=False)
    # evaluate_llm_prompting(predictor, model_id="chatgpt_individual_plain")
    # predictor = ZeroShotPromptingModel(model, use_structured_response=True)
    # evaluate_llm_prompting(predictor, model_id="chatgpt_individual_structured")
    # model = local_llm.OllamaModel(model_name="gemma3:27b")
    # predictor = ZeroShotPromptingModel(model, use_structured_response=False)
    # evaluate_llm_prompting(predictor, model_id="local_gemma_individual_plain")
    # predictor = ZeroShotPromptingModel(model, use_structured_response=True)
    # evaluate_llm_prompting(predictor, model_id="local_gemma_individual_structured")

    predictor = EventTimePredictor(model, use_structured_response=True)
    evaluate_llm_prompting_batch_model(predictor, model_id="chatgpt_batch_structured")
if __name__ == '__main__':
    evaluate_all_llms()

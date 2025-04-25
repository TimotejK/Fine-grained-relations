import re

from pydantic import BaseModel

from llm_testing.llm_models.gemini import GeminiModel

class TimeInterval(BaseModel):
    years: int
    months: int
    days: int
    hours: int
    minutes: int

class AbsoluteTimePredictor:
    def predict(self, row):
        raise NotImplementedError

class ZeroShotPromptingModel(AbsoluteTimePredictor):
    def __init__(self, llm_api, use_structured_response=False):
        self.llm = llm_api
        self.time_units_to_minutes = {
            "minute": 1,
            "hour": 60,
            "day": 1440,
            "month": 43200,  # Approximating 1 month as 30 days
            "year": 1440*365  # Approximating 1 year as 365 days
        }
        self.use_structured_response=use_structured_response

    def predict_part(self, row, prediction_point="start"):
        summary = row["text"]
        event_start = row["start_char"]
        event_end = row["end_char"]
        admission_time = row["admission_date_minutes"]
        discharge_time = row["discharge_date_minutes"]
        summary_with_event = summary[:event_start] + "<event>" + summary[event_start:event_end] + "</event>" + summary[
                                                                                                               event_end:]
        if self.use_structured_response:
            prompt = (
                "Below is a patient discharge summary. Guess how long before or after the admission date did the event "
                "marked with <event> <point>. Provide your guess as a number of months, days, hours, or minutes. Answer in json where numbers are negative if event occurred before admission time format with example: {{\"months\": 0, \"days\": -3, \"hours\": 0, \"minutes\": 0}}\n\n"
                "{summary}\n")
            response: TimeInterval = self.llm.predict_schema(prompt=prompt.format(summary=summary_with_event, point=prediction_point),
                                                             schema=TimeInterval)
            minutes = (
                    response.years * self.time_units_to_minutes["year"]
                    + response.months * self.time_units_to_minutes["month"]
                    + response.days * self.time_units_to_minutes["day"]
                    + response.hours * self.time_units_to_minutes["hour"]
                    + response.minutes
            )
            return minutes + admission_time
        else:
            prompt = (
                "Below is a patient discharge summary. Guess how long before or after the admission date did the event "
                "marked with <event> <point>. Provide your guess as a number of months, days, hours, or minutes.\n\n"
                "{summary}\n")
            response = self.llm.predict(prompt.format(summary=summary_with_event, point=prediction_point))

            print(response)
            patterns = [r"(\d+) (months?|days?|hours?|minutes?) (before|after)",
                        r"(\d+) (months?|days?|hours?|minutes?)"]
            for search_pattern in patterns:
                # print(re.search(search_pattern, response).group())
                match = re.search(search_pattern, response)
                if match:
                    value = int(match.group(1))
                    unit = match.group(2).rstrip('s')  # Remove plural 's'
                    direction = match.group(3)

                    # Convert to minutes
                    minutes = value * self.time_units_to_minutes[unit]

                    # Make it negative if "before", positive if "after"
                    if direction == "before":
                        minutes = -minutes

                    return minutes + admission_time

    def predict(self, row):
        start = self.predict_part(row, prediction_point="start")
        end = self.predict_part(row, prediction_point="end")
        return {"start_minutes": start, "end_minutes": end}


if __name__ == '__main__':
    text = "\\nADMISSION DATE :\\n10/17/95\\nDISCHARGE DATE :\\n10/20/95\\nHISTORY OF PRESENT ILLNESS :\\nThis is a 73-year-old man with squamous cell carcinoma of the lung , status post lobectomy and resection of left cervical recurrence , admitted here with fever and neutropenia .\\nRecently he had been receiving a combination of outpatient chemotherapy with the CAMP Program .\\nOther medical problems include hypothyroidism , hypercholesterolemia , hypertension and neuropathy from Taxol .\\nHOSPITAL COURSE :\\nHe was started on Neupogen , 400 mcg. subq. q.d.\\nHe was initially treated with antibiotic therapy .\\nChest x-ray showed questionable nodule in the right lower lobe , reasonably stable .\\nCalcium 8.7 , bilirubin 0.3/1.3 , creatinine 1.1 , glucose 128 .\\nHematocrit 24.6 .\\nWBC rose to 1.7 on 10/19 .\\nThe patient had some diarrhea .\\nThere was no diarrhea on 10/20 .\\nHe was feeling well and afebrile .\\nThe neutropenia resolved and he was felt to be in satisfactory condition on discharge on 10/20/95 .\\nHe was discharged home on Neupogen .\\n"
    start = 356
    end = 378
    start, end = 236,241


    model = GeminiModel("")
    predictor = ZeroShotPromptingModel(model)
    predictor.predict(summary=text, event_start=start, event_end=end, admission_time="", discharge_time="")
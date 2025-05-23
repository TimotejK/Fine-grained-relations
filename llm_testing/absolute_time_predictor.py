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
                "marked with <event> {point}. Provide your guess as a number of months, days, hours, or minutes. Answer in a JSON format where numbers are negative if event occurred before admission time.\n"
                "Example: {{\"months\": 0, \"days\": -3, \"hours\": 0, \"minutes\": 0}}\n\n"
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
                "marked with <event> tag {point}. Provide your guess as a number of months, days, hours, or minutes.\n\n"
                "{summary}\n")
            response = self.llm.predict(prompt.format(summary=summary_with_event, point=prediction_point))

            print(response)
            patterns = [r"PREDICTION: (\d+) (months?|days?|hours?|minutes?) (before|after)",
                        r"PREDICTION: (several) (months?|days?|hours?|minutes?) (before|after)",
                        r"(\d+) (months?|days?|hours?|minutes?) (before|after)",
                        r"(\d+) (months?|days?|hours?|minutes?)"]
            for search_pattern in patterns:
                # print(re.search(search_pattern, response).group())
                match = re.search(search_pattern, response)
                if match:
                    if match.group(1) == "several":
                        value = 3
                    else:
                        value = int(match.group(1))
                    unit = match.group(2).rstrip('s')  # Remove plural 's'
                    direction = "before"
                    if len(match.groups()) > 2:
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

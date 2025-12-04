import difflib
import re
from pydantic import BaseModel
import pandas as pd


class EventTimePredictorBatch:
    def __init__(self, llm_api, use_structured_response=False, use_absolute_times=True):
        self.llm = llm_api
        self.time_units_to_minutes = {
            "minute": 1,
            "hour": 60,
            "day": 1440,
            "month": 43200,  # Approximation (30 days in a month)
            "year": 525600  # Approximation (365 days in a year)
        }
        self.use_structured_response = use_structured_response
        self.use_absolute_times = use_absolute_times

    def prepare_prompt(self, document, events):
        """
        Prepares the input prompt by marking all events in the document.
        :param document: The full text of the document
        :param events: List of tuples (event_id, start, end) representing events
                       with start and end times.
                       in the document.
        :return: Tuple of (annotated_text, event_mapping, updated_events), where annotated_text
                 contains the document with annotated events, event_mapping
                 links event tags to event_ids, and updated_events includes start and end times.
                 links event tags (e.g., <eventX>) to event_ids.
        """
        # Sort events by start index for proper annotation
        events = sorted(events, key=lambda x: x[1])  # x[1]: start_char
        annotated_text = ""
        last_index = 0
        event_mapping = {}
        ground_truth = {}

        for i, (event_id, start, end,
                admission_date_minutes, discharge_date_minutes, start_time_minutes, start_lower_minutes,
                start_upper_minutes, end_time_minutes, end_lower_minutes, end_upper_minutes,
                duration_minutes, duration_lower_minutes, duration_upper_minutes) in enumerate(events):
            # Append text before this event
            annotated_text += document[last_index:start]
            # Annotate the text with the event
            event_tag = f"<event{i + 1}>"
            annotated_text += event_tag + document[start:end] + event_tag
            # Map the event tag to its event_id
            event_mapping[event_tag] = event_id
            event_mapping[f"event{i + 1}"] = event_id
            event_mapping[document[start:end]] = event_id
            ground_truth[event_id] = {
                "admission_date_minutes": admission_date_minutes, "discharge_date_minutes": discharge_date_minutes,
                "start_time_minutes": start_time_minutes, "start_lower_minutes": start_lower_minutes,
                "start_upper_minutes": start_upper_minutes, "end_time_minutes": end_time_minutes,
                "end_lower_minutes": end_lower_minutes, "end_upper_minutes": end_upper_minutes,
                "duration_minutes": duration_minutes, "duration_lower_minutes": duration_lower_minutes,
                "duration_upper_minutes": duration_upper_minutes
            }
            last_index = end

        # Append the remaining text after the last event
        annotated_text += document[last_index:]

        return annotated_text, event_mapping, ground_truth

    def predict_event_times_for_a_document(self, document, events, admission_time):
        """
        Predicts start and end times for all events in a document in one prompt.
        :param document: The full text of the document
        :param events: List of tuples (event_id, start_char, end_char)
                       representing events in the document.
        :param admission_time: Admission time in minutes to compute relative times.
        :return: List of predictions containing event_id, start_time_minutes, and end_time_minutes.
        """
        annotated_text, event_mapping, ground_truth = self.prepare_prompt(document, events)

        if self.use_structured_response:
            # Use structured response format
            if self.use_absolute_times:
                prompt = (
                    """
**You are given a patient discharge summary with annotated events.**
Each event is marked with a tag such as `<event1>`, `<event2>`, and so on.

**Task:** For each annotated event, predict when it **started** and **ended**. Specify the time in years, months, days, hours, and minutes.

**Output Format:** Return a JSON object containing an array with the following structure:

```json
"event_times": [
  {{
    "event_tag": "<event1>",
    "start": "2004-01-01 19:35:00",
    "end": "2004-01-31 08:12:00"
  }},
  {{
    "event_tag": "<event2>",
    "start": "2004-01-01 12:13:00",
    "end": "2004-01-01 12:20:00"
  }},
  ...
]
```

**Note:**

* Report absolute times of each event.
* If only the day of the event is defined in the text, try to estimate the time when the event is most likely to start and end.
* Ensure each event in the summary is represented in the output.
* As an event tag use literally <event1>, <event2>, and so on.

**Summary:**

```
{summary}
```
                    """
                )
            else:
                prompt = (
                    """
**You are given a patient discharge summary with annotated events.**
Each event is marked with a tag such as `<event1>`, `<event2>`, and so on.

**Task:** For each annotated event, predict when it **started** and **ended** relative to the **admission date**. Specify the time difference in years, months, days, hours, and minutes.

**Output Format:** Return a JSON object containing an array with the following structure:

```json
"event_times": [
  {{
    "event_tag": "<event1>",
    "start": {{ "years": 0, "months": 0, "days": -2, "hours": 0, "minutes": 0 }},
    "end":   {{ "years": 0, "months": 0, "days": -1, "hours": 0, "minutes": 0 }}
  }},
  {{
    "event_tag": "<event2>",
    "start": {{ "years": 0, "months": 0, "days": 0, "hours": 3, "minutes": 0 }},
    "end":   {{ "years": 0, "months": 0, "days": 0, "hours": 5, "minutes": 0 }}
  }},
  ...
]
```

**Note:**

* Use positive values if the event occurs *after* the admission date, and negative values if it occurs *before*.
* Ensure each event in the summary is represented in the output.
* As an event tag use literally <event1>, <event2>, and so on.

**Summary:**

```
{summary}
```
                    """
                )

            class TimePoint(BaseModel):
                years: int
                months: int
                days: int
                hours: int
                minutes: int
            class EventTime(BaseModel):
                event_tag: str
                start: TimePoint
                end: TimePoint
            class EventTimeISO(BaseModel):
                event_tag: str
                start: str
                end: str

            class EventTimes(BaseModel):
                event_times: list[EventTime]

            class EventTimesISO(BaseModel):
                event_times: list[EventTimeISO]

            if self.use_absolute_times:
                response = self.llm.predict_schema(
                    prompt=prompt.format(summary=annotated_text),
                    schema=EventTimesISO
                )
            else:
                response = self.llm.predict_schema(
                    prompt=prompt.format(summary=annotated_text),
                    schema=EventTimes
                )

            predictions = []
            if response is not None:
                for time in response.event_times:
                    event_tag = time.event_tag
                    start = time.start
                    end = time.end
                    BASE_DATETIME = pd.Timestamp("1900-01-01 00:00:00")
                    if self.use_absolute_times:
                        start_minutes = (pd.to_datetime(start
                                                       , errors='coerce') - BASE_DATETIME).total_seconds() // 60
                        end_minutes = (pd.to_datetime(end
                                                       , errors='coerce') - BASE_DATETIME).total_seconds() // 60
                        if pd.isna(start_minutes) or pd.isna(end_minutes):
                            # start_minutes = admission_time
                            # end_minutes = admission_time
                            start_minutes = 0
                            end_minutes = 0
                            # continue
                    else:
                        start_minutes = (
                                start.years * self.time_units_to_minutes["year"]
                                + start.months * self.time_units_to_minutes["month"]
                                + start.days * self.time_units_to_minutes["day"]
                                + start.hours * self.time_units_to_minutes["hour"]
                                + start.minutes
                        )
                        end_minutes = (
                                end.years * self.time_units_to_minutes["year"]
                                + end.months * self.time_units_to_minutes["month"]
                                + end.days * self.time_units_to_minutes["day"]
                                + end.hours * self.time_units_to_minutes["hour"]
                                + end.minutes
                        )
                    # Map the event tag back to the event_id
                    event_id = event_mapping.get(event_tag)
                    if event_id is None:
                        closest_key = difflib.get_close_matches(event_tag, event_mapping.keys(), n=1)
                        if closest_key:
                            event_id = event_mapping[closest_key[0]]
                        else:
                            event_id = None
                    if event_id is not None:
                        if self.use_absolute_times:
                            predictions.append({
                                "event_id": event_id,
                                # "predicted_start_time": f"{start.years}-{start.months}-{start.days} {start.hours}:{start.minutes}:00",
                                "predicted_start_time": start,
                                # "predicted_end_time": f"{end.years}-{end.months}-{end.days} {end.hours}:{end.minutes}:00",
                                "predicted_end_time": end,
                                "predicted_start_time_minutes": start_minutes,
                                "predicted_end_time_minutes": end_minutes
                            })
                        else:
                            predictions.append({
                                "event_id": event_id,
                                "predicted_start_time_minutes": start_minutes + admission_time,
                                "predicted_end_time_minutes": end_minutes + admission_time
                            })

            return predictions, ground_truth

        else:
            # Use plain text response format
            prompt = (
                "Below is a patient discharge summary with annotated events. "
                "For each event, write how much time before or after the admission date each event started and ended. Provide your answer "
                "in the format:\n"
                "event1: Start: X months, Y days, Z hours, W minutes after/before; "
                "End: A months, B days, C hours, D minutes after/before\n"
                "event2: ...\n\n"
                "Summary:\n{summary}\n"
            )

            response = self.llm.predict(prompt.format(summary=annotated_text))

            predictions = []
            try:
                response_data = eval(response) if isinstance(response, str) else response
            except (SyntaxError, NameError, TypeError):
                response_data = []  # Fallback to an empty list in case of a malformed response
            if isinstance(response_data, list) or isinstance(response_data, dict):  # Handle JSON list (new format)
                transformed_response = response_data
                if isinstance(response_data, dict):
                    transformed_response = []
                    for event_tag in response_data:
                        transformed_response.append({event_tag: response_data[event_tag]})

                for event in transformed_response:
                    for event_tag, time_string in event.items():
                        start_match = re.match(r".*Start: (\d+) months?, (\d+) days?, (\d+) hours?, (\d+) minutes? (before|after).*",
                                               time_string,
                                               re.IGNORECASE)
                        end_match = re.match(r".*End: (\d+) months?, (\d+) days?, (\d+) hours?, (\d+) minutes? (before|after).*",
                                         time_string,
                                         re.IGNORECASE)

                        start_total_minutes = 0
                        end_total_minutes = 0
                        if start_match:
                            start_months = int(start_match.group(1))
                            start_days = int(start_match.group(2))
                            start_hours = int(start_match.group(3))
                            start_minutes = int(start_match.group(4))
                            start_direction = start_match.group(5)
                            start_total_minutes = (
                                    start_months * self.time_units_to_minutes["month"] +
                                    start_days * self.time_units_to_minutes["day"] +
                                    start_hours * self.time_units_to_minutes["hour"] +
                                    start_minutes
                            )
                            if start_direction.lower() == "before":
                                start_total_minutes = -start_total_minutes

                        if end_match:
                            end_months = int(end_match.group(1))
                            end_days = int(end_match.group(2))
                            end_hours = int(end_match.group(3))
                            end_minutes = int(end_match.group(4))
                            end_direction = end_match.group(5)
                            end_total_minutes = (
                                    end_months * self.time_units_to_minutes["month"] +
                                    end_days * self.time_units_to_minutes["day"] +
                                    end_hours * self.time_units_to_minutes["hour"] +
                                    end_minutes
                            )
                            if end_direction.lower() == "before":
                                end_total_minutes = -end_total_minutes

                        # Map the event tag back to the event_id
                        event_id = event_mapping[event_tag]
                        prediction = {
                            "event_id": event_id,
                            "start_time_minutes": (start_total_minutes if start_match else 0) + admission_time,
                            "end_time_minutes": (end_total_minutes if end_match else 0) + admission_time
                        }
                        predictions.append(prediction)
            else:  # Handle plain text response
                for line in response.splitlines():
                    match = re.match(r"(event\d+): (\d+) (months?|days?|hours?|minutes?) (before|after)", line,
                                     re.IGNORECASE)
                    if match:
                        event_tag = match.group(1)
                        value = int(match.group(2))
                        unit = match.group(3).rstrip('s')  # Remove 's' for plural
                        direction = match.group(4)

                        # Convert to minutes
                        minutes = value * self.time_units_to_minutes[unit.lower()]

                        if direction.lower() == "before":
                            minutes = -minutes

                        # Map the event tag back to the event_id
                        event_id = event_mapping[event_tag]
                        predictions.append({
                            "event_id": event_id,
                            "time_minutes": minutes + admission_time
                        })
                        prediction = {
                            "event_id": event_id,
                            "start_time_minutes": minutes + admission_time,
                            "end_time_minutes": minutes + admission_time
                        }
                        predictions.append(prediction)

            return predictions, ground_truth

    def predict(self, dataframe):
        """
        Groups the DataFrame by document_id, processes each document, and predicts
        times for all events in the document.
        :param dataframe: Input DataFrame with columns: ['document_id', 'event_id',
                          'start_char', 'end_char', 'text', 'admission_date_minutes']
        :return: DataFrame with predictions
        """
        # Group by document_id to process all events for each document
        grouped = dataframe.groupby("document_id")

        all_predictions = []
        for document_id, group in grouped:
            # Extract required fields
            document_text = group.iloc[0]["text"]
            admission_time = group.iloc[0]["admission_date_minutes"]
            events = group[["event_id", "start_char", "end_char", "admission_date_minutes", "discharge_date_minutes", "start_time_minutes", "start_lower_minutes", "start_upper_minutes", "end_time_minutes", "end_lower_minutes", "end_upper_minutes", "duration_minutes", "duration_lower_minutes", "duration_upper_minutes"]].values.tolist()

            # Predict times for all events in this document
            predictions, ground_truth = self.predict_event_times_for_a_document(document_text, events, admission_time)

            for i in range(len(predictions)):
                predictions[i].update(ground_truth[predictions[i]["event_id"]])
                pass

            yield document_id, predictions
            # Store predictions
            for prediction in predictions:
                all_predictions.append({
                    "document_id": document_id,
                    **prediction
                })

        # Convert results to DataFrame
        # return pd.DataFrame(all_predictions)

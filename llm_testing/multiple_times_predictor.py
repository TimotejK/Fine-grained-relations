import re
from pydantic import BaseModel
import pandas as pd


class EventTimePredictor:
    def __init__(self, llm_api, use_structured_response=False):
        self.llm = llm_api
        self.time_units_to_minutes = {
            "minute": 1,
            "hour": 60,
            "day": 1440,
            "month": 43200,  # Approximation (30 days in a month)
            "year": 525600  # Approximation (365 days in a year)
        }
        self.use_structured_response = use_structured_response

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

        for i, (event_id, start, end) in enumerate(events):
            # Append text before this event
            annotated_text += document[last_index:start]
            # Annotate the text with the event
            event_tag = f"<event{i + 1}>"
            annotated_text += event_tag + document[start:end] + event_tag
            # Map the event tag to its event_id
            event_mapping[event_tag] = event_id
            event_mapping[f"event{i + 1}"] = event_id
            last_index = end

        # Append the remaining text after the last event
        annotated_text += document[last_index:]

        return annotated_text, event_mapping

    def predict_event_times_for_a_document(self, document, events, admission_time):
        """
        Predicts start and end times for all events in a document in one prompt.
        :param document: The full text of the document
        :param events: List of tuples (event_id, start_char, end_char)
                       representing events in the document.
        :param admission_time: Admission time in minutes to compute relative times.
        :return: List of predictions containing event_id, start_time_minutes, and end_time_minutes.
        """
        annotated_text, event_mapping = self.prepare_prompt(document, events)

        if self.use_structured_response:
            # Use structured response format
            prompt = (
                "Below is a patient discharge summary with annotated events. "
                "For each event marked as <event1>, <event2>, ..., predict how long "
                "before or after the admission date each event STARTED and ENDED. "
                "before or after the admission date each event occurred. Provide your answer "
                "as a JSON list, where each value is an object "
                "[{{\"event_tag\": \"<event1>\", \"start\": {{\"years\": 0, \"months\": 0, \"days\": 0, \"hours\": 0, \"minutes\": 0}}, "
                "\"end\": {{\"years\": 0, \"months\": 0, \"days\": 0, \"hours\": 0, \"minutes\": 0}} }}].\n\n"
                "Summary:\n{summary}\n"
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

            response = self.llm.predict_schema(
                prompt=prompt.format(summary=annotated_text),
                schema=list[EventTime]
            )

            predictions = []
            for event_tag, time_data in response.items():
                start_minutes = (
                        time_data["start"]["years"] * self.time_units_to_minutes["year"]
                        + time_data["start"]["months"] * self.time_units_to_minutes["month"]
                        + time_data["start"]["days"] * self.time_units_to_minutes["day"]
                        + time_data["start"]["hours"] * self.time_units_to_minutes["hour"]
                        + time_data["start"]["minutes"]
                )
                end_minutes = (
                        time_data["end"]["years"] * self.time_units_to_minutes["year"]
                        + time_data["end"]["months"] * self.time_units_to_minutes["month"]
                        + time_data["end"]["days"] * self.time_units_to_minutes["day"]
                        + time_data["end"]["hours"] * self.time_units_to_minutes["hour"]
                        + time_data["end"]["minutes"]
                )
                # Map the event tag back to the event_id
                event_id = event_mapping[event_tag]
                predictions.append({
                    "event_id": event_id,
                    "start_time_minutes": start_minutes + admission_time,
                    "end_time_minutes": end_minutes + admission_time
                })

            return predictions

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

            return predictions

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
            events = group[["event_id", "start_char", "end_char"]].values.tolist()

            # Predict times for all events in this document
            predictions = self.predict_event_times_for_a_document(document_text, events, admission_time)

            yield document_id, predictions
            # Store predictions
            for prediction in predictions:
                all_predictions.append({
                    "document_id": document_id,
                    "event_id": prediction["event_id"],
                    "start_time_minutes": prediction["start_time_minutes"],
                    "end_time_minutes": prediction["end_time_minutes"]
                })

        # Convert results to DataFrame
        # return pd.DataFrame(all_predictions)

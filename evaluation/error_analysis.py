def save_for_error_analysis(results, output_dir, filename):
    """
    Save the results of the error analysis to a file.

    Args:
        results (list): A list of dictionaries containing the results of the error analysis.
        output_dir (str): The directory where the results will be saved.
        filename (str): The name of the file to save the results to.
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(results, f, indent=4)
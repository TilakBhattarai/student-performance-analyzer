import numpy as np


def load_data(path):
    student_data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(2, 3, 4))

    student_names = np.genfromtxt(
        path, delimiter=",", skip_header=1, usecols=1, dtype=str
    )
    header_content = np.genfromtxt(path, delimiter=",", max_rows=1, dtype=str)
    subject_names = header_content[2:]

    return student_data, student_names, subject_names


def inspect_data(student_data):
    print("Shape: ", student_data.shape)
    print("Dimension: ", student_data.ndim)
    print("Total Values: ", student_data.size)
    print("Data Type: ", student_data.dtype)


def analyze_scores(student_data, student_names, subject_names, weights):
    subject_avg_marks = np.nanmean(student_data, axis=0)

    missing_value = np.isnan(student_data)
    missing_value_count = missing_value.sum()
    student_cleaned_data = np.where(missing_value, subject_avg_marks, student_data)

    student_avg_marks = np.nanmean(student_data, axis=1)

    high_scoring_students = student_avg_marks >= 75
    high_performing_subject = np.argmax(subject_avg_marks)
    best_subject = subject_names[high_performing_subject]

    high_scoring_names = student_names[high_scoring_students]
    high_scoring_averages = student_avg_marks[high_scoring_students]

    highest_individual_mark = np.nanmax(student_data)

    weighted_scores = student_cleaned_data @ weights

    return (
        subject_avg_marks,
        student_avg_marks,
        weighted_scores,
        high_scoring_names,
        high_scoring_averages,
        best_subject,
        highest_individual_mark,
        missing_value_count,
        student_cleaned_data,
    )


def normalize_scores(student_cleaned_data):
    subject_means = np.mean(student_cleaned_data, axis=0)
    student_stds = np.std(student_cleaned_data, axis=0)

    normalized_scores = (student_cleaned_data - subject_means) / student_stds

    return normalized_scores


def save_results(path, normalized_scores):

    np.savetxt(
        path,
        normalized_scores,
        header="Math,Science,English",
        delimiter=",",
        fmt="%.2f",
        comments="",
    )


def main():
    student_data, student_names, subject_names = load_data("./data/scores.csv")

    inspect_data(student_data)

    weights = np.array([0.40, 0.35, 0.25])

    (
        subject_avg_marks,
        student_avg_marks,
        weighted_scores,
        high_scoring_names,
        high_scoring_averages,
        best_subject,
        highest_individual_mark,
        missing_value_count,
        student_cleaned_data,
    ) = analyze_scores(student_data, student_names, subject_names, weights)

    print("Subject averages:", subject_avg_marks)
    print("Student averages:", student_avg_marks)
    print("Weighted scores:", weighted_scores)
    print("High-performing students:", high_scoring_names)
    print("Their averages:", high_scoring_averages)
    print("Best subject:", best_subject)
    print("Highest individual mark:", highest_individual_mark)
    print("Missing values:", missing_value_count)

    normalized_scores = normalize_scores(student_cleaned_data)
    save_results("./data/normalized_score_output.csv", normalized_scores)


if __name__ == "__main__":
    main()

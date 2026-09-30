import os
import json
import argparse

from Levenshtein import ratio


def clear_text(text):
    return text.lower().strip().strip('.!?\'",”')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('output_dir', type=str)
    parser.add_argument('--tasks', type=str, nargs='*', default='mmlu lambada siqa hellaswag')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    TASKS = args.tasks
    OUTPUT_DIR = args.output_dir

    tasks_accuracy = dict()

    for task in TASKS:
        print(OUTPUT_DIR, task)
        with open(os.path.join(OUTPUT_DIR, task+'.jsonl')) as f:
            lines = f.readlines()
            entries = list(map(lambda x: json.loads(x), lines))

        correct_ids = list()
        incorrect_ids = list()

        for i in range(len(entries)):
            entries[i]['other'] = ' '.join(entries[i]['generate'].split()[1:])
            entries[i]['generate'] = entries[i]['generate'].split()[0]

            if clear_text(entries[i]['generate']) == clear_text(entries[i]['ground_truth']):
                correct_ids.append(entries[i]['id'])
            else:
                incorrect_ids.append(entries[i]['id'])

        acc = len(correct_ids) / (len(correct_ids) + len(incorrect_ids))
        tasks_accuracy[task] = acc


    for task, accuracy in tasks_accuracy.items():
        print(f'{task} accuracy = {accuracy:.4f}')

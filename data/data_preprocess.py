import pandas as pd


def data_preprocess(data_path: str) -> None:
    df = pd.read_csv(data_path)
    prev_id = None
    new_df = pd.DataFrame(columns=['id', 'predicted'])

    predictions = []
    for i, row in df.iterrows():
        if prev_id is not None and prev_id != row['Sample']:
            new_df = new_df.append({'id': prev_id, 'predicted': predictions}, ignore_index=True)
            predictions = []
        predictions.append(1 if row['Predicted'] else 0)
        prev_id = row['Sample']
        if i == len(df) - 1:
            new_df = new_df.append({'id': prev_id, 'predicted': predictions}, ignore_index=True)
    new_df.to_csv('train_solutions_preprocessed.csv', index=False, header=True)


if __name__ == '__main__':
    data_preprocess('train_solutions.csv')

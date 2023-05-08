import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm


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
  

def resize_images(image_path: str):      
    image = Image.open(file_path)
      
    # resized = image.resize((1920, 1080))
    resized = image.resize((1280, 720))
    esized.save(file_path.replace('/images/', '/images_compressed/'))
    return


if __name__ == '__main__':
    data_preprocess('train_solutions.csv')
    for file in tqdm(Path('/home/turib/multi_label_image_classification/data/images').rglob(f'*_orig.jpg')):
    resize_images(str(file))

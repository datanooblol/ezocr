from glob import glob
import random
import os
import shutil

def sampling(base, train=0.7, val=0.2, test=0.1):
    """
    Split the input list into train, validation, and test sets without repetition.
    
    Args:
        base (list): The list of elements to split.
        train (float): Proportion of data for the training set.
        val (float): Proportion of data for the validation set.
        test (float): Proportion of data for the test set.
    
    Returns:
        tuple: Three lists representing the train, validation, and test sets.
    """
    if not (0 <= train <= 1 and 0 <= val <= 1 and 0 <= test <= 1):
        raise ValueError("Proportions must be between 0 and 1.")
    
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Proportions must sum to 1.")
    
    # Shuffle the base list
    shuffled = base[:]
    random.shuffle(shuffled)
    
    # Calculate the split indices
    n = len(base)
    train_end = int(n * train)
    val_end = train_end + int(n * val)
    
    # Split the data
    trainset = shuffled[:train_end]
    valset = shuffled[train_end:val_end]
    testset = shuffled[val_end:]
    
    return trainset, valset, testset

def save_dataset(target_path, trainset, valset, testset):
    objectives = ['train', 'val', 'test']
    for path in objectives:
        os.makedirs(f"{target_path}/{path}/images", exist_ok=True)
        os.makedirs(f"{target_path}/{path}/labels", exist_ok=True)
    for objective, dataset in dict(zip(objectives, [trainset, valset, testset])).items():
        for data in dataset:
            file_name = os.path.basename(data).split('.')[0]
            dir_name = os.path.dirname(data)
            shutil.copy(f"""{os.path.join(*dir_name.split("/")[0:-1], "images", f"{file_name}.jpg")}""", f"""{os.path.join(*target_path.split("/"), objective, "images", f"{file_name}.jpg")}""")
            shutil.copy(f"""{os.path.join(*dir_name.split("/")[0:-1], "labels", f"{file_name}.txt")}""", f"""{os.path.join(*target_path.split("/"), objective, "labels", f"{file_name}.txt")}""")

    print(f"moved successfully to {target_path}: {len(trainset)+len(valset)+len(testset)} files")
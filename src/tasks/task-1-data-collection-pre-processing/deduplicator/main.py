import json
import hashlib
from glob import glob
from tqdm import tqdm
from typing import List

import fire


def calculate_md5_hash(file_path: str) -> str:
    """
    Calculate md5 hash.
    
    Args:
        file_path: str. full path

    Returns:
        str. hash
    """
    return hashlib.md5(open(file_path,'rb').read()).hexdigest()


def find_duplicates_md5(image_list: List) -> dict:
    """
    Find duplicates by calculating & comparing md5 hashes.
    
    Args:
        image_list: List. list of images with absolute paths.

    Returns:
        dict. dictionary of duplicates. e.g. {ref_image: [duplicate_1, duplicate_2, ..]}
    """
    find_occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)

    md5_list = [calculate_md5_hash(image_path) for image_path in tqdm(image_list)]

    duplicates = {}

    processed_items = []  # to process each image only once.
    for ref_img_path, ref_md5 in tqdm(zip(image_list, md5_list)):

        if ref_img_path not in processed_items:
            processed_items.append(ref_img_path)
            
            for sim_img_ind in list(find_occurrences(ref_md5, md5_list)):
                sim_img_path = image_list[sim_img_ind]
                
                if sim_img_path == ref_img_path or sim_img_path in processed_items:
                    continue
                tmp_duplicate_list = duplicates.get(ref_img_path, [])
                tmp_duplicate_list.append(sim_img_path)
                duplicates[ref_img_path] = tmp_duplicate_list
                processed_items.append(sim_img_path)
    return duplicates


def run(img_dir: str) -> None:
    """
    Main function to process images in a directory.
    
    Args:
        img_dir: str. full path with wildcards. e.g. '/data/tmp/*.jpg'
    
    Returns:
        dict. dictionary of duplicates. e.g. {ref_image: [duplicate_1, duplicate_2, ..]}
    """
    image_list = glob(img_dir)
    duplicates = find_duplicates_md5(image_list)
    # print(json.dumps(duplicates, indent=2))
    return duplicates


if __name__ == "__main__":
    fire.Fire()

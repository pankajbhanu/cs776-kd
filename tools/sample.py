import json
import os
from collections import defaultdict
import random
from pycocotools.coco import COCO
import shutil

def generate_proportional_coco_subset(full_coco_annotation_file, output_annotation_file, output_image_dir, num_subset_images, full_image_dir):
    """
    Generates a subset of the COCO dataset with proportional object categories.

    Args:
        full_coco_annotation_file (str): Path to the full COCO annotation JSON file.
        output_annotation_file (str): Path to save the subset annotation JSON file.
        output_image_dir (str): Path to save the subset images.
        num_subset_images (int): Desired number of images in the subset.
        full_image_dir (str): Path to the directory containing the full COCO images.
    """
    coco = COCO(full_coco_annotation_file)
    cat_ids = coco.getCatIds()
    all_img_ids = coco.getImgIds()

    # Count the number of instances for each category in the full dataset
    category_counts_full = defaultdict(int)
    annotations = coco.loadAnns(coco.getAnnIds())
    for ann in annotations:
        category_counts_full[ann['category_id']] += 1

    total_annotations_full = len(annotations)
    if total_annotations_full == 0:
        print("No annotations found in the full dataset.")
        return

    # Calculate the proportion of each category
    category_proportions = {}
    for cat_id in cat_ids:
        category_proportions[cat_id] = category_counts_full[cat_id] / total_annotations_full

    # Calculate the average number of annotations per image in the full dataset
    num_full_images = len(all_img_ids)
    if num_full_images == 0:
        print("No images found in the full dataset.")
        return
    avg_anns_per_image = total_annotations_full / num_full_images

    # Estimate the target number of annotations in the subset
    num_subset_annotations = int(num_subset_images * avg_anns_per_image)

    # Calculate the target number of annotations for each category in the subset
    target_category_counts_subset = defaultdict(int)
    for cat_id, proportion in category_proportions.items():
        target_category_counts_subset[cat_id] = int(proportion * num_subset_annotations)

    print(f"Total annotations in full dataset: {total_annotations_full}")
    print(f"Number of images in full dataset: {num_full_images}")
    print(f"Average annotations per image: {avg_anns_per_image:.2f}")
    print(f"Desired number of subset images: {num_subset_images}")
    print(f"Estimated number of subset annotations: {num_subset_annotations}")
    print("Target category counts in subset:")
    for cat_id, count in target_category_counts_subset.items():
        cat_info = coco.loadCats(ids=[cat_id])[0]
        print(f"  {cat_info['name']}: {count}")

    selected_annotations = []
    current_category_counts_subset = defaultdict(int)
    available_annotations = list(annotations)
    random.shuffle(available_annotations)

    # Greedily select annotations to meet the target counts
    for ann in available_annotations:
        cat_id = ann['category_id']
        if current_category_counts_subset[cat_id] < target_category_counts_subset[cat_id]:
            selected_annotations.append(ann)
            current_category_counts_subset[cat_id] += 1
            if len(selected_annotations) >= num_subset_annotations:
                break

    selected_image_ids = set(ann['image_id'] for ann in selected_annotations)
    selected_images = coco.loadImgs(list(selected_image_ids))
    selected_categories = coco.loadCats(cat_ids) # Keep all original categories

    # Create the subset annotation JSON
    subset_coco_data = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": selected_categories
    }

    os.makedirs(os.path.dirname(output_annotation_file), exist_ok=True)
    with open(output_annotation_file, 'w') as f:
        json.dump(subset_coco_data, f)

    # Copy the selected images
    os.makedirs(output_image_dir, exist_ok=True)
    for img_data in selected_images:
        image_filename = img_data['file_name']
        full_image_path = os.path.join(full_image_dir, image_filename)
        output_image_path = os.path.join(output_image_dir, image_filename)
        if os.path.exists(full_image_path):
            shutil.copy(full_image_path, output_image_path)
        else:
            print(f"Warning: Image not found: {full_image_path}")

    print(f"\nSubset generated with {len(selected_images)} images and {len(selected_annotations)} annotations.")
    print(f"Annotations saved to: {output_annotation_file}")
    print(f"Images saved to: {output_image_dir}")

if __name__ == '__main__':
    # Example usage:
    # full_annotation_file = '/Users/junth/temp/coco-minitrain/src/data/coco/annotations/instances_train2017.json'
    # full_image_directory = '/Users/junth/temp/coco-minitrain/src/data/coco/train2017'
    # output_annotation_path = '/Users/junth/temp/coco-minitrain/src/sample_data/coco/annotations/instances_train2017.json'
    # output_image_directory = '/Users/junth/temp/coco-minitrain/src/sample_data/coco/train2017'
    full_annotation_file = '/users/phd/bpankaj24/proj/cs776-kd/dataset/coco25k/instances_minitrain2017.json'
    full_image_directory = '/users/phd/bpankaj24/proj/cs776-kd/dataset/coco25k/images'
    output_annotation_path = '/users/phd/bpankaj24/proj/cs776-kd/dataset/sample_mini/sample_instances_train2017.json'
    output_image_directory = '/users/phd/bpankaj24/proj/cs776-kd/dataset/sample_mini/sample_train2017'
    num_desired_images = 1000  # Specify the desired number of images in the subset

    generate_proportional_coco_subset(full_annotation_file, output_annotation_path, output_image_directory, num_desired_images, full_image_directory)

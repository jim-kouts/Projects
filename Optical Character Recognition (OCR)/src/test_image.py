'''
test_image.py
Script 5/7 - Test loading and saving an image from the dataset to verify that the image data is correctly processed and can be accessed as expected.
'''

from datasets import load_from_disk
ds = load_from_disk("data/raw/funsd_subset")
ds["test"][0]["image"].save("test_doc.png")
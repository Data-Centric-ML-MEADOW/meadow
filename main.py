import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from wilds import get_dataset

dataset = get_dataset(dataset="iwildcam", download=True)

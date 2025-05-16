>>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
>>> dataset.get_labels()
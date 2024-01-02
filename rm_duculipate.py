import os

path = 'runs/segment\\predict9\\runs/segment\\predict9\\hyc.jpg'
normalized_path = os.path.normpath(path)
unique_paths = set(normalized_path.split(os.path.sep))

result_path = os.path.sep.join(unique_paths)
print(result_path)
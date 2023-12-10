# simple script to rename and sort files in the object-training dataset since 
# some image object names were too long for git and sorting
# Jacob Hammond
import os

def rename_sort_files():
    count = 1
    # sort training images first
    directory_paths = ['./train/images', './train/labels'] 
    for directory in directory_paths:
        files = os.listdir(directory)
        files.sort()
        for file in files:
            file_extension = os.path.splitext(file)[1]
            new_name = str(count) + file_extension
            os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
            count += 1

    # sort validation images first, keep counter running
    directory_paths = ['./valid/images', './valid/labels'] 
    for directory in directory_paths:
        files = os.listdir(directory)
        files.sort()
        for file in files:
            file_extension = os.path.splitext(file)[1]
            new_name = str(count) + file_extension
            os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
            count += 1
    
    # sort test images last, keep counter running
    directory_paths = ['./test/images', './test/labels'] 
    for directory in directory_paths:
        files = os.listdir(directory)
        files.sort()
        for file in files:
            file_extension = os.path.splitext(file)[1]
            new_name = str(count) + file_extension
            os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
            count += 1

if __name__ == '__main__':
    rename_sort_files()


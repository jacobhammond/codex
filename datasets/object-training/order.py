# simple script to rename and sort files in the object-training dataset since 
# some image object names were too long for git and sorting
# Jacob Hammond
import os

def rename_sort_files():
    count = 1
    # sort training images first
    directory_paths = ['./train/images', './train/labels'] 
    file1 = os.listdir(directory_paths[0])
    file2 = os.listdir(directory_paths[1])
    file1.sort()
    file2.sort()
    for a_file, b_file in zip(file1, file2):
        exta = os.path.splitext(a_file)[1]
        extb = os.path.splitext(b_file)[1]
        new1 = str(count) + exta
        new2 = str(count) + extb
        os.rename(os.path.join(directory_paths[0], a_file), os.path.join(directory_paths[0], new1))
        os.rename(os.path.join(directory_paths[1], b_file), os.path.join(directory_paths[1], new2))
        count += 1

    # sort validation images first, keep counter running
    directory_paths = ['./valid/images', './valid/labels'] 
    file1 = os.listdir(directory_paths[0])
    file2 = os.listdir(directory_paths[1])
    file1.sort()
    file2.sort()
    for a_file, b_file in zip(file1, file2):
        exta = os.path.splitext(a_file)[1]
        extb = os.path.splitext(b_file)[1]
        new1 = str(count) + exta
        new2 = str(count) + extb
        os.rename(os.path.join(directory_paths[0], a_file), os.path.join(directory_paths[0], new1))
        os.rename(os.path.join(directory_paths[1], b_file), os.path.join(directory_paths[1], new2))
        count += 1
    
    # sort test images last, keep counter running
    directory_paths = ['./test/images', './test/labels'] 
    file1 = os.listdir(directory_paths[0])
    file2 = os.listdir(directory_paths[1])
    file1.sort()
    file2.sort()
    for a_file, b_file in zip(file1, file2):
        exta = os.path.splitext(a_file)[1]
        extb = os.path.splitext(b_file)[1]
        new1 = str(count) + exta
        new2 = str(count) + extb
        os.rename(os.path.join(directory_paths[0], a_file), os.path.join(directory_paths[0], new1))
        os.rename(os.path.join(directory_paths[1], b_file), os.path.join(directory_paths[1], new2))
        count += 1

def shorten_filenames():

    #for all files in train, valid, and test directories
    # recursively search through all directories and shorten all filenames to 16 characters or less

    directory_paths = ['./train', './valid', './test']
    for directory in directory_paths:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                # get the file extension
                ext = os.path.splitext(filename)[1]
                # get the file name without extension
                name = os.path.splitext(filename)[0]
                # if the file name is longer than 16 characters, shorten it
                if len(name) > 16:
                    new_name = name[-16:] + ext
                    os.rename(os.path.join(root, filename), os.path.join(root, new_name))




if __name__ == '__main__':
    rename_sort_files()
    #shorten_filenames()


import csv
import os

from sklearn.model_selection import train_test_split


def list_files(dir: str, file_extension: str) -> list:
    """give a directory,list all files with given extension"""
    if not os.path.isdir(dir):
        return None

    files = os.listdir(dir)
    return files


def get_image_label_pairs(image_dir: str, label: str) -> tuple:
    """assuming the image dir contains a single label image, create two lists.
    the first list contains filenames, the second list contains label for corresponding image
    e.g. (file1,file2,...n)
    """

    filenames = list_files(
        image_dir, ""
    )  # Update the file extension as per  requirement
    labels = [label] * len(filenames)
    return filenames, labels

def read_as_csv(csv_file):
    image_path= []
    labels= []
    with open(csv_file , 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_path.append(row[0])
            labels.append(row[1])
    return image_path, labels


def save_as_csv(image_paths, labels, outfile):
    """Assume image_paths = [file1, file2, ...filen] and labels = [label1,label2...labelk]
    Save a CSV file with a default name 'output.csv' such that each row contains:
    file1, label1
    file2, label2
    """

    # outfile = 'output.csv'

    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "label"])

        for image_path, label in zip(image_paths, labels):
            writer.writerow([image_path, label])

def save_prediction_as_csv(test_files, y_tests,y_preds, outfile):
    """Assume image_paths = [file1, file2, ...filen] and labels = [label1,label2...labelk]
    Save a CSV file with a default name 'output.csv' such that each row contains:
    file1, label1
    file2, label2
    """

    # outfile = 'output.csv'

    with open(outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["test_file", "y_test","y_pred"])

        for test_file,y_test,y_pred  in zip(test_files, y_tests,y_preds):
            writer.writerow([test_file,y_test,y_pred])



# if __name__=='__main__':
        # images_No_DR,labels_No_DR=get_image_label_pairs(r'data\No_DR','No_DR')
        # images_Mild,labels_Mild=get_image_label_pairs(r'data\Mild','Mild')
        # images_Moderate,labels_Moderate=get_image_label_pairs(r'data\Moderate','Moderate')
        # images_Proliferate_DR,labels_Proliferate_DR=get_image_label_pairs(r'data\Proliferate_DR','Proliferate_DR')
        # images_Severe,labels_Severe=get_image_label_pairs(r'data\Severe','Severe')
        
        

        # save_as_csv(images_No_DR,labels_No_DR,'data/No_DR.csv')
        # save_as_csv(images_Mild,labels_Mild,'data/Mild.csv')
        # save_as_csv(images_Moderate,labels_Moderate,'data/Moderate.csv')
        # save_as_csv(images_Proliferate_DR,labels_Proliferate_DR,'data/Proliferate_DR.csv')
        # save_as_csv(images_Severe,labels_Severe,'data/Severe.csv')
        

        # x = []
        # y = []
        # folders = ["No_DR", "Mild", "Moderate", "Proliferate_DR","Severe"]
        # for i in folders:
        #     images_path, label = get_image_label_pairs(f"data/{i}", f"{i}")
        #     x.extend(images_path)
        #     y.extend(label)
        
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        # save_as_csv(x_train, y_train, "data/train.csv")
        # save_as_csv(x_test, y_test, "data/test.csv")
        
if __name__ == '__main__':
    
    categories = ["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"]
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for category in categories:
        images, labels = get_image_label_pairs(f"data/{category}", category)
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        
        train_images.extend(x_train)
        train_labels.extend(y_train)
        test_images.extend(x_test)
        test_labels.extend(y_test)

    save_as_csv(train_images, train_labels, "data/train.csv")
    save_as_csv(test_images, test_labels, "data/test.csv")

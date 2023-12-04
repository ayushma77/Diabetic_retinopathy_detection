import pandas as pd

# Load the merged CSV file
merged_csv_path = "data\merged_train.csv"  # Update with the actual path
merged_df = pd.read_csv(merged_csv_path)

# Count the number of rows (which corresponds to the number of images)
num_images = len(merged_df)

print("Number of images in merged_train.csv:", num_images)
mild_csv_path=r"data\Mild.csv"
moderate_csv_path=r"data\Moderate.csv"
No_DR_csv_path=r"data\No_DR.csv"
Proliferate_csv_path=r"data\Proliferate_DR.csv"
Severe_csv_path=r"data\Severe.csv"

mild_df = pd.read_csv(mild_csv_path)
moderate_df = pd.read_csv(moderate_csv_path)
No_DR_df = pd.read_csv(No_DR_csv_path)
Proliferate_df = pd.read_csv(Proliferate_csv_path)
Severe_df = pd.read_csv(Severe_csv_path)
num_images_mild = len(mild_df)
num_images_moderate = len(moderate_df)
num_images_no_dr = len(No_DR_df)
num_images_Proliferate = len(Proliferate_df)
num_images_Severe = len(Severe_df)




print("Number of images in mild.csv:", num_images_mild)
print("Number of images in moderate.csv:", num_images_moderate)
print("Number of images in no_dr.csv:", num_images_no_dr)
print("Number of images in proliferate.csv:", num_images_Proliferate)
print("Number of images in severe.csv:", num_images_Severe)








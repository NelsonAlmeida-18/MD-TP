import re
import os
import shutil

# Create a directory named "ADoutputs" if it doesn't exist
output_directory = "ADoutputs"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
else:
    # Clear the directory if it already exists
    shutil.rmtree(output_directory)
    os.makedirs(output_directory)

# Read the text from the .txt file
with open("programas_texts/AD.txt", "r") as file:
    text = file.read()

# Use regular expressions to capture the names occurring before "1. Porque é preciso mudar"
name_matches = re.findall(r'(?<=1\. Porque é preciso mudar\n)(.*?)(?=\n)', text)
if name_matches:
    names = [name.strip() for name in name_matches]
else:
    names = ["Unknown"]

# Use regular expressions to capture the text between the occurrences of "1. Porque é preciso mudar"
matches = re.findall(r'1\. Porque é preciso mudar\n(.*?)(?=\n1\. Porque é preciso mudar|\Z)', text, re.DOTALL)

# Save the captured text into different .txt files in the "ADoutputs" directory
for name, match in zip(names, matches):
    filename = f"{name}.txt"
    with open(os.path.join(output_directory, filename), "w") as output_file:
        output_file.write(match.strip())

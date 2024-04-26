def remove_lixo(file_path, string_to_remove):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Remove all occurrences of the specified string
    modified_content = content.replace(string_to_remove, '')

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(modified_content)

def remove_text_before_sumario(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sumario_index = -1
    for i, line in enumerate(lines):
        if '248Sumário ' in line:
            sumario_index = i
            break

    if sumario_index != -1:
        content_after_sumario = ''.join(lines[sumario_index:])
        with open('programas_texts/AD.txt', 'w', encoding='utf-8') as new_file:
            new_file.write(content_after_sumario)
        print("Text before '248Sumário ' removed. Saved to 'programas_texts/AD.txt'")
    else:
        print("No '248Sumário ' found in the file.")

def main():
    file_path = 'programas_texts/AD.txt'  # Replace 'your_file.txt' with the path to your .txt file
    string_to_remove = 'ELEITORAL20'

    remove_lixo(file_path, string_to_remove)
    remove_text_before_sumario(file_path)

if __name__ == "__main__":
    main()

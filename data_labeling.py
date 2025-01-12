import zipfile
import os


def process_zip(zip_path, output_file):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            files = [f for f in file_list if not f.endswith('/')]

            output_lines = []
            for file in files:
                filename = os.path.basename(file)

                if filename.endswith(".wav"):
                    parts = filename.split('-')
                    if len(parts) >= 3:
                        word = parts[0]
                        output_lines.append(f"{filename}\t{word}")
            with open(output_file, 'w') as f:
                f.write("\n".join(output_lines))

            print(f"Output written to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


zip_path = 'proj_dataset-20250108T145657Z-001.zip'
output_file = 'transcript.txt'
process_zip(zip_path, output_file)

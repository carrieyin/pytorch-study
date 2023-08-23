from pytorch.src.seq_2_seq.config import raw_file_path, raw_zip_path
from pytorch.src.seq_2_seq.preprocess_data import download_extract, extract_content, preprocess_nmt, tokenize_nmt



if __name__ == '__main__':
    content = extract_content()
    data = preprocess_nmt(content)
    #print(data)
    token_data = tokenize_nmt(data, 600)
    print(token_data)
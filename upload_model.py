from huggingface_hub import create_repo
from huggingface_hub import notebook_login
from huggingface_hub import HfApi



class UploadConfig:
    Wtoken: str = ''
    RNAME: str = ''
    FNAME: str = ''
    TYPE: str = 'model'



def main():
    # Initialize api
    config = UploadConfig()
    api = HfApi()
    # Create repository
    create_repo(config.RNAME, token = config.Wtoken)
    api.upload_folder(
        folder_path = config.FNAME, 
        repo_id = config.RNAME, 
        repo_type = config.TYPE, 
        path_in_repo = "./", 
        token = config.Wtoken
    )

if __name__ == "__main__":
    main()

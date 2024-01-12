from models.pidnet import (
    get_pidnet_model,
    load_pretrained_pt_file
)

if __name__ == "__main__":
    pidnet_model = get_pidnet_model("medium", 10)
    pt_file_path_str = "../pretrained_models/best_mizba.pt"
    model = load_pretrained_pt_file(pidnet_model, pt_file_path_str)
    # model.parameters():
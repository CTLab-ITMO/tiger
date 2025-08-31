from modeling.utils import generate_inter_json, create_logger

logger = create_logger(name=__name__)

def main():
    all_data_path = "../data/Beauty/all_data.txt"
    output_dir = "../data/Beauty"
    result_path = generate_inter_json(all_data_path, output_dir)
    logger.info(f"inter json file was successfully generated in {result_path}")

if __name__ == '__main__':
    main()